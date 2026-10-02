"""
shared_fitting.py
=================
Shared functions for real-space model-map evaluation.
Used by 04_fitting_bioemu.py, 05_fitting_msa.py, 06_fitting_alphaflow.py.
"""

import os
import copy
import numpy as np
import pandas as pd
import mdtraj as md
import gemmi
import urllib.request
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from Bio import pairwise2
import diptest
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EMDB_DIR     = 'emdb_maps'
PDB_DIR      = 'pdb_structures'
DATASET      = 'data/dataset_multistate_v1.xlsx'
MAX_FRAMES       = 500
MIN_CLUSTER_FRAC = 0.03  # a cluster must hold >=3% of frames to count as a
                         # distinct conformational family (else = noise)
MIN_CLUSTER_ABS  = 3     # ...and at least this many frames regardless of %
DIP_ALPHA        = 0.05  # Hartigan's dip-test significance level: below this,
                         # reject unimodality and search for sub-populations

os.makedirs(EMDB_DIR, exist_ok=True)
os.makedirs(PDB_DIR,  exist_ok=True)


# ── download helpers ──────────────────────────────────────────────────────────
def download_pdb(pdb_id):
    for fmt in ['pdb', 'cif']:
        p = f'{PDB_DIR}/{pdb_id}.{fmt}'
        if os.path.exists(p):
            return p
        try:
            urllib.request.urlretrieve(
                f'https://files.rcsb.org/download/{pdb_id}.{fmt}', p)
            print(f'  Downloaded {pdb_id}.{fmt}')
            return p
        except:
            continue
    return None


def download_map(emdb_id):
    path = f'{EMDB_DIR}/{emdb_id}.map.gz'
    if os.path.exists(path):
        return path
    num = emdb_id.replace('EMD-', '')
    url = (f'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/'
           f'EMD-{num}/map/emd_{num}.map.gz')
    try:
        urllib.request.urlretrieve(url, path)
        print(f'  Downloaded {emdb_id}')
        return path
    except Exception as e:
        print(f'  FAILED {emdb_id}: {e}')
        return None


# ── map loading: light Gaussian pre-smoothing + z-score normalisation ────────
def load_map(path, resolution, smooth_frac=1 / 3):
    """
    Load a CCP4/MRC map, lightly Gaussian-smooth it, then z-score normalise.

    The smoothing is what makes point-sampled density reads (see
    mean_density_at_backbone) robust to small backbone positioning noise in
    predicted conformations -- without it, a fraction-of-an-Angstrom shift
    in a predicted atom's position can land it on a sharp noise spike vs a
    neighbouring low-density voxel, producing a large score swing that has
    nothing to do with whether the conformation is actually right. Smoothing
    only the (fixed, shared-across-all-frames) experimental map is far
    cheaper than generating a per-frame simulated density for every one of
    up to 500 predicted conformations, and gives the same practical benefit
    at the read-out step.

    sigma = resolution * smooth_frac (Angstrom), converted to voxel units
    via the map's own spacing. smooth_frac=1/3 is a standard rule-of-thumb
    relating nominal resolution to a real-space Gaussian blur width; this is
    a tunable parameter, not a fixed constant from theory.
    """
    ccp4 = gemmi.read_ccp4_map(path)
    ccp4.setup(float('nan'))
    grid = ccp4.grid
    arr  = np.array(grid, copy=False)

    arr_filled = np.nan_to_num(arr, nan=0.0)
    sigma_voxels = (resolution * smooth_frac) / np.array(grid.spacing)
    smoothed = gaussian_filter(arr_filled, sigma=sigma_voxels, mode='nearest')

    mean = float(np.mean(smoothed))
    std  = float(np.std(smoothed))
    if std > 0:
        smoothed = (smoothed - mean) / std
    arr[:] = smoothed  # writes through into `grid` (arr is a view, not a copy)
    return grid


# ── real-space interpolation ──────────────────────────────────────────────────
def mean_density_at_backbone(traj_frame_nm, bb_idx, grid):
    """
    Interpolate experimental map density at backbone atom positions.
    traj_frame_nm : (N_atoms, 3) in nm (mdtraj convention)
    bb_idx        : indices of backbone atoms (N, CA, C, O)
    Returns mean density value over backbone atoms.
    """
    coords = traj_frame_nm[bb_idx]
    total  = 0.0
    for c in coords:
        pos = gemmi.Position(float(c[0]*10), float(c[1]*10), float(c[2]*10))
        total += grid.interpolate_value(pos)
    return total / len(coords)


# ── superposition onto target PDB via sequence-aligned common Cα ─────────────
# Previously matched Cα atoms by comparing resSeq numbers directly (set
# intersection of author-numbering labels). That breaks whenever the
# ensemble's topology and the target PDB's numbering don't line up 1:1 --
# e.g. an N-terminal offset or an unresolved internal loop on the target
# side shifts the numbering out of registration, silently pairing the
# wrong atoms together (confirmed to badly misalign SPNS2 against
# 8ex5/8ex7). Fixed by finding the correspondence via actual sequence
# alignment instead of numeric labels -- validated in
# outputs/fig_revision/spns2_refit_v2.py before being merged in here.
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'MSE': 'M',
}


def _ca_list(top):
    """Ordered list of (residue, ca_atom_index) for protein residues, in
    chain (N->C) order -- NOT keyed/sorted by resSeq number."""
    out = []
    for r in top.residues:
        if not r.is_protein:
            continue
        ca = next((a.index for a in r.atoms if a.name == 'CA'), None)
        if ca is not None:
            out.append((r, ca))
    return out


def seqaln_common_ca(traj_top, target_top):
    """
    Find true residue correspondence via gapped global sequence alignment
    (Biopython), rather than comparing author-numbering resSeq values.
    Only aligned columns where both sides carry the same residue identity
    are kept. This dataset's ensembles and reference PDBs are all 100%
    sequence-identical, so this cleanly recovers the full common region,
    correctly skipping any gaps from unresolved loops on the reference
    side, regardless of numbering offsets.
    """
    traj_list   = _ca_list(traj_top)
    target_list = _ca_list(target_top)
    tseq = ''.join(AA3TO1.get(r.name, 'X') for r, _ in traj_list)
    rseq = ''.join(AA3TO1.get(r.name, 'X') for r, _ in target_list)
    aln = pairwise2.align.globalms(tseq, rseq, 2, -1, -5, -0.5,
                                    one_alignment_only=True)[0]
    a_t, a_r = aln.seqA, aln.seqB
    ti = ri = 0
    traj_idx, target_idx = [], []
    for ct, cr in zip(a_t, a_r):
        if ct != '-' and cr != '-' and ct == cr:
            traj_idx.append(traj_list[ti][1])
            target_idx.append(target_list[ri][1])
        if ct != '-':
            ti += 1
        if cr != '-':
            ri += 1
    return np.array(traj_idx), np.array(target_idx)


def superpose_to_target(traj, target_pdb_path):
    """
    Superpose every frame of `traj` onto `target_pdb_path` using a
    sequence-aligned common-Cα correspondence (handles missing-residue
    and numbering-offset mismatches). Modifies traj in-place.
    """
    target = md.load(target_pdb_path)
    traj_idx, target_idx = seqaln_common_ca(traj.topology, target.topology)
    if len(traj_idx) < 10:
        raise ValueError(f'Only {len(traj_idx)} common Cα residues (seq-aligned)')
    traj.superpose(target, atom_indices=traj_idx, ref_atom_indices=target_idx)
    return traj


# ── pairwise Cα-RMSD ─────────────────────────────────────────────────────────
def pairwise_ca_rmsd(traj):
    ca  = traj.topology.select('name CA')
    xyz = traj.xyz[:, ca, :]
    n   = len(xyz)
    mat = np.zeros((n, n))
    for i in range(n):
        d      = xyz - xyz[i]
        mat[i] = np.sqrt(np.mean((d**2).sum(axis=2), axis=1)) * 10
    return mat


# ── GROMOS/Daura neighbour-counting clustering ────────────────────────────────
def cluster_daura(rmsd_mat, cutoff):
    """
    GROMOS/Daura clustering (Daura et al. 1999) — the field-standard method
    for clustering MD/conformational ensembles by pairwise RMSD, as
    implemented in GROMACS `gmx cluster`. Cluster count is not chosen by
    maximising a global separation score (that approach — Ward + argmax
    silhouette — was found to be biased: it rewards peeling off one or two
    far outliers into their own tiny clusters, and *penalises* real but
    closely-spaced sub-populations, since splitting them lowers average
    silhouette even when the split is genuine). Instead, cluster count
    emerges from local neighbour density:
      - every frame within `cutoff` Å of a candidate centre is a neighbour
      - repeatedly take the remaining frame with the most neighbours as a
        new cluster centre, assign it + its neighbours, remove them, repeat
    Returns (labels, n_clusters); clusters ordered largest first.
    """
    n = rmsd_mat.shape[0]
    adj = rmsd_mat <= cutoff
    remaining = set(range(n))
    clusters = []
    while remaining:
        rem_idx = np.array(sorted(remaining))
        sub = adj[np.ix_(rem_idx, rem_idx)]
        counts = sub.sum(axis=1)
        center_local = np.argmax(counts)
        members = rem_idx[sub[center_local]]
        clusters.append(set(members.tolist()))
        remaining -= set(members.tolist())
    clusters.sort(key=len, reverse=True)
    labels = np.empty(n, dtype=int)
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return labels, len(clusters)


def select_daura_cutoff(rmsd_mat):
    """
    Choose the RMSD cutoff for cluster_daura from the data itself, rather
    than a fixed value. Standard GROMOS practice is to scan the cutoff and
    inspect how clustering changes; here the scan is automated in two
    stages:

    1. Hartigan's dip test (Hartigan & Hartigan 1985) on the pairwise-RMSD
       sample tests H0: the distribution is unimodal. If we fail to reject
       H0 (p >= DIP_ALPHA), there is no statistical evidence of more than
       one conformational family and we stop here: n_clusters=1. This gate
       matters because Daura clustering *always* finds "clusters" once you
       hand it a cutoff, even in genuinely unimodal data (it just carves
       the densest region into a big cluster + stragglers) -- so cluster
       count must never be decided by clustering alone without first
       asking whether the underlying distribution supports multimodality.
    2. Only if the dip test rejects unimodality do we search for the
       cutoff: local minima ("valleys") of the pairwise-RMSD density,
       scanned across several KDE bandwidths (Scott's rule down to a much
       narrower one) since a single default bandwidth can over-smooth
       subtle-but-real valleys away.

    Selection rule (transparent / reportable for methods write-up): among
    candidate cutoffs, pick the one giving the most *significant* clusters
    (>= MIN_CLUSTER_FRAC of frames, and >= MIN_CLUSTER_ABS); break ties by
    the cutoff with the highest fraction of frames covered by a significant
    cluster ("coverage"); further ties -> smallest cutoff (tighter
    definition of "same state"). Frames that fall outside every significant
    cluster are reassigned to the nearest significant cluster (by mean RMSD
    to its members) so every frame still gets a label. If no candidate
    cutoff yields >=2 significant clusters, the ensemble is still reported
    as structurally unimodal (n_clusters=1).
    """
    n = rmsd_mat.shape[0]
    min_size = max(MIN_CLUSTER_ABS, int(np.ceil(MIN_CLUSTER_FRAC * n)))
    vals = rmsd_mat[np.triu_indices(n, k=1)]

    dip, dip_p = diptest.diptest(vals)
    if dip_p >= DIP_ALPHA:
        return dict(cutoff=None, n_clusters=1, labels=np.zeros(n, dtype=int),
                    min_size=min_size, dip_p=float(dip_p), scan=[])

    xs = np.linspace(vals.min(), vals.max(), 400)
    candidates = set()
    for bw in [None, 0.3, 0.15, 0.08, 0.05]:  # None = Scott's rule (default)
        density = gaussian_kde(vals, bw_method=bw)(xs)
        valley_idx, _ = find_peaks(-density, prominence=density.max() * 0.02)
        candidates.update(np.round(xs[valley_idx], 3).tolist())
    if not candidates:
        candidates = {float(np.median(vals))}
    candidates = np.array(sorted(candidates))

    scan = []
    for cutoff in candidates:
        lbl, nk = cluster_daura(rmsd_mat, cutoff)
        sizes = np.bincount(lbl)
        sig_sizes = sizes[sizes >= min_size]
        scan.append(dict(cutoff=float(cutoff), n_clusters=nk,
                         n_significant=len(sig_sizes),
                         coverage=float(sig_sizes.sum() / n),
                         labels=lbl, sizes=sizes))

    scan.sort(key=lambda r: (-r['n_significant'], -r['coverage'], r['cutoff']))
    best = scan[0]
    raw_labels, sizes = best['labels'], best['sizes']
    # lightweight log for reporting/methods write-up (no per-frame arrays)
    scan_log = [{k: v for k, v in r.items() if k not in ('labels', 'sizes')}
               for r in scan]

    if best['n_significant'] < 2:
        return dict(cutoff=None, n_clusters=1, labels=np.zeros(n, dtype=int),
                    min_size=min_size, dip_p=float(dip_p), scan=scan_log)

    sig_old_ids = sorted([i for i in range(len(sizes)) if sizes[i] >= min_size],
                         key=lambda i: -sizes[i])
    remap = {old: new for new, old in enumerate(sig_old_ids)}
    member_idx = {old: np.where(raw_labels == old)[0] for old in sig_old_ids}

    final_labels = np.empty(n, dtype=int)
    for f in range(n):
        old = raw_labels[f]
        if old in remap:
            final_labels[f] = remap[old]
        else:
            dists = {new: rmsd_mat[f, member_idx[old2]].mean()
                     for old2, new in remap.items()}
            final_labels[f] = min(dists, key=dists.get)

    return dict(cutoff=best['cutoff'], n_clusters=len(sig_old_ids),
               labels=final_labels, min_size=min_size, dip_p=float(dip_p),
               scan=scan_log)


# ── load experimental maps + compute CC_ref ───────────────────────────────────
def load_exp_data(emdb_list):
    """
    For each (emdb_id, pdb_id, resolution), download map and PDB,
    compute CC_ref (deposited PDB vs its own map).
    Returns dict keyed by emdb_id.
    """
    exp_data = {}
    for emdb_id, pdb_id, resolution in emdb_list:
        map_path = download_map(emdb_id)
        if map_path is None: continue
        pdb_path = download_pdb(pdb_id)
        if pdb_path is None: continue
        try:
            grid       = load_map(map_path, resolution)
            ref_struct = md.load(pdb_path)
            ref_bb_idx = ref_struct.topology.select('name N CA C O')
            cc_ref     = mean_density_at_backbone(
                             ref_struct.xyz[0], ref_bb_idx, grid)
            # CC_REF_MIN used to skip any map whose cc_ref fell below a fixed
            # threshold -- but after the earlier z-score-normalisation fix,
            # real cc_ref values are consistently ~10-40, so 0.1 never
            # actually filtered anything meaningful. Only guard against a
            # genuine computation failure (non-positive/NaN), and don't
            # silently drop real (if low) values.
            if not np.isfinite(cc_ref) or cc_ref <= 0:
                print(f'  {emdb_id}: CC_ref={cc_ref} -> invalid, skipping')
                continue
            print(f'  {emdb_id}: CC_ref={cc_ref:.4f}')
            exp_data[emdb_id] = dict(grid=grid, cc_ref=cc_ref,
                                     pdb_id=pdb_id, pdb_path=pdb_path)
        except Exception as e:
            print(f'  Error {emdb_id}: {e}')
    return exp_data


# ── score matrix (n_frames × n_maps) ─────────────────────────────────────────
def compute_score_matrix(traj_raw, exp_data, emdb_ids):
    """
    For each experimental map, superpose traj onto that map's PDB,
    then interpolate density at backbone positions.
    Returns score_mat (n_frames × n_maps).
    """
    n_frames  = traj_raw.n_frames
    n_maps    = len(emdb_ids)
    score_mat = np.zeros((n_frames, n_maps))

    for j, eid in enumerate(emdb_ids):
        d = exp_data[eid]
        print(f'\n  Superposing onto {d["pdb_id"]} for {eid}...')
        traj_j = copy.deepcopy(traj_raw)
        try:
            superpose_to_target(traj_j, d['pdb_path'])
        except Exception as e:
            print(f'  Superpose failed: {e}'); continue

        bb_idx = traj_j.topology.select('name N CA C O')
        print(f'  Interpolating {n_frames} frames in {eid}...')
        for f in range(n_frames):
            if f % 100 == 0:
                print(f'    frame {f}/{n_frames}')
            raw = mean_density_at_backbone(traj_j.xyz[f], bb_idx, d['grid'])
            score_mat[f, j] = raw / d['cc_ref'] if d['cc_ref'] > 0 else 0.0

    return score_mat


# NOTE: the experimental cross-comparison matrix (PDB_i vs map_j) now lives
# in the standalone 00_cross_comparison.py, using real CCmask (Phenix-style
# masked Pearson correlation with actual B-factors via gemmi's
# DensityCalculatorE), not the interpolation-ratio method below. The old
# compute_cross_matrix() reused the no-B-factor interpolation method (meant
# for BioEmu conformations, which have no B-factors) and hard-coded the
# diagonal to 1.0 -- that was a bug, not a design choice; see 00_cross_comparison.py.


# ── figures ───────────────────────────────────────────────────────────────────
def make_figures(protein, n_exp, score_mat, labels, csizes, best_k, best_sil,
                 rmsd_vals, cl_score, short_ids, out_dir,
                 model_name=None, daura_cutoff=None, dip_test_p=None):

    n_maps = len(short_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, max(best_k, 2)))

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A: Pairwise RMSD histogram
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(rmsd_vals, bins=60, color='steelblue',
              edgecolor='black', linewidth=0.4, alpha=0.85)
    ax_a.set_xlabel('Pairwise Cα-RMSD (Å)')
    ax_a.set_ylabel('Count')
    ax_a.set_title('A. Pairwise RMSD distribution', fontweight='bold')

    # B: Cluster × map histogram grid
    all_vals = score_mat.flatten()
    xmin = max(0,   float(np.nanmin(all_vals)) - 0.05)
    xmax = min(1.3, float(np.nanmax(all_vals)) + 0.05)
    inner = gridspec.GridSpecFromSubplotSpec(
        best_k, n_maps, subplot_spec=gs[0, 1], hspace=0.5, wspace=0.3)
    for ci in range(best_k):
        mask = labels == ci
        for mi in range(n_maps):
            ax_g = fig.add_subplot(inner[ci, mi])
            ax_g.hist(score_mat[mask, mi], bins=12, color=colors[ci],
                      alpha=0.8, edgecolor='black', linewidth=0.3)
            ax_g.set_xlim(xmin, xmax)
            ax_g.tick_params(labelsize=5)
            ax_g.set_yticks([])
            if ci == 0:
                ax_g.set_title(short_ids[mi], fontsize=6, pad=1)
            if mi == 0:
                ax_g.set_ylabel(f'C{ci}', fontsize=6, rotation=0,
                                labelpad=12, va='center')
    pos_b = gs[0, 1].get_position(fig)
    fig.text(pos_b.x0 + pos_b.width / 2, pos_b.y1 + 0.012,
             'B. Density ratio distributions (cluster × exp map)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

    # C: Scatter (n_maps==2) or PCA scatter (n_maps>2)
    ax_d = fig.add_subplot(gs[0, 2])
    if n_maps == 2:
        for c in range(best_k):
            mask = labels == c
            ax_d.scatter(score_mat[mask, 0], score_mat[mask, 1],
                         c=[colors[c]], alpha=0.4, s=12,
                         label=f'C{c}(n={csizes[c]})')
        lim = [xmin, xmax]
        ax_d.plot(lim, lim, 'k--', linewidth=0.7, alpha=0.4)
        ax_d.set_xlabel(f'Density ratio – {short_ids[0]}')
        ax_d.set_ylabel(f'Density ratio – {short_ids[1]}')
        ax_d.set_title('C. Scatter: map A vs map B\n(coloured by cluster)',
                       fontweight='bold')
        ax_d.legend(fontsize=7, markerscale=1.5)
    else:
        pca      = PCA(n_components=2)
        coords2d = pca.fit_transform(score_mat)
        for c in range(best_k):
            mask = labels == c
            ax_d.scatter(coords2d[mask, 0], coords2d[mask, 1],
                         c=[colors[c]], alpha=0.4, s=12,
                         label=f'C{c}(n={csizes[c]})')
        ax_d.set_xlabel(
            f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
        ax_d.set_ylabel(
            f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
        ax_d.set_title('C. PCA of density ratios\n(coloured by cluster)',
                       fontweight='bold')
        ax_d.legend(fontsize=7)

    # D: Density ratio histogram per exp map
    ax_e = fig.add_subplot(gs[1, 0])
    bins = np.linspace(xmin, xmax, 30)
    for mi in range(n_maps):
        ax_e.hist(score_mat[:, mi], bins=bins, alpha=0.5,
                  label=short_ids[mi], edgecolor='none')
    ax_e.set_xlabel('Density ratio')
    ax_e.set_ylabel('Count')
    ax_e.set_title('D. Density ratio distribution\n'
                   '(all conformations, per exp map)', fontweight='bold')
    ax_e.legend(fontsize=7)

    # E: Heatmap cluster × map
    ax_f = fig.add_subplot(gs[1, 1])
    im_f = ax_f.imshow(cl_score, cmap='RdYlGn', vmin=0, vmax=1.0,
                        aspect='auto')
    ax_f.set_xticks(range(n_maps))
    ax_f.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=8)
    ax_f.set_yticks(range(best_k))
    ax_f.set_yticklabels([f'C{i}(n={csizes[i]})' for i in range(best_k)],
                          fontsize=8)
    ax_f.set_xlabel('Experimental map')
    ax_f.set_ylabel('Predicted cluster')
    ax_f.set_title('E. Mean density ratio\n(cluster × experimental map)',
                   fontweight='bold')
    plt.colorbar(im_f, ax=ax_f, label='Density ratio', shrink=0.85)
    for i in range(best_k):
        for j in range(n_maps):
            ax_f.text(j, i, f'{cl_score[i,j]:.2f}',
                      ha='center', va='center', fontsize=8)

    # Summary table (fills the otherwise-empty bottom-right cell)
    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.axis('off')
    ax_s.set_title('Summary', fontweight='bold', loc='left')
    n_frames   = score_mat.shape[0]
    best_idx   = np.unravel_index(np.nanargmax(score_mat), score_mat.shape)
    best_ratio = float(score_mat[best_idx])
    best_map   = short_ids[best_idx[1]]
    cutoff_str = f'{daura_cutoff:.2f} Å' if daura_cutoff is not None else 'n/a'
    dip_str    = f'{dip_test_p:.4f}' if dip_test_p is not None else 'n/a'

    rows = [
        ['Model',              model_name or 'n/a'],
        ['Daura cutoff',       cutoff_str],
        ['Dip test p',         dip_str],
        ['n_frames',           str(n_frames)],
        ['Best density_ratio', f'{best_ratio:.2f}'],
        ['Best frame',         f'{best_idx[0]} (EMD-{best_map})'],
    ]
    tbl = ax_s.table(cellText=rows, colWidths=[0.4, 0.6],
                      cellLoc='left', loc='upper left', bbox=[0, 0.15, 1, 0.8])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('none')
        cell.PAD = 0.02
        if c == 0:
            cell.set_text_props(fontweight='bold')

    title = f'{protein}: {best_k} cluster(s) vs {n_exp} experimental states'
    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig_path = f'{out_dir}/{protein}_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {fig_path}')


# ── main analysis loop (model-agnostic) ───────────────────────────────────────
def run_analysis(load_ensemble_fn, model_name, out_dir):
    """
    Core analysis loop.
    load_ensemble_fn(protein) → mdtraj.Trajectory or None
    model_name: string label for output files ('bioemu', 'msa', 'alphaflow')
    out_dir: output directory
    (Experimental cross-comparison is standalone now -- see
    00_cross_comparison.py -- it doesn't depend on any predicted ensemble.)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    df       = pd.read_excel(DATASET)
    proteins = df['protein_label'].unique()
    all_rows       = []
    all_frame_rows = []

    for protein in proteins:
        print(f'\n{"="*60}\nProtein: {protein}\n{"="*60}')

        prot_rows = df[df['protein_label'] == protein]
        emdb_list = prot_rows[['emdb_id', 'pdb_id', 'resolution']].values
        n_exp     = len(emdb_list)
        print(f'  Experimental states: {n_exp}')

        traj_raw = load_ensemble_fn(protein)
        if traj_raw is None:
            print('  No ensemble results, skipping'); continue
        if traj_raw.n_frames > MAX_FRAMES:
            idx      = np.random.choice(traj_raw.n_frames, MAX_FRAMES,
                                        replace=False)
            traj_raw = traj_raw.slice(idx)
        print(f'  Loaded {traj_raw.n_frames} frames, {traj_raw.n_atoms} atoms')
        n_frames = traj_raw.n_frames

        exp_data = load_exp_data(emdb_list)
        if not exp_data:
            print('  No valid maps, skipping'); continue

        emdb_ids  = list(exp_data.keys())
        short_ids = [e.replace('EMD-', '') for e in emdb_ids]
        n_maps    = len(emdb_ids)

        score_mat = compute_score_matrix(traj_raw, exp_data, emdb_ids)

        # clustering on first map's superposed traj
        traj_cl = copy.deepcopy(traj_raw)
        try:
            superpose_to_target(traj_cl, exp_data[emdb_ids[0]]['pdb_path'])
        except Exception as e:
            print(f'  Cannot superpose for clustering: {e}'); continue

        print(f'\n  Computing pairwise Cα-RMSD...')
        rmsd_mat  = pairwise_ca_rmsd(traj_cl)
        rmsd_vals = rmsd_mat[np.triu_indices(n_frames, k=1)]

        daura   = select_daura_cutoff(rmsd_mat)
        best_k  = daura['n_clusters']
        labels  = daura['labels']
        csizes  = np.bincount(labels, minlength=best_k)
        # silhouette of the resulting partition, QC/reporting only -- NOT
        # used to choose k (see cluster_daura/select_daura_cutoff docstrings
        # for why argmax-silhouette was dropped as the selection criterion)
        best_sil = (silhouette_score(rmsd_mat, labels, metric='precomputed',
                                     sample_size=min(500, len(labels)),
                                     random_state=0)
                   if best_k >= 2 else float('nan'))
        cutoff_str = f"{daura['cutoff']:.2f}" if daura['cutoff'] is not None else 'n/a'
        print(f"  Dip test for unimodality: p={daura['dip_p']:.4f} "
             f"({'reject unimodal -> searching for sub-populations' if daura['dip_p'] < DIP_ALPHA else 'fail to reject unimodal -> k=1, no cutoff search'})")
        print(f"  Daura clustering: cutoff={cutoff_str}A  min_size={daura['min_size']}  "
             f'k={best_k}  sizes={csizes.tolist()}  silhouette(QC)={best_sil:.3f}')
        if daura['scan']:
            print('  cutoff scan:',
                 pd.DataFrame(daura['scan']).round(3).to_string(index=False))

        cl_score = np.zeros((best_k, n_maps))
        for c in range(best_k):
            cl_score[c] = score_mat[labels == c].mean(axis=0)

        print(pd.DataFrame(cl_score,
                           index=[f'C{i}(n={csizes[i]})' for i in range(best_k)],
                           columns=emdb_ids).round(4).to_string())



        make_figures(protein, n_exp, score_mat, labels, csizes, best_k,
                     best_sil, rmsd_vals, cl_score, short_ids, out_dir,
                     model_name=model_name, daura_cutoff=daura['cutoff'],
                     dip_test_p=daura['dip_p'])

        for f in range(n_frames):
            for j, eid in enumerate(emdb_ids):
                all_frame_rows.append(dict(
                    protein=protein, frame=f,
                    emdb_id=eid,
                    density_ratio=score_mat[f, j],
                    cluster=int(labels[f])))

        for c in range(best_k):
            for j, eid in enumerate(emdb_ids):
                all_rows.append(dict(
                    protein=protein,
                    model=model_name,
                    n_pred_clusters=best_k,
                    daura_cutoff=daura['cutoff'],
                    dip_test_p=daura['dip_p'],
                    silhouette_qc=round(best_sil, 4) if best_k >= 2 else np.nan,
                    n_exp_states=n_exp,
                    cluster_id=c,
                    cluster_size=int(csizes[c]),
                    emdb_id=eid,
                    cc_ref=exp_data[eid]['cc_ref'],
                    mean_density_ratio=cl_score[c, j]))

    pd.DataFrame(all_rows).to_excel(
        f'{out_dir}/clustering_density_results.xlsx', index=False)
    pd.DataFrame(all_frame_rows).to_excel(
        f'{out_dir}/per_frame_density.xlsx', index=False)
    print(f'\nResults saved to {out_dir}/')

    print('\n' + '='*60 + '\nSUMMARY\n' + '='*60)
    res = pd.DataFrame(all_rows)
    if not res.empty:
        for protein in proteins:
            sub = res[res['protein'] == protein]
            if sub.empty: continue
            n_pred = sub['n_pred_clusters'].iloc[0]
            cutoff = sub['daura_cutoff'].iloc[0]
            n_exp  = sub['n_exp_states'].iloc[0]
            best   = sub.groupby('emdb_id')['mean_density_ratio'].max()
            cov    = (best > 0.7).sum()
            dip_p  = sub['dip_test_p'].iloc[0]
            if n_pred > 1:
                note = f'meaningful (Daura cutoff={cutoff:.2f}A, dip p={dip_p:.4f})'
            elif dip_p >= DIP_ALPHA:
                note = f'unimodal (dip test p={dip_p:.4f} >= {DIP_ALPHA})'
            else:
                note = (f'dip test rejects unimodal (p={dip_p:.4f}) but no '
                        f'cutoff gave >=2 significant (>={MIN_CLUSTER_FRAC:.0%}) '
                        f'clusters -> reported as k=1')
            print(f'{protein}: {n_pred} cluster(s) [{note}], '
                  f'{n_exp} exp states, {cov}/{n_exp} covered (ratio>0.7)')