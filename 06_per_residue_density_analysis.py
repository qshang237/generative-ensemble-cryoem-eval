# -*- coding: utf-8 -*-
"""
06_per_residue_density_analysis.py
===================================
Generates the per-residue backbone-density panels used in the manuscript's
best-frame figures (Figures 3, 5, 7, 9, 11: GltPh Intermediate, GPR4 pH 6.2,
AUX1 Lab 1, SLC37A4 cytosol-open, SPNS2 Outward-occluded), plus the same
panel for every other (protein, map) combination in the benchmark.

Two metrics appear in these figures and this script keeps them explicitly
distinct (see manuscript Methods):

  - density ratio (scalar, per frame per map): already computed by
    04_fitting_bioemu.py / 04_fitting_msa.py / 04_fitting_alphaflow.py via
    shared_fitting.compute_score_matrix, and stored in
    results/{method}/per_frame_density.xlsx. This script only READS that
    file to pick each method's best-scoring frame per map -- it does not
    recompute density ratio.

  - per-residue backbone-density difference (vector, one best frame vs. the
    deposited reference, same map): computed here. Both the generated
    frame's and the reference structure's per-residue backbone density are
    read from the SAME whole-map z-scored grid (shared_fitting.load_map),
    with NO cc_ref (reference-fit) normalisation applied -- deliberately
    different scaling from density ratio. Units are z-scored map-density
    units. This is a local, within-map diagnostic; per Methods, its
    absolute magnitude is not compared across proteins/maps (global map
    z-score statistics vary by up to ~100x across this benchmark's 15 maps,
    dominated by map-specific box geometry/processing -- see
    analysis_qc/map_geometry_qc.py).

Frame selection: for each (protein, method, map), the best-scoring frame is
read directly from the current production results/{method}/per_frame_density.xlsx
(density_ratio already reflects shared_fitting.superpose_to_target's
sequence-alignment-based superposition -- no separate "corrected" pass is
needed here, unlike earlier drafts of this analysis).

Usage:
    python 06_per_residue_density_analysis.py [PROTEIN ...]
    (defaults to all five proteins; each protein's every deposited map is
    processed)

Outputs, per (protein, map):
    figures/per_residue/{protein}_{emdb_id}_bestframe_diff.png
    figures/per_residue/{protein}_{emdb_id}_bestframe_diff.xlsx
        (per-residue profiles for bioemu/alphaflow/msa/reference, plus the
        summary mean/s.d. per method quoted in Results/captions)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import mdtraj as md
import gemmi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_fitting import (
    DATASET, load_exp_data, mean_density_at_backbone, superpose_to_target,
    seqaln_common_ca,
)

OUT_DIR = 'figures/per_residue'
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {'bioemu': '#0000FF', 'alphaflow': '#FF9500', 'msa': '#34C759'}
LW = {'bioemu': 2.0, 'alphaflow': 1.2, 'msa': 1.2}
RESULTS_DIR = {'bioemu': 'results/bioemu', 'alphaflow': 'results/alphaflow', 'msa': 'results/msa'}


def load_bioemu(protein):
    return md.load(f'bioemu_results/{protein}/samples.xtc',
                    top=f'bioemu_results/{protein}/topology.pdb')


def load_alphaflow(protein):
    return md.load(f'alphaflow_results/{protein}.pdb')


def load_msa(protein):
    pdbs = sorted(glob.glob(f'msa_results/{protein}/{protein}_unrelaxed_rank_*.pdb'))
    ref = md.load(pdbs[0])
    n_atoms = ref.n_atoms
    xyz_list = [ref.xyz[0]]
    for p in pdbs[1:]:
        t = md.load(p)
        if t.n_atoms == n_atoms:
            xyz_list.append(t.xyz[0])
    return md.Trajectory(np.stack(xyz_list, axis=0), ref.topology)


LOADERS = {'bioemu': load_bioemu, 'alphaflow': load_alphaflow, 'msa': load_msa}


def bb_idx_and_resmap(top):
    """Backbone (N, CA, C, O) atom indices, plus each one's parent-residue
    index, for grouping per-atom density reads into per-residue values."""
    bb_idx = top.select('name N CA C O')
    res_of_atom = np.array([top.atom(i).residue.index for i in bb_idx])
    n_res = top.n_residues
    return bb_idx, res_of_atom, n_res


def per_residue_density(frame_xyz_nm, bb_idx, res_of_atom, n_res, grid):
    """Interpolate the (already Gaussian-smoothed, whole-map z-scored) map
    density at each backbone atom, then average the ~4 backbone atoms within
    each residue. No cc_ref normalisation -- see module docstring. Returns
    an (n_res,) array; residues with no backbone atoms found are NaN."""
    sums = np.zeros(n_res)
    counts = np.zeros(n_res)
    coords = frame_xyz_nm[bb_idx]
    for c, r in zip(coords, res_of_atom):
        pos = gemmi.Position(float(c[0] * 10), float(c[1] * 10), float(c[2] * 10))
        sums[r] += grid.interpolate_value(pos)
        counts[r] += 1
    out = np.full(n_res, np.nan)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def best_frame_profile(protein, method, emdb_id, pdb_path, grid):
    """Read the best-scoring frame for this (protein, method, map) from the
    current production per_frame_density.xlsx, superpose just that one frame
    with shared_fitting.superpose_to_target (sequence-aligned), and return
    its per-residue backbone-density profile."""
    pf = pd.read_excel(f'{RESULTS_DIR[method]}/per_frame_density.xlsx')
    g = pf[(pf['protein'] == protein) & (pf['emdb_id'] == emdb_id)]
    row = g.loc[g['density_ratio'].idxmax()]
    fi, ratio = int(row['frame']), float(row['density_ratio'])

    traj_full = LOADERS[method](protein)
    traj1 = traj_full[fi]
    traj1 = superpose_to_target(traj1, pdb_path)

    bb_idx, res_of_atom, n_res = bb_idx_and_resmap(traj1.topology)
    profile = per_residue_density(traj1.xyz[0], bb_idx, res_of_atom, n_res, grid)
    return profile, fi, ratio, traj1.topology


def reference_profile_on_shared_axis(pred_top, ref_pdb_path, grid, n_res_pred):
    """Deposited reference structure's own per-residue backbone density on
    this map, remapped onto the generated ensemble's residue-index axis via
    sequence alignment (reference structures often have unresolved
    termini/loops, so a naive index alignment would misregister this
    curve)."""
    ref = md.load(ref_pdb_path)
    traj_idx, target_idx = seqaln_common_ca(pred_top, ref.topology)
    ref_bb_idx, ref_res_of_atom, ref_n_res = bb_idx_and_resmap(ref.topology)
    ref_profile = per_residue_density(ref.xyz[0], ref_bb_idx, ref_res_of_atom, ref_n_res, grid)

    out = np.full(n_res_pred, np.nan)
    for ta, ra in zip(traj_idx, target_idx):
        pred_res = pred_top.atom(int(ta)).residue.index
        ref_res = ref.topology.atom(int(ra)).residue.index
        out[pred_res] = ref_profile[ref_res]
    return out, len(traj_idx)


def build_bestframe_diff_figure(protein, emdb_id, pdb_id, resolution):
    print(f'=== {protein} / {emdb_id} ({pdb_id}) ===')
    exp = load_exp_data([(emdb_id, pdb_id, resolution)])[emdb_id]
    grid, pdb_path = exp['grid'], exp['pdb_path']

    profiles, frame_info, shared_top = {}, {}, None
    for method in ['bioemu', 'alphaflow', 'msa']:
        profile, fi, ratio, top = best_frame_profile(protein, method, emdb_id, pdb_path, grid)
        profiles[method] = profile
        frame_info[method] = (fi, ratio)
        if shared_top is None:
            shared_top = top
        print(f'    {method}: best frame {fi}, density_ratio={ratio:.3f}')

    n_res = len(next(iter(profiles.values())))
    ref_prof, n_common_ref = reference_profile_on_shared_axis(shared_top, pdb_path, grid, n_res)
    print(f'    reference {pdb_id}: seq-aligned common CA with predicted topology = {n_common_ref}/{n_res}')

    diffs = {m: profiles[m] - ref_prof for m in ['bioemu', 'alphaflow', 'msa']}
    summary = {m: (float(np.nanmean(diffs[m])), float(np.nanstd(diffs[m]))) for m in diffs}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    for method in ['bioemu', 'alphaflow', 'msa']:
        fi, ratio = frame_info[method]
        ax1.plot(np.arange(n_res), profiles[method], color=COLORS[method], linewidth=LW[method],
                  label=f'{method} (best frame {fi}, density_ratio={ratio:.2f})',
                  zorder=3 if method == 'bioemu' else 2)
    ax1.plot(np.arange(n_res), ref_prof, color='black', linewidth=1.8, linestyle='--',
              label=f'reference ({pdb_id})')
    ax1.set_ylabel('Backbone density\n(z-scored map units)', fontsize=13)
    ax1.set_title(f'{protein} ({emdb_id}) — best-scoring frame per method vs. reference', fontsize=14)
    ax1.axhline(0, color='gray', linestyle=':', linewidth=0.7)
    ax1.legend(fontsize=10, loc='upper right')

    for method in ['bioemu', 'alphaflow', 'msa']:
        mean, sd = summary[method]
        ax2.plot(np.arange(n_res), diffs[method], color=COLORS[method], linewidth=LW[method] * 0.8,
                  label=f'{method}: mean={mean:+.1f}, s.d.={sd:.1f}',
                  zorder=3 if method == 'bioemu' else 2)
    ax2.axhline(0, color='black', linewidth=1.0)
    ax2.set_ylabel('Per-residue difference\n(generated − reference)', fontsize=13)
    ax2.set_xlabel('Residue index (topology order, 0-based)', fontsize=13)
    ax2.legend(fontsize=9, loc='lower right')
    fig.tight_layout()

    tag = f'{protein}_{emdb_id.replace("EMD-", "")}'
    fig.savefig(f'{OUT_DIR}/{tag}_bestframe_diff.png', dpi=150)
    plt.close(fig)

    out_rows = []
    for i in range(n_res):
        row = dict(residue_index=i, reference=ref_prof[i])
        for m in ['bioemu', 'alphaflow', 'msa']:
            row[f'{m}_profile'] = profiles[m][i]
            row[f'{m}_diff'] = diffs[m][i]
        out_rows.append(row)
    pd.DataFrame(out_rows).to_excel(f'{OUT_DIR}/{tag}_bestframe_diff.xlsx', index=False)

    print(f'    summary: ' + ', '.join(f'{m} mean={s[0]:+.1f} sd={s[1]:.1f}' for m, s in summary.items()))
    print(f'    saved {OUT_DIR}/{tag}_bestframe_diff.png (+ .xlsx)')
    return summary


if __name__ == '__main__':
    df = pd.read_excel(DATASET)
    targets = sys.argv[1:] if len(sys.argv) > 1 else sorted(df['protein_label'].unique())
    for protein in targets:
        rows = df[df['protein_label'] == protein]
        for _, r in rows.iterrows():
            build_bestframe_diff_figure(protein, r['emdb_id'], r['pdb_id'], r['resolution'])
    print('all done')
