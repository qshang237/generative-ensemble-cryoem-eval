"""
07_coordinate_pca.py
====================
Principal-component analysis of Cα coordinates for every generated ensemble,
coloured by the GROMOS/Daura clusters, to show how the clusters separate.

Why coordinates and not the pairwise-RMSD matrix
------------------------------------------------
Clustering (shared_fitting.run_analysis) superposes every frame once onto the
deposited model of the protein's first map and then computes pairwise Cα-RMSD
in that common frame, without refitting each pair. In that case
RMSD_ij = ||x_i - x_j|| / sqrt(L), i.e. a scaled Euclidean distance between the
flattened coordinate vectors. Classical MDS of that matrix is therefore
mathematically identical to PCA of the coordinates (checked numerically below
for every ensemble). PCA of the raw RMSD matrix itself is not equivalent: its
rows are strongly correlated, which inflates PC1 and bends continuous
variation into an arch.

Scaling: scores are divided by sqrt(n_atoms), so distances in the PC plane are
in Å and are a lower bound of the Cα-RMSD between two structures.

Output: results/coordinate_pca/
    {protein}_coordinate_pca.png   left: pairwise RMSD, right: PCA (3 methods)
    summary.xlsx                   variance explained, checks, correlations
    log.txt
Run from the model-comparing folder:  python3 07_coordinate_pca.py
"""

import os
import copy
import importlib.util
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from shared_fitting import (DATASET, download_pdb, superpose_to_target,
                            pairwise_ca_rmsd, select_daura_cutoff)

OUT = 'results/coordinate_pca'
os.makedirs(OUT, exist_ok=True)
LOG = open(f'{OUT}/log.txt', 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n'); LOG.flush()


def _load(script, fn):
    spec = importlib.util.spec_from_file_location(script[:-3], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


METHODS = [('bioemu', 'BioEmu', _load('04_fitting_bioemu.py', 'load_bioemu')),
           ('alphaflow', 'AlphaFlow', _load('04_fitting_alphaflow.py', 'load_alphaflow')),
           ('msa', 'AF2-MSA', _load('04_fitting_msa.py', 'load_msa'))]
COLORS = plt.get_cmap('tab10').colors
SLC_GROUP = {'EMD-67847': 'A', 'EMD-62584': 'A'}   # 21nq, 9kuy; all other SLC37A4 maps are group B


def ca_indices(top):
    return np.array([a.index for a in top.atoms if a.name == 'CA' and a.residue.is_protein])


def classical_mds(rmsd, L, k=2):
    """Classical MDS of a distance matrix given as RMSD; returns scores in Å."""
    d2 = (rmsd ** 2) * L
    n = d2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    return v[:, :k] * np.sqrt(np.clip(w[:k], 0, None)) / np.sqrt(L), w


def pca(X, k=2):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return mu, Vt[:k], (Xc @ Vt[:k].T), var


def run_one(protein, rows, method, label, loader):
    traj = loader(protein)
    if traj is None:
        log(f'  {label}: no ensemble'); return None
    n = traj.n_frames
    ref_pdb = download_pdb(rows.iloc[0]['pdb_id'])

    # 1. same superposition as clustering: onto the first map's deposited model
    tr = copy.deepcopy(traj)
    superpose_to_target(tr, ref_pdb)
    ca = ca_indices(tr.topology)

    # 2. reproduce clustering exactly and compare with the saved labels
    rmsd = pairwise_ca_rmsd(tr)
    daura = select_daura_cutoff(rmsd)
    labels = daura['labels']
    saved = pd.read_excel(f'results/{method}/per_frame_density.xlsx')
    saved = saved[saved.protein == protein].drop_duplicates('frame').sort_values('frame')
    match = (len(saved) == n) and np.array_equal(saved.cluster.values, labels)
    log(f'  {label}: {n} frames, {len(ca)} Cα, clusters {np.bincount(labels).tolist()}, '
        f'labels identical to saved run: {match}')

    # 3. PCA of all Cα coordinates (the same atoms and frame as the RMSD matrix),
    #    checked against classical MDS of that matrix
    X = (tr.xyz[:, ca, :] * 10).reshape(n, -1)
    _, _, scores, var = pca(X)
    scores = scores / np.sqrt(len(ca))
    s_mds, _ = classical_mds(rmsd, len(ca))
    eq = [abs(np.corrcoef(scores[:, i], s_mds[:, i])[0, 1]) for i in range(2)]
    log(f'      PCA: PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%; '
        f'agreement with MDS of the RMSD matrix |r| {eq[0]:.6f}, {eq[1]:.6f}')

    # 5. SLC37A4 only: correlation of PC1 with a state-preference and a fit-quality score
    corr = {}
    if protein == 'SLC37A4':
        f = pd.read_excel(f'results/{method}/per_frame_density.xlsx')
        f = f[f.protein == protein].pivot(index='frame', columns='emdb_id', values='density_ratio').sort_index()
        a = [c for c in f.columns if SLC_GROUP.get(c) == 'A']
        b = [c for c in f.columns if c not in a]
        pref = (f[b].mean(axis=1) - f[a].mean(axis=1)).values
        fit = f.mean(axis=1).values
        corr = dict(r_pc1_pref=np.corrcoef(scores[:, 0], pref)[0, 1],
                    r_pc1_fit=np.corrcoef(scores[:, 0], fit)[0, 1],
                    r_pref_fit=np.corrcoef(pref, fit)[0, 1])
        log('      SLC37A4 PC1 vs state preference (mean B - mean A) r = {r_pc1_pref:.2f}, '
            'vs fit quality (mean over 6 maps) r = {r_pc1_fit:.2f}, preference vs fit r = {r_pref_fit:.2f}'.format(**corr))

    return dict(label=label, rmsd=rmsd, labels=labels, scores=scores, var=var,
                row=dict(protein=protein, method=label, n_frames=n, n_ca=len(ca),
                         n_clusters=int(labels.max() + 1), labels_match_saved=match,
                         pc1_var=var[0], pc2_var=var[1],
                         mds_equivalence_r_pc1=eq[0], mds_equivalence_r_pc2=eq[1], **corr))


def figure(protein, res):
    fig, ax = plt.subplots(len(res), 2, figsize=(11.5, 3.3 * len(res)), squeeze=False)
    for i, r in enumerate(res):
        v = r['rmsd'][np.triu_indices(r['rmsd'].shape[0], k=1)]
        ax[i, 0].hist(v, bins=60, color='steelblue', edgecolor='black', linewidth=0.3)
        ax[i, 0].set_xlabel('Pairwise Cα-RMSD (Å)'); ax[i, 0].set_ylabel('Count')
        ax[i, 0].set_title(f"{r['label']}: pairwise RMSD", fontsize=11, fontweight='bold', loc='left')
        s, lab = r['scores'], r['labels']
        for c in range(lab.max() + 1):
            m = lab == c
            ax[i, 1].scatter(s[m, 0], s[m, 1], s=10, alpha=0.5, color=COLORS[c],
                             label=f'C{c} (n={m.sum()})')
        ax[i, 1].set_xlabel(f"PC1 (Å, {r['var'][0]*100:.0f}% variance)")
        ax[i, 1].set_ylabel(f"PC2 (Å, {r['var'][1]*100:.0f}%)")
        ax[i, 1].set_title(f"{r['label']}: PCA of Cα coordinates", fontsize=11, fontweight='bold', loc='left')
        ax[i, 1].set_aspect('equal', adjustable='datalim')
        ax[i, 1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{protein}_coordinate_pca.png', dpi=200)
    plt.close(fig)


def cache_path(protein, method):
    return f'{OUT}/cache/{protein}_{method}.npz'


def save(protein, method, r):
    os.makedirs(f'{OUT}/cache', exist_ok=True)
    np.savez(cache_path(protein, method), rmsd=r['rmsd'], labels=r['labels'], scores=r['scores'],
             var=r['var'], label=r['label'], row=np.array([r['row']], dtype=object))


def load(protein, method):
    z = np.load(cache_path(protein, method), allow_pickle=True)
    return dict(rmsd=z['rmsd'], labels=z['labels'], scores=z['scores'], var=z['var'],
                label=str(z['label']), row=z['row'][0])


if __name__ == '__main__':
    # usage:  python3 07_coordinate_pca.py                 all proteins, all methods
    #         python3 07_coordinate_pca.py GltPh alphaflow one ensemble (cached)
    #         python3 07_coordinate_pca.py --figures       figures + summary from cache
    import sys
    df = pd.read_excel(DATASET)
    proteins = list(df['protein_label'].unique())
    args = sys.argv[1:]
    if args[:1] != ['--figures']:
        todo = [(p, m) for p in proteins for m, _, _ in METHODS]
        if len(args) == 2:
            todo = [tuple(args)]
        for protein, method in todo:
            log(f'\n===== {protein} {method}')
            rows = df[df['protein_label'] == protein].reset_index(drop=True)
            label, loader = {m: (l, f) for m, l, f in METHODS}[method]
            r = run_one(protein, rows, method, label, loader)
            if r is not None:
                save(protein, method, r)
    if args[:1] == ['--figures'] or not args:
        summary = []
        for protein in proteins:
            res = [load(protein, m) for m, _, _ in METHODS if os.path.exists(cache_path(protein, m))]
            if res:
                figure(protein, res)
                summary += [r['row'] for r in res]
        pd.DataFrame(summary).to_excel(f'{OUT}/summary.xlsx', index=False)
        log('figures and summary written')
