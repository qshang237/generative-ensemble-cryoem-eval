"""
CC_mask of every generated frame with every experimental map of its protein.

A second density-based score, to check the density ratio. The frames, the atoms and the
placement are the same as for the density ratio; only the scoring function is different.

Model density  sum over the backbone atoms (N, CA, C, O) of  Z * exp(-d^2 / (2 sigma^2)),
               sigma = 0.225 * map resolution, Z = atomic number. This is the convention of
               ChimeraX molmap: the blur comes from the map resolution, no B-factor is used.
Mask           map grid points within 2.5 A of a backbone atom of the frame
               (the radius used in 03_cross_comparison.py).
CC_mask        Pearson correlation of the model density with the deposited map (raw map,
               not smoothed) over the mask.
Placement      superposition on the deposited model of the map
               (shared_fitting.superpose_to_target), as for the density ratio.
Reference      the deposited model, backbone atoms only, same calculation on its own map.
               cc_ratio = CC_mask / CC_mask(deposited).

Usage: python 18_cc_mask.py [--budget SECONDS] [--workers N]
       The run can be stopped and started again; finished ensemble-map pairs are skipped.
Outputs: results/cc_mask/
"""
import os
import sys
import time
import argparse
import importlib.util
import multiprocessing as mp
import numpy as np
import pandas as pd
import mdtraj as md
import gemmi

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import shared_fitting as sf

OUT = 'results/cc_mask'
SIGMA_FACTOR = 0.225      # sigma = SIGMA_FACTOR * resolution (ChimeraX molmap: 1 / (pi * sqrt(2)))
N_SIGMA = 5.0             # the Gaussian of an atom is cut at >= 5 sigma (ChimeraX molmap: 5)
R_MASK = 2.5              # A
BACKBONE = 'name N CA C O'
Z = {'N': 7.0, 'C': 6.0, 'O': 8.0}
METHODS = [('bioemu', 'BioEmu', '04_fitting_bioemu.py', 'load_bioemu'),
           ('alphaflow', 'AlphaFlow', '04_fitting_alphaflow.py', 'load_alphaflow'),
           ('msa', 'AF2-MSA', '04_fitting_msa.py', 'load_msa')]


def _load(script, fn):
    spec = importlib.util.spec_from_file_location(script[:-3], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


# ── model density and correlation ────────────────────────────────────────────
def cc_mask_frame(xyz, z, rho, spacing, sigma, r_mask=R_MASK, n_sigma=N_SIGMA, chunk=300):
    """
    xyz (A, 3) in Angstrom in the frame of the map; z (A,) atomic numbers;
    rho (nx, ny, nz) map with origin at 0 and orthogonal axes; spacing (3,) in Angstrom.
    The model density is calculated exactly at the map grid points (no interpolation).
    Returns (CC_mask, number of mask points).
    """
    nmap = np.array(rho.shape)
    g = xyz / spacing                                # atom positions in grid units
    base = np.rint(g).astype(np.int64)               # nearest grid point of each atom
    hg = int(np.ceil(n_sigma * sigma / spacing.min()))
    hm = int(np.ceil(r_mask / spacing.min()))
    assert hm <= hg, 'mask radius larger than the Gaussian cut'
    lo = base.min(0) - hg                            # local box that holds all contributions
    shape = base.max(0) + hg + 1 - lo
    size = int(np.prod(shape))
    sim = np.zeros(size)
    mask = np.zeros(size, dtype=bool)
    og, om = np.arange(-hg, hg + 1), np.arange(-hm, hm + 1)

    def flat_index(i, j, k):
        return ((i - lo[0])[:, :, None, None] * shape[1]
                + (j - lo[1])[:, None, :, None]) * shape[2] + (k - lo[2])[:, None, None, :]

    for s in range(0, len(xyz), chunk):
        b, gg, zz = base[s:s + chunk], g[s:s + chunk], z[s:s + chunk]
        # Gaussian of each atom: product of three 1-D Gaussians (the grid is orthogonal)
        idx = [b[:, a, None] + og[None] for a in range(3)]
        e = [np.exp(-((idx[a] - gg[:, a, None]) * spacing[a]) ** 2 / (2 * sigma ** 2)) for a in range(3)]
        w = (zz[:, None, None, None] * e[0][:, :, None, None]) * e[1][:, None, :, None] * e[2][:, None, None, :]
        sim += np.bincount(flat_index(*idx).ravel(), weights=w.ravel(), minlength=size)
        # mask: grid points within r_mask of the atom
        idm = [b[:, a, None] + om[None] for a in range(3)]
        d2 = [((idm[a] - gg[:, a, None]) * spacing[a]) ** 2 for a in range(3)]
        near = (d2[0][:, :, None, None] + d2[1][:, None, :, None] + d2[2][:, None, None, :]) <= r_mask ** 2
        mask[flat_index(*idm)[near]] = True

    ax = [(lo[a] + np.arange(shape[a])) % nmap[a] for a in range(3)]   # periodic, as gemmi
    obs = rho[np.ix_(*ax)].ravel()[mask].astype(np.float64)
    cal = sim[mask]
    if len(obs) < 10 or obs.std() == 0 or cal.std() == 0:
        return float('nan'), int(mask.sum())
    return float(np.corrcoef(obs, cal)[0, 1]), int(mask.sum())


def load_raw_map(emdb_id):
    """Deposited map as it is (no smoothing, no normalisation). Returns (array, spacing)."""
    ccp4 = gemmi.read_ccp4_map(sf.download_map(emdb_id))
    ccp4.setup(0.0)
    grid = ccp4.grid
    c = grid.unit_cell
    assert (c.alpha, c.beta, c.gamma) == (90.0, 90.0, 90.0), f'{emdb_id}: axes not orthogonal'
    assert all(ccp4.header_i32(k) == 0 for k in (5, 6, 7)), f'{emdb_id}: map start is not 0'
    assert all(ccp4.header_float(k) == 0 for k in (50, 51, 52)), f'{emdb_id}: map origin is not 0'
    rho = np.array(grid, copy=True).astype(np.float32)
    return rho, np.array([c.a / grid.nu, c.b / grid.nv, c.c / grid.nw])


def atomic_numbers(top, idx):
    return np.array([Z[top.atom(int(i)).element.symbol] for i in idx])


# ── ensembles ────────────────────────────────────────────────────────────────
def backbone_traj(key, loader, protein):
    """Backbone-only trajectory of an ensemble. Cached: large PDB files load slowly."""
    os.makedirs(f'{OUT}/cache', exist_ok=True)
    fx, ft = f'{OUT}/cache/{key}_{protein}_xyz.npy', f'{OUT}/cache/{key}_{protein}_top.pdb'
    if os.path.exists(fx) and os.path.exists(ft):
        return md.Trajectory(np.load(fx), md.load(ft).topology)
    traj = loader(protein)
    if traj is None:
        return None
    assert traj.n_frames <= sf.MAX_FRAMES, 'more frames than the density-ratio analysis used'
    traj = traj.atom_slice(traj.topology.select(BACKBONE))
    traj[0].save_pdb(ft)
    np.save(fx, traj.xyz)
    return traj


_G = {}


def _init(rho, spacing, sigma, z):
    _G.update(rho=rho, spacing=spacing, sigma=sigma, z=z)


def _work(xyz):
    return [cc_mask_frame(x, _G['z'], _G['rho'], _G['spacing'], _G['sigma']) for x in xyz]


def score_ensemble(traj, row, rho, spacing, workers):
    """CC_mask of all frames of a backbone trajectory with one map. Returns (cc, n_mask)."""
    t = md.Trajectory(traj.xyz.copy(), traj.topology)
    sf.superpose_to_target(t, sf.download_pdb(row.pdb_id))
    xyz = t.xyz.astype(np.float64) * 10
    z = atomic_numbers(t.topology, range(t.n_atoms))
    parts = np.array_split(xyz, max(1, min(len(xyz), workers * 8)))
    with mp.Pool(workers, initializer=_init, initargs=(rho, spacing, SIGMA_FACTOR * row.resolution, z)) as pool:
        res = [r for part in pool.map(_work, parts) for r in part]
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def deposited_cc(row, rho, spacing):
    m = md.load(sf.download_pdb(row.pdb_id))
    idx = m.topology.select('protein and ' + BACKBONE)
    return cc_mask_frame(m.xyz[0, idx].astype(np.float64) * 10, atomic_numbers(m.topology, idx),
                         rho, spacing, SIGMA_FACTOR * row.resolution)


def deposited_cross(rows):
    """
    CC_mask of every deposited model of one protein (rows of the dataset table) with every
    map of that protein, after superposition on the deposited model of the map.
    Rows = model, columns = map. Used by 19_metric_summary.py.
    """
    M = pd.DataFrame(index=rows.emdb_id.values, columns=rows.emdb_id.values, dtype=float)
    for _, rj in rows.iterrows():
        rho, spacing = load_raw_map(rj.emdb_id)
        for _, ri in rows.iterrows():
            m = md.load(sf.download_pdb(ri.pdb_id))
            m = m.atom_slice(m.topology.select('protein and ' + BACKBONE))
            if ri.pdb_id != rj.pdb_id:
                sf.superpose_to_target(m, sf.download_pdb(rj.pdb_id))
            M.loc[ri.emdb_id, rj.emdb_id] = cc_mask_frame(
                m.xyz[0].astype(np.float64) * 10, atomic_numbers(m.topology, range(m.n_atoms)),
                rho, spacing, SIGMA_FACTOR * rj.resolution)[0]
    return M


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=1e9, help='stop cleanly after this many seconds')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(f'{OUT}/parts', exist_ok=True)
    df = pd.read_excel(sf.DATASET)

    todo = 0
    for _, row in df.iterrows():
        fdep = f'{OUT}/parts/deposited_{row.emdb_id}.csv'
        units = [(k, lab, s, fn) for k, lab, s, fn in METHODS
                 if not os.path.exists(f'{OUT}/parts/{k}_{row.protein_label}_{row.emdb_id}.csv')]
        if os.path.exists(fdep) and not units:
            continue
        if time.time() - t0 > a.budget:
            todo += 1
            continue
        rho, spacing = load_raw_map(row.emdb_id)
        if not os.path.exists(fdep):
            cc, n = deposited_cc(row, rho, spacing)
            pd.DataFrame([dict(protein=row.protein_label, emdb_id=row.emdb_id, pdb_id=row.pdb_id,
                               resolution=row.resolution, cc_mask_deposited=cc, n_mask=n)]).to_csv(fdep, index=False)
            print(f'deposited {row.pdb_id} in {row.emdb_id}: CC_mask {cc:.3f}', flush=True)
        for key, label, script, fn in units:
            if time.time() - t0 > a.budget:
                todo += 1
                continue
            traj = backbone_traj(key, _load(script, fn), row.protein_label)
            if traj is None:
                continue
            t1 = time.time()
            cc, n = score_ensemble(traj, row, rho, spacing, a.workers)
            pd.DataFrame(dict(protein=row.protein_label, frame=np.arange(len(cc)), emdb_id=row.emdb_id,
                              cc_mask=cc, n_mask=n)).to_csv(
                f'{OUT}/parts/{key}_{row.protein_label}_{row.emdb_id}.csv', index=False)
            print(f'{label} {row.protein_label} {row.emdb_id}: {len(cc)} frames, CC_mask median '
                  f'{np.nanmedian(cc):.3f} max {np.nanmax(cc):.3f} ({time.time() - t1:.0f} s)', flush=True)
    if todo:
        print(f'time budget reached; {todo} map(s) or ensemble(s) not finished. Run again.')
        return

    # merge: one table per method with density ratio, RMSD and CC_mask
    dep = pd.concat([pd.read_csv(f'{OUT}/parts/deposited_{e}.csv') for e in df.emdb_id])
    dep.to_csv(f'{OUT}/deposited_cc_mask.csv', index=False)
    for key, label, _, _ in METHODS:
        parts = [pd.read_csv(f'{OUT}/parts/{key}_{r.protein_label}_{r.emdb_id}.csv')
                 for _, r in df.iterrows()
                 if os.path.exists(f'{OUT}/parts/{key}_{r.protein_label}_{r.emdb_id}.csv')]
        if not parts:
            continue
        c = pd.concat(parts).merge(dep[['emdb_id', 'cc_mask_deposited']], on='emdb_id')
        c['cc_ratio'] = c.cc_mask / c.cc_mask_deposited
        base = pd.read_csv(f'results/rmsd_to_deposited/{key}_density_ratio_and_rmsd.csv')
        m = base.merge(c.drop(columns='n_mask'), on=['protein', 'frame', 'emdb_id'], how='left')
        assert m.cc_mask.notna().all() and len(m) == len(c), f'{key}: frames do not match'
        m.to_csv(f'{OUT}/{key}_density_ratio_rmsd_cc_mask.csv', index=False)
    print('all done; merged tables written to', OUT)


if __name__ == '__main__':
    main()
