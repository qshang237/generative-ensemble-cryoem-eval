"""
Backbone Q-score of every generated frame with every experimental map of its protein.

Q-score (Pintilie et al. 2020, Nat. Methods 17, 328; program MapQ): for each atom, the map
values around the atom are compared with a reference Gaussian centred on the atom.

  points     at the atom centre and on shells of radius 0.1, 0.2, ... 2.0 A around it.
             On each shell, 8 points that are not within 0.9 * radius of any other atom of
             the model (so the points belong to this atom). If fewer than 8 of the 8 trial
             points pass, the shell is tried again with 10, 12, ... trial points (MapQ rule).
  u          map value at the points (deposited map, raw; trilinear interpolation)
  v          reference Gaussian exp(-r^2 / (2 sigma^2)), sigma = SIGMA
  Q (atom)   Pearson correlation of u and v
  Q (model)  mean over the atoms

The model is the backbone only (N, CA, C, O) for the frames and for the deposited models:
BioEmu gives no side chains, so all three methods are treated the same way. Frames are
placed as for the density ratio (superposition on the deposited model of the map).
q_ratio = Q(frame) / Q(deposited model, backbone only, on its own map).

Usage: python 20_q_score.py [--budget SECONDS] [--workers N]
       The run can be stopped and started again; finished blocks of frames are skipped.
Outputs: results/q_score/
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')      # the work is split over processes; threads would compete
import sys
import time
import argparse
import importlib.util
import multiprocessing as mp
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import shared_fitting as sf

OUT = 'results/q_score'
SIGMA = 0.4               # A. MapQ default and the value used for EMDB validation reports
                          # (the 2020 paper used 0.6)
N_PTS = 8                 # points per shell
RADII = np.round(np.arange(1, 21) * 0.1, 1)     # 0.1 ... 2.0 A
CLASH = 0.9               # a point is rejected if another atom is within CLASH * radius
MAX_TRIES = 50
MAX_ELEMENTS = 2e7        # largest atoms x neighbours x points array made in one piece
BLOCK = 100               # frames per saved block
BACKBONE = 'name N CA C O'


def _module(script):
    spec = importlib.util.spec_from_file_location('m' + script[:2], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cc = _module('18_cc_mask.py')          # same maps, same ensembles, same cache as CC_mask
METHODS, load_raw_map, backbone_traj = _cc.METHODS, _cc.load_raw_map, _cc.backbone_traj


# ── Q-score ──────────────────────────────────────────────────────────────────
def sphere_points(n):
    """n points spread evenly on the unit sphere (spiral; function SpherePts of MapQ)."""
    k = np.arange(1, n + 1)
    h = -1.0 + 2.0 * (k - 1) / (n - 1)
    phi = np.arccos(h)
    theta = np.zeros(n)
    for i in range(1, n - 1):
        theta[i] = (theta[i - 1] + 3.6 / np.sqrt(n * (1.0 - h[i] ** 2))) % (2 * np.pi)
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)


_UNIT = [sphere_points(N_PTS + 2 * i) for i in range(MAX_TRIES)]


def sample_points(xyz):
    """
    Sample points of all atoms of a model. xyz (A, 3) in Angstrom.
    Returns atom index (P,), radius (P,), offset from the atom (P, 3).
    The points depend on the model only, not on the map.
    """
    n_atoms = len(xyz)
    # neighbours of each atom that can reject one of its points (table sorted by distance)
    tree = cKDTree(xyz)
    cut = (1 + CLASH) * RADII[-1] + 1e-6
    k = max(2, int(tree.query_ball_point(xyz, cut, return_length=True).max()))
    dist, nb = tree.query(xyz, k=k, distance_upper_bound=cut)
    dist, nb = dist[:, 1:], nb[:, 1:]                                  # column 0 is the atom itself
    rel = np.vstack([xyz, [[1e9, 1e9, 1e9]]])[nb] - xyz[:, None, :]    # missing neighbour = far away
    rel2 = (rel ** 2).sum(-1)
    atom = [np.repeat(np.arange(n_atoms), N_PTS)]            # centre, counted N_PTS times
    rad = [np.zeros(n_atoms * N_PTS)]
    off = [np.zeros((n_atoms * N_PTS, 3))]
    for r in RADII:
        kr = int((dist <= (1 + CLASH) * r).sum(1).max())     # neighbours that matter on this shell
        left = np.arange(n_atoms)
        for i in range(MAX_TRIES):
            u = _UNIT[i] * r                                 # trial points, relative to the atom
            if kr:
                # squared distance of every trial point to every neighbour: |u - rel|^2
                # (atoms in pieces, so that a collapsed model with hundreds of neighbours
                # per atom does not need a very large array; the values are the same)
                step = max(1, int(MAX_ELEMENTS // (kr * len(u))))
                ok = np.concatenate([
                    ((r * r - 2.0 * (rel[left[s:s + step], :kr] @ u.T) + rel2[left[s:s + step], :kr, None])
                     >= (CLASH * r) ** 2).all(1) for s in range(0, len(left), step)])
            else:
                ok = np.ones((len(left), len(u)), dtype=bool)
            done = ok.sum(1) >= N_PTS
            if i == MAX_TRIES - 1:
                done[:] = True                               # last try: use the points there are
            a, p = np.nonzero(ok & done[:, None])
            atom.append(left[a])
            rad.append(np.full(len(a), r))
            off.append(u[p])
            left = left[~done]
            if len(left) == 0:
                break
    return np.concatenate(atom), np.concatenate(rad), np.concatenate(off)


def q_atoms(points, atom, rad, n_atoms, rho, spacing, sigma=SIGMA):
    """Q-score of each atom. points (P, 3) in Angstrom in the frame of the map."""
    u = map_coordinates(rho, (points / spacing).T, order=1, mode='grid-wrap').astype(np.float64)
    v = np.exp(-0.5 * (rad / sigma) ** 2)
    n = np.bincount(atom, minlength=n_atoms).astype(np.float64)
    s = lambda w: np.bincount(atom, weights=w, minlength=n_atoms)
    su, sv = s(u), s(v)
    cov = s(u * v) - su * sv / n
    var_u = s(u * u) - su ** 2 / n
    var_v = s(v * v) - sv ** 2 / n
    flat = var_u <= 1e-12 * (s(u * u) / n + 1e-300)          # constant map around the atom
    q = np.zeros(n_atoms)
    q[~flat] = cov[~flat] / np.sqrt(var_u[~flat] * var_v[~flat])
    return q


def q_model(xyz, rho, spacing, sigma=SIGMA):
    """Mean Q-score of a model that is already in the frame of the map."""
    atom, rad, off = sample_points(xyz)
    return float(q_atoms(xyz[atom] + off, atom, rad, len(xyz), rho, spacing, sigma).mean())


def rotation(a, b):
    """Rotation R with (a - mean) @ R.T = (b - mean) for two copies of one rigid body."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    U, _, Vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(U @ Vt))
    return (U @ np.diag([1.0, 1.0, d]) @ Vt).T


# ── ensembles ────────────────────────────────────────────────────────────────
_G = {}


def _init(maps):
    _G['maps'] = maps


def _work(args):
    """x0 (F, A, 3) model coordinates; xs: one array (F, A, 3) per map, in the map frame."""
    x0, xs = args
    out = np.zeros((len(x0), len(xs)))
    for f in range(len(x0)):
        atom, rad, off = sample_points(x0[f])
        for j, (rho, spacing) in enumerate(_G['maps']):
            pts = xs[j][f][atom] + off @ rotation(x0[f], xs[j][f]).T
            out[f, j] = q_atoms(pts, atom, rad, x0.shape[1], rho, spacing).mean()
    return out


def deposited_backbone(row):
    m = md.load(sf.download_pdb(row.pdb_id))
    return m.atom_slice(m.topology.select('protein and ' + BACKBONE))


def deposited_cross(rows):
    """Q-score of every deposited model of one protein (backbone) with every map of that
    protein, after superposition on the deposited model of the map. Rows = model, columns = map."""
    M = pd.DataFrame(index=rows.emdb_id.values, columns=rows.emdb_id.values, dtype=float)
    for _, rj in rows.iterrows():
        rho, spacing = load_raw_map(rj.emdb_id)
        for _, ri in rows.iterrows():
            m = deposited_backbone(ri)
            if ri.pdb_id != rj.pdb_id:
                sf.superpose_to_target(m, sf.download_pdb(rj.pdb_id))
            M.loc[ri.emdb_id, rj.emdb_id] = q_model(m.xyz[0].astype(np.float64) * 10, rho, spacing)
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
    unfinished = False

    for protein in df.protein_label.unique():
        rows = df[df.protein_label == protein]
        fdep = f'{OUT}/parts/deposited_{protein}.csv'
        todo = [(k, lab, s, fn) for k, lab, s, fn in METHODS if not os.path.exists(f'{OUT}/parts/{k}_{protein}.csv')]
        if os.path.exists(fdep) and not todo:
            continue
        if time.time() - t0 > a.budget:
            unfinished = True
            continue
        maps = [load_raw_map(e) for e in rows.emdb_id]
        if not os.path.exists(fdep):
            dep = [dict(protein=protein, emdb_id=r.emdb_id, pdb_id=r.pdb_id, resolution=r.resolution,
                        q_deposited=q_model(deposited_backbone(r).xyz[0].astype(np.float64) * 10, *maps[j]))
                   for j, (_, r) in enumerate(rows.iterrows())]
            pd.DataFrame(dep).to_csv(fdep, index=False)
            print(f'deposited {protein}: ' + ', '.join(f"{d['pdb_id']} {d['q_deposited']:.3f}" for d in dep), flush=True)
        for key, label, script, fn in todo:
            traj = backbone_traj(key, getattr(_module(script), fn), protein)
            if traj is None:
                continue
            x0 = traj.xyz.astype(np.float64) * 10
            xs = []
            for _, r in rows.iterrows():
                t = md.Trajectory(traj.xyz.copy(), traj.topology)
                sf.superpose_to_target(t, sf.download_pdb(r.pdb_id))
                xs.append(t.xyz.astype(np.float64) * 10)
            blocks = [(s, min(s + BLOCK, len(x0))) for s in range(0, len(x0), BLOCK)]
            with mp.Pool(a.workers, initializer=_init, initargs=(maps,)) as pool:
                for s, e in blocks:
                    fb = f'{OUT}/parts/block_{key}_{protein}_{s}.npy'
                    if os.path.exists(fb):
                        continue
                    if time.time() - t0 > a.budget:
                        unfinished = True
                        break
                    idx = np.array_split(np.arange(s, e), a.workers)
                    res = pool.map(_work, [(x0[i], [x[i] for x in xs]) for i in idx if len(i)])
                    np.save(fb, np.concatenate(res))
            fbs = [f'{OUT}/parts/block_{key}_{protein}_{s}.npy' for s, _ in blocks]
            if all(os.path.exists(f) for f in fbs):
                q = np.concatenate([np.load(f) for f in fbs])
                pd.concat([pd.DataFrame(dict(protein=protein, frame=np.arange(len(q)), emdb_id=e, q_score=q[:, j]))
                           for j, e in enumerate(rows.emdb_id)]).to_csv(f'{OUT}/parts/{key}_{protein}.csv', index=False)
                for f in fbs:
                    os.remove(f)
                print(f'{label} {protein}: {len(q)} frames; best Q per map ' + ', '.join(f'{v:.3f}' for v in q.max(0)), flush=True)
    if unfinished:
        print('time budget reached; not finished. Run again.')
        return

    dep = pd.concat([pd.read_csv(f'{OUT}/parts/deposited_{p}.csv') for p in df.protein_label.unique()])
    dep.to_csv(f'{OUT}/deposited_q_score.csv', index=False)
    for key, label, _, _ in METHODS:
        parts = [pd.read_csv(f'{OUT}/parts/{key}_{p}.csv') for p in df.protein_label.unique()
                 if os.path.exists(f'{OUT}/parts/{key}_{p}.csv')]
        if not parts:
            continue
        q = pd.concat(parts).merge(dep[['emdb_id', 'q_deposited']], on='emdb_id')
        q['q_ratio'] = q.q_score / q.q_deposited
        q.to_csv(f'{OUT}/{key}_q_score.csv', index=False)
    print('all done; tables written to', OUT)


if __name__ == '__main__':
    main()
