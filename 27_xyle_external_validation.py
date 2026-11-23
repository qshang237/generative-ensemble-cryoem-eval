"""
External validation of the clustering procedure: ensembles with known state labels.

The clustering of shared_fitting.py is applied, without any change, to synthetic XylE
ensembles (14_xyle_states.py, 16_xyle_ensembles.py), exactly as to a generated ensemble:
superposition of all frames on one reference model, pairwise Calpha-RMSD in that common
frame, Hartigan's dip test, cut-off from the valleys of the RMSD density, GROMOS/Daura
clustering (shared_fitting.select_daura_cutoff). The clusters are then compared with the
known state of every frame (adjusted Rand index, ARI; 1 = the same partition).

Sets      XylE_uniform, XylE_main, XylE_skewed: five states with different populations.
          XylE_one_state: one state only; the correct answer is one cluster.
Noise     median pairwise Calpha-RMSD within one state. With more noise the peaks of the
          pairwise-RMSD distribution merge, so the task becomes harder.
Repeats   REPEATS random ensembles for every set and noise level (seed = SEED + repeat).
          All sets of one repeat are always generated in the same order, so a case gives
          the same ensemble whether or not the run was stopped and started again.

Usage: python 27_xyle_external_validation.py [--budget seconds]
       The run can be stopped and started again; finished cases are skipped.
Outputs: results/xyle/external_validation.csv           one row per ensemble
         results/xyle/external_validation_summary.csv   per noise level and set
"""
import os
import sys
import time
import argparse
import importlib.util
import numpy as np
import pandas as pd
import mdtraj as md
from sklearn.metrics import adjusted_rand_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import shared_fitting as sf

NOISE = (1.0, 1.5, 2.0)        # Angstrom
REPEATS = 10
OUT = 'results/xyle'
REFERENCE = f'{OUT}/XylE_4GBY.pdb'      # all frames are superposed on this model before the RMSD


def _module(script):
    spec = importlib.util.spec_from_file_location(script[:-3], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def topology(e, m):
    """Backbone topology of the XylE frames (residue names and numbers of the structures)."""
    top = md.Topology()
    chain = top.add_chain()
    for name, number in zip(m.resname, m.res):
        res = top.add_residue(name, chain, resSeq=int(number))
        for atom, element in zip(e.BACKBONE, e.ELEMENT):
            top.add_atom(atom, md.element.get_by_symbol(element), res)
    return top


def one_case(e, traj, lab):
    """The clustering of the main analysis on one ensemble; returns one row of the table."""
    sf.superpose_to_target(traj, REFERENCE)
    R = sf.pairwise_ca_rmsd(traj)
    r = sf.select_daura_cutoff(R)
    cl = r['labels']
    tab = pd.crosstab(lab, cl)
    return dict(true_k=len(set(lab)), k=r['n_clusters'], dip_p=r['dip_p'], cutoff=r['cutoff'],
                sizes=' / '.join(map(str, np.bincount(cl))), ari=adjusted_rand_score(lab, cl),
                table='; '.join(f'{e.STATES[s]}: ' + ' '.join(map(str, tab.loc[s].values)) for s in tab.index))


def summary(d):
    g = d.groupby(['noise', 'set'], sort=False)
    s = g.agg(n=('ari', 'size'), true_k=('true_k', 'first'), ari_mean=('ari', 'mean'), ari_min=('ari', 'min'),
              ari_max=('ari', 'max'), cutoff_mean=('cutoff', 'mean')).reset_index()
    s.insert(4, 'clusters_found', [', '.join(f'{k} in {n} of {len(x)}' for k, n in x.k.value_counts().sort_index().items())
                                   for _, x in g])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=1e9, help='stop cleanly after this many seconds')
    a = ap.parse_args()
    t0 = time.time()
    e = _module('16_xyle_ensembles.py')
    f = f'{OUT}/external_validation.csv'
    done = set()
    if os.path.exists(f):
        d = pd.read_csv(f)
        done = set(zip(d.noise, d.set, d.repeat))
    left = 0
    for noise in NOISE:
        for rep in range(REPEATS):
            if all((noise, name, rep) in done for name in e.SETS):
                continue
            if time.time() - t0 > a.budget:
                left += 1
                continue
            m = e.Model(noise=noise, seed=e.SEED + rep)
            top = topology(e, m)
            for name, population in e.SETS.items():
                x, lab = m.ensemble(population)              # always generated, in the same order
                if (noise, name, rep) in done:
                    continue
                traj = md.Trajectory(x.reshape(len(x), -1, 3) / 10.0, top)
                row = dict(noise=noise, set=name, repeat=rep, **one_case(e, traj, lab))
                pd.DataFrame([row]).to_csv(f, mode='a', header=not os.path.exists(f), index=False)
                print(f'noise {noise} repeat {rep} {name}: {row["k"]} cluster(s), known {row["true_k"]}, '
                      f'ARI {row["ari"]:.2f} ({time.time() - t0:.0f} s)', flush=True)
    if left:
        print(f'time budget reached; {left} repeat(s) not finished. Run again.')
        return
    d = pd.read_csv(f).sort_values(['noise', 'repeat'], kind='stable')
    d.to_csv(f, index=False)
    s = summary(d)
    s.to_csv(f'{OUT}/external_validation_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(s.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
