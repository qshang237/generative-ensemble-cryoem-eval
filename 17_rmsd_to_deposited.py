"""
Calpha-RMSD of every generated frame to every deposited model of its protein.

This is the conventional, coordinate-based reference used by the generators' own
benchmarks. Here it is a cross-check of the density-based scores, not a main metric: the
deposited model was built into the map, so the two are not independent.

Atoms     Calpha of residues resolved in every deposited model of the protein
          (sequence-aligned correspondence), so the RMSDs to different states of one
          protein use the same atoms and can be compared.
RMSD      after optimal superposition of the frame on the deposited model.

Usage: python 17_rmsd_to_deposited.py            (skips ensembles already done)
Outputs: results/rmsd_to_deposited/
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import mdtraj as md

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import shared_fitting as sf

OUT = 'results/rmsd_to_deposited'
METHODS = [('bioemu', 'BioEmu', '04_fitting_bioemu.py', 'load_bioemu'),
           ('alphaflow', 'AlphaFlow', '04_fitting_alphaflow.py', 'load_alphaflow'),
           ('msa', 'AF2-MSA', '04_fitting_msa.py', 'load_msa')]


def _load(script, fn):
    spec = importlib.util.spec_from_file_location(script[:-3], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


def ca_traj(xyz_nm):
    top = md.Topology()
    ch = top.add_chain()
    for _ in range(xyz_nm.shape[1]):
        top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    return md.Trajectory(xyz_nm, top)


def rmsd_to_models(traj, rows):
    """DataFrame frame x emdb_id of Calpha-RMSD (A) on the residues common to all models."""
    maps = {}
    for _, r in rows.iterrows():
        m = md.load(sf.download_pdb(r.pdb_id))
        ti, mi = sf.seqaln_common_ca(traj.topology, m.topology)
        maps[r.emdb_id] = (dict(zip(ti.tolist(), mi.tolist())), m)
    core = sorted(set.intersection(*[set(v[0]) for v in maps.values()]))
    frames = ca_traj(traj.xyz[:, core])
    out = {}
    for emdb, (corr, m) in maps.items():
        ref = ca_traj(m.xyz[:1, [corr[i] for i in core]])
        out[emdb] = md.rmsd(frames, ref, 0) * 10
    return pd.DataFrame(out), len(core)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_excel(sf.DATASET)
    for key, label, script, fn in METHODS:
        loader = _load(script, fn)
        for protein in df.protein_label.unique():
            f = f'{OUT}/{key}_{protein}.csv'
            if os.path.exists(f):
                continue
            traj = loader(protein)
            if traj is None:
                continue
            wide, n_core = rmsd_to_models(traj, df[df.protein_label == protein])
            long = wide.rename_axis('frame').reset_index().melt('frame', var_name='emdb_id', value_name='rmsd')
            long.insert(0, 'protein', protein)
            long['n_core_ca'] = n_core
            long.to_csv(f, index=False)
            print(f'{label} {protein}: {len(wide)} frames, {n_core} Calpha; '
                  + ', '.join(f'{c} min {wide[c].min():.2f} median {wide[c].median():.2f}' for c in wide.columns),
                  flush=True)
    # merge with the density ratio
    for key, label, _, _ in METHODS:
        parts = [pd.read_csv(f'{OUT}/{key}_{p}.csv') for p in df.protein_label.unique()
                 if os.path.exists(f'{OUT}/{key}_{p}.csv')]
        if not parts:
            continue
        r = pd.concat(parts)
        d = pd.read_excel(f'results/{key}/per_frame_density.xlsx')
        m = d.merge(r, on=['protein', 'frame', 'emdb_id'], how='left')
        assert m.rmsd.notna().all(), f'{key}: frames without RMSD'
        m.to_csv(f'{OUT}/{key}_density_ratio_and_rmsd.csv', index=False)


if __name__ == '__main__':
    main()
