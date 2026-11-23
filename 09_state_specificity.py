"""
State labels of the maps, and the reference values for state specificity.

STATES        experimental state of every map. Maps of one state have the same label.
              Same-state controls (GPR4, AUX1) and the two SPNS2 sub-states have one label
              per map.
STATE_ORDER   the proteins with clearly different states, and the order of these states.
deposited_cross_dr(protein, df)
              density ratio of every deposited model with every map of its protein,
              calculated exactly as for a generated frame: superposition on the deposited
              model of the map (sequence-aligned common Calpha), mean smoothed and z-scored
              density at the backbone atoms N, CA, C, O, divided by the value of the map's
              own deposited model. The diagonal is 1 by construction. The table shows how
              well a structure of one state fits the map of another state.

Used by 19_metric_summary.py.

Usage: python 09_state_specificity.py
Outputs: results/state_specificity/deposited_cross_density_ratio_<protein>.csv
"""
import os
import pandas as pd
import mdtraj as md

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import shared_fitting as sf

OUT = 'results/state_specificity'

# map -> state (Table S1). Same-state controls use the condition as pseudo-state.
STATES = {
    'GltPh': {'EMD-44530': 'Outward-open', 'EMD-44529': 'Intermediate',
              'EMD-21986': 'Inward-open'},
    'SLC37A4': {'EMD-67847': 'Outward-open', 'EMD-62584': 'Outward-open',
                'EMD-64144': 'Cytosol-open', 'EMD-64147': 'Cytosol-open',
                'EMD-66194': 'Cytosol-open', 'EMD-66660': 'Cytosol-open'},
    'SPNS2': {'EMD-28651': 'Substate EMD-28651', 'EMD-28653': 'Substate EMD-28653'},
    'GPR4': {'EMD-60057': 'pH 7.6', 'EMD-61839': 'pH 6.2'},
    'AuxinTransporter': {'EMD-64490': 'Apo EMD-64490', 'EMD-61397': 'Apo EMD-61397'},
}
STATE_ORDER = {'GltPh': ['Outward-open', 'Intermediate', 'Inward-open'],
               'SLC37A4': ['Outward-open', 'Cytosol-open']}


def deposited_cross_dr(protein, df):
    """Density ratio of every deposited model (rows) on every map (columns)."""
    rows = df[df.protein_label == protein]
    maps = {}
    for _, r in rows.iterrows():
        grid = sf.load_map(sf.download_map(r.emdb_id), r.resolution)
        ref = md.load(sf.download_pdb(r.pdb_id))
        dref = sf.mean_density_at_backbone(ref.xyz[0], ref.topology.select('name N CA C O'), grid)
        maps[r.emdb_id] = dict(grid=grid, dref=dref, pdb=sf.download_pdb(r.pdb_id))
    M = pd.DataFrame(index=rows.emdb_id, columns=rows.emdb_id, dtype=float)
    for _, ri in rows.iterrows():
        for ej, d in maps.items():
            t = md.load(sf.download_pdb(ri.pdb_id))
            sf.superpose_to_target(t, d['pdb'])
            M.loc[ri.emdb_id, ej] = sf.mean_density_at_backbone(
                t.xyz[0], t.topology.select('name N CA C O'), d['grid']) / d['dref']
    M.index.name = 'model_of'
    return M


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_excel(sf.DATASET)
    for protein in STATES:
        M = deposited_cross_dr(protein, df)
        M.to_csv(f'{OUT}/deposited_cross_density_ratio_{protein}.csv')
        print(f'{protein}: density ratio of the deposited models (rows) with the maps (columns)\n{M.round(2)}\n', flush=True)


if __name__ == '__main__':
    main()
