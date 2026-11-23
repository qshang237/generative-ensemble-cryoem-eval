"""
State specificity of the clusters: precision and recall of every cluster with every map
of its protein.

One ensemble = one method and one protein. For one cluster C and one map M of that
protein, every frame of the ensemble is one item:

    judged positive    the frame is in cluster C
    judged negative    the frame is not in cluster C
    actually positive  density ratio of the frame with map M >= BAR  (it fits this conformation)
    actually negative  density ratio of the frame with map M <  BAR

    TP  in C,     density ratio >= BAR
    FP  in C,     density ratio <  BAR
    FN  not in C, density ratio >= BAR
    TN  not in C, density ratio <  BAR

    precision = TP / (TP + FP)   share of the frames of the cluster that fit the map
                                 (is the cluster specific to one state?)
    recall    = TP / (TP + FN)   share of the frames that fit the map that are in the cluster
                                 (is the state recovered in one cluster?)
                                 Empty when no frame of the ensemble reaches the bar.

Every cluster is calculated with every map of its protein; no cluster is assigned to a
conformation by hand. With one cluster, TP + FN are all in that cluster, so recall is 1
whenever it is defined.

Reads   <results>/metric_summary/per_frame_metrics_<method>.csv   (19_metric_summary.py)
Writes  <results>/cluster_precision_recall/cluster_precision_recall.xlsx   (one sheet per bar)
Usage: python 24_cluster_precision_recall.py [--results results]
"""
import os
import argparse
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
METHODS = {'bioemu': 'BioEmu', 'alphaflow': 'AlphaFlow', 'msa': 'AF2-MSA'}
BARS = (0.75, 0.70, 0.65)         # 0.75 is the bar of the figures; the others show its effect


def table(results, bar):
    rows = []
    for key, method in METHODS.items():
        t = pd.read_csv(f'{results}/metric_summary/per_frame_metrics_{key}.csv')
        for protein, g in t.groupby('protein', sort=False):
            dr = g.pivot(index='frame', columns='emdb_id', values='density_ratio')
            dr = dr[list(dict.fromkeys(g.emdb_id))]                      # maps in the order of the data table
            state = g.drop_duplicates('emdb_id').set_index('emdb_id').state
            cl = g.drop_duplicates('frame').set_index('frame').cluster.reindex(dr.index)
            for c in sorted(cl.unique()):
                inside = (cl == c).values
                for m in dr.columns:
                    high = (dr[m] >= bar).values
                    tp, fp = int((inside & high).sum()), int((inside & ~high).sum())
                    fn, tn = int((~inside & high).sum()), int((~inside & ~high).sum())
                    rows.append(dict(method=method, protein=protein, n_frames=len(dr),
                                     n_clusters=int(cl.nunique()), cluster=f'C{c}', cluster_size=tp + fp,
                                     emdb_id=m, state=state[m], frames_that_fit_map=tp + fn,
                                     TP=tp, FP=fp, FN=fn, TN=tn,
                                     precision=tp / (tp + fp), recall=tp / (tp + fn) if tp + fn else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results', help='folder with the results of the analysis')
    a = ap.parse_args()
    out = f'{a.results}/cluster_precision_recall'
    os.makedirs(out, exist_ok=True)
    with pd.ExcelWriter(f'{out}/cluster_precision_recall.xlsx') as xl:
        for bar in BARS:
            t = table(a.results, bar)
            t.to_excel(xl, sheet_name=f'bar_{bar:.2f}', index=False)
            print(f'bar {bar:.2f}: {len(t)} cluster-map pairs')
    print('written', f'{out}/cluster_precision_recall.xlsx')


if __name__ == '__main__':
    main()
