"""
State specificity of the clusters: precision and recall of every cluster with every map of
its protein, all five proteins in one figure. One figure for precision, one for recall.

Numbers: results/cluster_precision_recall/cluster_precision_recall.xlsx, written by
24_cluster_precision_recall.py (definitions there). Rows = clusters of each method,
columns = maps; criterion: density ratio >= BAR (default 0.75).

    precision  frames in the cluster with density ratio >= BAR / all frames in the cluster
               (is the cluster specific to one state?)
    recall     frames in the cluster with density ratio >= BAR / all frames with density ratio >= BAR
               (is the state recovered in one cluster?)

In the precision figure every cluster also has one small number: the highest density ratio of
a frame of the cluster, written in the cell of the map where it is found.

Reads   results/cluster_precision_recall/cluster_precision_recall.xlsx
        results/metric_summary/per_frame_metrics_<method>.csv
Writes  results/figures/fig_precision_all_bar<BAR>.png, fig_recall_all_bar<BAR>.png
Usage: python 26_fig_precision_recall.py [bar]        (bar = 0.75, 0.70 or 0.65; default 0.75)
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))                      # the repository
OUT = f'{ROOT}/results/figures'
os.makedirs(OUT, exist_ok=True)
BAR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
B = f'{BAR:.2f}'
T = pd.read_excel(f'{ROOT}/results/cluster_precision_recall/cluster_precision_recall.xlsx', sheet_name=f'bar_{BAR:.2f}')

METHODS = ['BioEmu', 'AlphaFlow', 'AF2-MSA']
PANELS = [('precision', 'Precision', 'Is the cluster specific to one state?', f'in the cluster and ≥ {B}\n/  all in the cluster'),
          ('recall', 'Recall', 'Is the state recovered in one cluster?', f'in the cluster and ≥ {B}\n/  all ≥ {B}')]
P = {
    'GltPh': dict(maps=['EMD-44530', 'EMD-44529', 'EMD-21986'], cols=['Outward-\nopen', 'Inter-\nmediate', 'Inward-\nopen'], cell=0.6325),
    'SLC37A4': dict(maps=['EMD-67847', 'EMD-62584', 'EMD-64144', 'EMD-64147', 'EMD-66194', 'EMD-66660'],
                    cols=['67847', '62584', '64144', '64147', '66194', '66660'], cell=0.56,
                    groups=[('Outward-open', 0, 1), ('Cytosol-open', 2, 5)]),
    'SPNS2': dict(maps=['EMD-28651', 'EMD-28653'], cols=['Outward-\nopen', 'Outward-\noccluded'], cell=0.95),
    'AuxinTransporter': dict(maps=['EMD-64490', 'EMD-61397'], cols=['Apo\n(Lab 1)', 'Apo\n(Lab 2)'], cell=0.95, name='AUX1'),
    'GPR4': dict(maps=['EMD-60057', 'EMD-61839'], cols=['pH 7.6', 'pH 6.2'], cell=0.95),
}
GREY = '#e6e6e6'
# precision in blue; recall in light green (white to light green), so that the recall matrix does not dominate
from matplotlib.colors import LinearSegmentedColormap
CMAP = {'precision': plt.cm.Blues,
        'recall': LinearSegmentedColormap.from_list('light_greens', plt.cm.Greens(np.linspace(0.0, 0.45, 256)))}
# sizes in inches
LEFT, GAP = 1.90, 0.15                    # LEFT holds the method name and the cluster label
ROW, BLOCK_GAP = 0.3632, 0.14
HEAD, XGAP, RGAP, STRIP, FOOT, MARGIN_L, MARGIN_R = 0.80, 0.28, 0.45, 1.05, 0.42, 0.08, 0.09


def content_height(protein):
    t = T[T.protein == protein]
    return sum(t[t.method == m].cluster.nunique() for m in METHODS) * ROW + (len(METHODS) - 1) * BLOCK_GAP


def block_width(protein, cell, n_panels=2):
    return LEFT + n_panels * len(P[protein]['maps']) * cell + (n_panels - 1) * GAP


def draw_block(fig, fw, fh, x0, y0, letter, protein, cell, panels=PANELS, head=None, show_name=True, best=None):
    """One protein: one matrix per entry of panels. (x0, y0) = top-left corner in inches from the top-left of the figure."""
    head = HEAD if head is None else head
    d = P[protein]; t = T[T.protein == protein]; nc = len(d['maps']); panel = nc * cell
    has_dash = False
    fig.text(x0 / fw, 1 - y0 / fh, f"{letter}   {d.get('name', protein)}", ha='left', va='top', fontsize=14, fontweight='bold')
    for p, (col, name, _, _) in enumerate(panels):
        px = x0 + LEFT + p * (panel + GAP)
        if show_name:
            fig.text((px + panel / 2) / fw, 1 - (y0 + 0.10) / fh, name, ha='center', va='top', fontsize=11.5, fontweight='bold')
        y = y0 + head
        for i, m in enumerate(METHODS):
            g = t[t.method == m]
            M = g.pivot(index='cluster', columns='emdb_id', values=col)[d['maps']]
            n = g.drop_duplicates('cluster').set_index('cluster').cluster_size
            h = len(M) * ROW
            ax = fig.add_axes([px / fw, 1 - (y + h) / fh, panel / fw, h / fh]); ax.set_facecolor(GREY)
            ax.imshow(np.ma.masked_invalid(M.values), vmin=0, vmax=1, cmap=CMAP[col], aspect='auto')
            ax.set_xticks(range(nc)); ax.set_yticks(range(len(M))); ax.tick_params(length=0)
            ax.set_yticklabels([f'{c}  (n = {n[c]})' for c in M.index] if p == 0 else [], fontsize=9.5)
            if i == 0:
                ax.xaxis.tick_top(); ax.set_xticklabels(d['cols'], fontsize=8.2)
                for label, a, b in d.get('groups', []):
                    yy = -0.5 - 0.62 / len(M)
                    ax.annotate('', xy=(a - 0.45, yy), xytext=(b + 0.45, yy), annotation_clip=False, arrowprops=dict(arrowstyle='-', lw=0.8))
                    ax.text((a + b) / 2, yy - 0.08 / len(M), label, ha='center', va='bottom', fontsize=8.5)
            else:
                ax.set_xticklabels([])
            for sp in ax.spines.values(): sp.set_color('black'); sp.set_linewidth(0.5)
            ax.set_xticks(np.arange(-0.5, nc, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(M), 1), minor=True)
            ax.grid(which='minor', color='black', linewidth=0.5); ax.tick_params(which='minor', length=0)
            for r in range(M.shape[0]):
                for k in range(nc):
                    v = M.values[r, k]; has_dash |= bool(np.isnan(v))
                    ax.text(k, r, '\u2013' if np.isnan(v) else f'{v * 100:.0f}%', ha='center', va='center', fontsize=10.5,
                            color='0.45' if np.isnan(v) else ('white' if (col == 'precision' and v > 0.6) else 'black'))
                    hit = best.get((m, protein, M.index[r])) if best else None
                    if hit and hit[0] == d['maps'][k]:          # best frame of this cluster: small number, lower right corner
                        ax.text(k + 0.46, r + 0.45, f'{hit[1]:.2f}', ha='right', va='bottom', fontsize=6.5,
                                color='white' if (col == 'precision' and v > 0.6) else '0.25')
            if p == 0:
                fig.text(x0 / fw, 1 - (y + h / 2) / fh, m, ha='left', va='center', fontsize=10.5, fontweight='bold')
            y += h + BLOCK_GAP
    return has_dash


# ── one figure per quantity: all five proteins ───────────────────────────────
def best_frames():
    """Highest density ratio of a frame of each cluster, and the map where it is found: {(method, protein, cluster): (emdb_id, value)}."""
    root = 'results'
    out = {}
    for key, method in [('bioemu', 'BioEmu'), ('alphaflow', 'AlphaFlow'), ('msa', 'AF2-MSA')]:
        t = pd.read_csv(f'{ROOT}/{root}/metric_summary/per_frame_metrics_{key}.csv')
        for (protein, c), g in t.groupby(['protein', 'cluster']):
            r = g.loc[g.density_ratio.idxmax()]
            out[(method, protein, f'C{c}')] = (r.emdb_id, float(r.density_ratio))
    return out


ALL_ONE = [[('a', 'GltPh', 0.64), ('b', 'SLC37A4', 0.58)],
           [('c', 'GPR4', 0.575), ('d', 'AuxinTransporter', 0.575), ('e', 'SPNS2', 0.575)]]
HEAD_ONE, RGAP_ONE = 0.50, 0.16          # less space between the two rows of proteins


def build_one(metric, show_title=True, show_best=False):
    panel = [q for q in PANELS if q[0] == metric]
    _, name, meaning, sub = panel[0]
    dash = bool(T[metric].isna().any())
    widths = [sum(block_width(pr, c, 1) for _, pr, c in row) + XGAP * (len(row) - 1) for row in ALL_ONE]
    heights = [HEAD_ONE + max(content_height(pr) for _, pr, _ in row) for row in ALL_ONE]
    foot = FOOT if (dash or show_best) else 0.12
    best = best_frames() if show_best else None
    fw = MARGIN_L + max(widths) + MARGIN_R
    up = 0.0 if show_title else 0.30                 # without the title the two lines below it move up
    strip = STRIP - up
    fh = strip + sum(heights) + RGAP_ONE * (len(ALL_ONE) - 1) + foot
    fig = plt.figure(figsize=(fw, fh))
    if show_title:
        fig.text(MARGIN_L / fw, 1 - 0.10 / fh, name, ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(MARGIN_L / fw, 1 - (0.42 - up) / fh, meaning, ha='left', va='top', fontsize=11, color='0.05')
    fig.text(MARGIN_L / fw, 1 - (0.66 - up) / fh, sub.replace('\n', '  '), ha='left', va='top', fontsize=9.2, color='0.35')
    fig.text(1 - MARGIN_R / fw, 1 - 0.14 / fh, f'Criterion: density ratio \u2265 {B}', ha='right', va='top', fontsize=12.5, fontweight='bold', color='#b03a2e')
    y = strip
    for row, hrow in zip(ALL_ONE, heights):
        x = MARGIN_L
        for letter, protein, cell in row:
            draw_block(fig, fw, fh, x, y, letter, protein, cell, panels=panel, head=HEAD_ONE, show_name=False, best=best)
            x += block_width(protein, cell, 1) + XGAP
        y += hrow + RGAP_ONE
    if dash:
        fig.text(MARGIN_L / fw, 0.07 / fh, f'\u2013 : no frame of the ensemble has a density ratio \u2265 {B} with this map.', ha='left', va='bottom', fontsize=9, color='0.3')
    if show_best:
        fig.text(MARGIN_L / fw, 0.07 / fh, 'Small number: highest density ratio of a frame of the cluster, in the cell of its map.', ha='left', va='bottom', fontsize=9, color='0.3')
    out = f"{OUT}/fig_{metric}_all_bar{B}.png"
    fig.savefig(out, dpi=140, facecolor='white'); plt.close(fig); print('saved', out)


build_one('precision', show_title=False, show_best=True)
build_one('recall')
