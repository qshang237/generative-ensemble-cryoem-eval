# -*- coding: utf-8 -*-
"""
Separate figure: the two other density metrics (CC_mask, Q-score), side by side for each
protein, as violins of each cluster on each map (the code of panel c of the main figures),
and below them the agreement of the metrics with the density ratio (Spearman).
Row 1: GltPh, SLC37A4.  Row 2: GPR4, AUX1, SPNS2.  Row 3: agreement.

Reads   results/metric_summary/per_frame_metrics_<method>.csv  (written by 19_metric_summary.py)
Writes  results/figures/fig_other_metrics.png
Usage: python 25_fig_other_metrics.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT = 'results/figures'
os.makedirs(OUT, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr

FS_TITLE, FS_ROW, FS_HEAD, FS_AXIS, FS_TICK, FS_LEG, FS_ANNOT = 30, 26, 21, 22, 17, 13.5, 13
R = 'results/metric_summary/'
MODELS = [('bioemu', 'BioEmu'), ('alphaflow', 'AlphaFlow'), ('msa', 'AF2-MSA')]
METRICS = [('cc_ratio', 'CC_mask'), ('q_ratio', 'Q-score')]
YLIM = (-0.2, 1.0)                 # the same axis in every panel; these two metrics can be below 0
CCOL = ['#1f77b4', '#17becf', '#8c564b']          # C0, C1, C2
# protein -> (name in the tables, maps in the order of the main figure, x tick positions, x tick labels)
P = {
 'GltPh':   ('GltPh', ['EMD-44530', 'EMD-44529', 'EMD-21986'], [0, 1, 2], ['Outward\n-open', 'Intermediate', 'Inward\n-open']),
 'SLC37A4': ('SLC37A4', ['EMD-67847', 'EMD-62584', 'EMD-64144', 'EMD-64147', 'EMD-66194', 'EMD-66660'], [0.5, 3.5], ['Outward-open', 'Cytosol-open']),
 'GPR4':    ('GPR4', ['EMD-60057', 'EMD-61839'], [0, 1], ['pH 7.6', 'pH 6.2']),
 'AUX1':    ('AuxinTransporter', ['EMD-64490', 'EMD-61397'], [0, 1], ['Apo\n(Lab 1)', 'Apo\n(Lab 2)']),
 'SPNS2':   ('SPNS2', ['EMD-28651', 'EMD-28653'], [0, 1], ['Outward\n-open', 'Outward\n-occluded']),
}
T = {key: pd.read_csv(R + f'per_frame_metrics_{key}.csv') for key, _ in MODELS}


def semibold(*text_objs, lw=0.35):
    for t in text_objs:
        t.set_path_effects([pe.withStroke(linewidth=lw, foreground='black'), pe.Normal()])


def violin_panel(fig, ax_e, score_mat, labels, legend):
    """Violins of each cluster on each map: the E block of the main-figure script (same violin widths,
    mean marker with s.d. bar, same rule for the place of the numbers)."""
    n_maps = score_mat.shape[1]
    clusters = sorted(np.unique(labels)); best_k = len(clusters); csizes = {c: int((labels == c).sum()) for c in clusters}
    width = 0.35
    offs = np.linspace(-0.5, 0.5, best_k + 2)[1:-1] if best_k > 1 else [0]
    txt_off = 0.1 if best_k <= 2 else 0.05
    txt_fs = FS_ANNOT if best_k <= 2 else FS_ANNOT - 2.5
    min_gap = 0.09 * (YLIM[1] - YLIM[0]) * 0.6
    cluster_means = {}
    for ci, c in enumerate(clusters):
        positions = np.arange(n_maps) + offs[ci]
        m = labels == c
        data = [score_mat[m, mi] for mi in range(n_maps)]
        vw = (width / max(1, best_k - 1) * 1.4) if best_k > 1 else width
        parts = ax_e.violinplot(data, positions=positions, widths=vw, showmeans=False, showextrema=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(CCOL[ci]); pc.set_edgecolor(CCOL[ci]); pc.set_alpha(0.55)
        means = [d.mean() for d in data]; sds = [d.std() for d in data]
        ax_e.errorbar(positions, means, yerr=sds, fmt='o', color=CCOL[ci], markersize=6.5, markeredgecolor='black', markeredgewidth=0.8,
                      capsize=4.5, linewidth=1.7, zorder=5, label=f'C{c}(n={csizes[c]})')
        cluster_means[ci] = (positions, means)
    ax_e.set_ylim(*YLIM); ax_e.set_xlim(-0.6, n_maps - 0.4)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()

    def _marker_bbox(x_data, y_data, r_px=8):
        px, py = ax_e.transData.transform((x_data, y_data))
        return Bbox.from_extents(px - r_px, py - r_px, px + r_px, py + r_px)

    def _overlaps(b1, b2, pad=2.0):
        return not (b1.x1 + pad < b2.x0 or b2.x1 + pad < b1.x0 or b1.y1 + pad < b2.y0 or b2.y1 + pad < b1.y0)

    avoid_boxes = [_marker_bbox(x, y) for ci in range(best_k) for x, y in zip(*cluster_means[ci])]
    for mi in range(n_maps):
        entries = sorted(((ci, cluster_means[ci][0][mi], cluster_means[ci][1][mi]) for ci in range(best_k)), key=lambda e: -e[2])
        prev_y = None
        for ci, x, mn in entries:
            ty = mn if prev_y is None else min(mn, prev_y - min_gap)
            prev_y = ty
            ty = max(ty, YLIM[0] + 0.035)                 # keep the number inside the panel
            s = f'{mn:.2f}'
            candidates = [('left', x + txt_off), ('right', x - txt_off), ('left', x + txt_off * 2.2), ('right', x - txt_off * 2.2)]
            t, bb, clean = None, None, False
            for ha, tx in candidates:
                t = ax_e.text(tx, ty, s, ha=ha, va='center', fontsize=txt_fs)
                bb = t.get_window_extent(renderer=renderer)
                if not any(_overlaps(bb, ob) for ob in avoid_boxes):
                    clean = True
                    break
                t.remove()
            if not clean:
                t = ax_e.text(x + txt_off, ty, s, ha='left', va='center', fontsize=txt_fs)
                bb = t.get_window_extent(renderer=renderer)
            semibold(t); avoid_boxes.append(bb)
    ax_e.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_e.axhline(1.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5); ax_e.axhline(0, color='0.6', lw=0.7)
    for spine in ax_e.spines.values():
        spine.set_linewidth(0.7)
    if legend:
        narrow = best_k == 3 and n_maps <= 3            # three entries in a narrow panel: smaller text
        leg = ax_e.legend(fontsize=FS_LEG - (3.5 if narrow else 0), loc='upper right', ncol=best_k, framealpha=0.92, handletextpad=0.05,
                          columnspacing=0.4 if narrow else 0.6, borderaxespad=0.2, borderpad=0.3)
        semibold(*leg.get_texts()); leg.get_frame().set_linewidth(0.7)


def protein_block(fig, spec, protein, letter, row_labels):
    table_name, maps, xt, xl = P[protein]
    inner = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=spec, hspace=0.10, wspace=0.07)
    for ri, (key, model) in enumerate(MODELS):
        g = T[key][T[key].protein == table_name]
        labels = g.drop_duplicates('frame').sort_values('frame').cluster.values
        for j, (col, mname) in enumerate(METRICS):
            ax = fig.add_subplot(inner[ri, j])
            violin_panel(fig, ax, g.pivot(index='frame', columns='emdb_id', values=col)[maps].values, labels, legend=(j == 0))
            ax.set_xticks(xt)
            if ri == 2:
                semibold(*ax.set_xticklabels(xl, fontsize=FS_TICK - (3.5 if protein == 'GltPh' else 1)))
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.tick_params(axis='y', labelsize=FS_TICK); semibold(*ax.get_yticklabels())
            else:
                ax.set_yticklabels([])
            p = ax.get_position()
            if ri == 0:
                semibold(fig.text((p.x0 + p.x1) / 2, p.y1 + 0.004, mname, fontsize=FS_HEAD, ha='center', va='bottom'))
                if j == 0:
                    fig.text(p.x0 - 0.012, p.y1 + 0.019, letter, fontsize=FS_TITLE, fontweight='bold', ha='right', va='bottom')
                    fig.text(p.x0, p.y1 + 0.020, protein, fontsize=FS_TITLE - 4, fontweight='bold', ha='left', va='bottom')
            if row_labels and j == 0:
                fig.text(0.018, (p.y0 + p.y1) / 2, model, ha='center', va='center', rotation=90, fontsize=FS_ROW, fontweight='bold')


fig = plt.figure(figsize=(27, 36))
outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 3, 1.6], hspace=0.20, left=0.065, right=0.99, top=0.955, bottom=0.035)
top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[3 + 2.0, 6 + 0.9], wspace=0.09)
protein_block(fig, top[0], 'GltPh', 'a', True)
protein_block(fig, top[1], 'SLC37A4', 'b', False)
bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.13)
for i, (prot, letter) in enumerate([('GPR4', 'c'), ('AUX1', 'd'), ('SPNS2', 'e')]):
    protein_block(fig, bot[i], prot, letter, i == 0)

rows = []
for key, model in MODELS:                       # one value per cluster and map: mean (RMSD: median, it has a long tail)
    gb = T[key].groupby(['protein', 'cluster', 'emdb_id'])
    m = gb[['density_ratio', 'cc_ratio', 'q_ratio']].mean().reset_index(); m['rmsd'] = gb.rmsd.median().values; m['model'] = model
    rows.append(m)
M = pd.concat(rows)
gsc = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.28)
COLM = {'BioEmu': 'tab:blue', 'AlphaFlow': 'tab:orange', 'AF2-MSA': 'tab:green'}
for j, (col, lab, ylim, scale) in enumerate([('cc_ratio', 'CC_mask', (-0.1, 1.0), None), ('q_ratio', 'Q-score', (-0.25, 1.0), None),
                                              ('rmsd', 'Cα-RMSD (Å), median', (40, 1.5), 'log')]):
    ax = fig.add_subplot(gsc[0, j])
    for model, c in COLM.items():
        h = M[M.model == model]; ax.scatter(h.density_ratio, h[col], s=130, color=c, alpha=0.8, edgecolor='white', linewidth=0.8, label=model)
    if scale:
        ax.set_yscale(scale); ax.set_yticks([2, 3, 5, 10, 20, 40]); ax.set_yticklabels(['2', '3', '5', '10', '20', '40']); ax.minorticks_off()
    ax.set_ylim(*ylim); ax.set_xlim(0, 1)
    semibold(ax.set_xlabel('Density ratio', fontsize=FS_AXIS), ax.set_ylabel(lab, fontsize=FS_AXIS))
    semibold(ax.set_title(f'Spearman r = {abs(spearmanr(M.density_ratio, M[col])[0]):.2f}', fontsize=FS_AXIS - 1, pad=10))
    ax.tick_params(labelsize=FS_TICK); semibold(*ax.get_xticklabels(), *ax.get_yticklabels())
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    if j == 0:
        leg = ax.legend(fontsize=FS_LEG + 3, loc='upper left'); semibold(*leg.get_texts()); leg.get_frame().set_linewidth(0.7)
        p = ax.get_position(); fig.text(p.x0 - 0.012, p.y1 + 0.022, 'f', fontsize=FS_TITLE, fontweight='bold', ha='right', va='bottom')
        fig.text(p.x0, p.y1 + 0.023, f'All five proteins: agreement with the density ratio (one dot = one cluster on one map, {len(M)} dots)',
                 fontsize=FS_TITLE - 6, fontweight='bold', ha='left', va='bottom')
out = f'{OUT}/fig_other_metrics.png'
fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white'); print('saved', out)
