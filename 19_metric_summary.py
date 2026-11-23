"""
The same three results for every frame-level metric.

Result 1  Rank agreement with the density ratio (Spearman, for each map and method).
Result 2  The best frame: is it the same frame as by the density ratio, how good is it,
          and which method has the best one.
Result 3  Fraction of frames that represent each state (proteins with different states).

Metrics   density ratio          results/<method>/per_frame_density.xlsx
          RMSD to deposited      17_rmsd_to_deposited.py
          CC_mask ratio          18_cc_mask.py
          Q-score ratio          20_q_score.py   (used if its output exists)
          "ratio" = value of the frame / value of the deposited model on its own map.

Result 3  The score of a frame for a state is the mean over the maps of that state.
          A frame represents state X if (a) X is its best state and (b) the score for X
          passes the bar of the metric. Condition (a) makes sure that no frame is counted
          for two states. The bars are in METRICS. Each bar is stricter than the best value
          that a deposited model of a DIFFERENT state reaches (sheet 'deposited_cross'):
          a frame that passes fits the map better than any wrong-state deposited model.

Usage: python 19_metric_summary.py
Outputs: results/metric_summary/
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.stats import spearmanr

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import shared_fitting as sf

OUT = 'results/metric_summary'
METHODS = {'bioemu': 'BioEmu', 'alphaflow': 'AlphaFlow', 'msa': 'AF2-MSA'}
# col = column of the per-frame table; higher = True if a higher value is a better fit;
# bar = limit for "represents the state"; alt = other limits, to show the effect of the bar
METRICS = {
    'density_ratio': dict(col='density_ratio', higher=True, bar=0.70, alt=(0.65, 0.75), label='Density ratio'),
    'rmsd':          dict(col='rmsd', higher=False, bar=3.0, alt=(2.5, 2.0), label='RMSD (A)'),
    'cc_mask':       dict(col='cc_ratio', higher=True, bar=0.50, alt=(0.40, 0.60), label='CC_mask ratio'),
    'q_score':       dict(col='q_ratio', higher=True, bar=0.50, alt=(0.40, 0.60), label='Q-score ratio'),
}


def _module(script):
    spec = importlib.util.spec_from_file_location('m' + script[:2], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ss = _module('09_state_specificity.py')
STATES, STATE_ORDER = _ss.STATES, _ss.STATE_ORDER       # map -> state; proteins with different states


# ── per-frame table ──────────────────────────────────────────────────────────
def per_frame(key):
    t = pd.read_csv(f'results/cc_mask/{key}_density_ratio_rmsd_cc_mask.csv')
    fq = f'results/q_score/{key}_q_score.csv'
    if os.path.exists(fq):
        q = pd.read_csv(fq)[['protein', 'frame', 'emdb_id', 'q_score', 'q_ratio']]
        n = len(t)
        t = t.merge(q, on=['protein', 'frame', 'emdb_id'], how='left')
        assert len(t) == n and t.q_score.notna().all(), f'{key}: Q-score frames do not match'
    t['state'] = [STATES[p][e] for p, e in zip(t.protein, t.emdb_id)]
    return t


# ── deposited models on the maps of the other states ─────────────────────────
def deposited_cross(df, metrics):
    """Value of every deposited model (superposed) on every map of its protein, per metric."""
    f = f'{OUT}/deposited_cross.csv'
    have = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=['protein', 'model_emdb', 'map_emdb', 'metric', 'value'])
    for name in metrics:
        if (have.metric == name).any():
            continue
        new = []
        for p in STATE_ORDER:
            rows = df[df.protein_label == p]
            if name == 'density_ratio':
                M = _ss.deposited_cross_dr(p, df)
            elif name == 'rmsd':
                r17 = _module('17_rmsd_to_deposited.py')
                M = pd.DataFrame({ri.emdb_id: r17.rmsd_to_models(md.load(sf.download_pdb(ri.pdb_id)), rows)[0].iloc[0]
                                  for _, ri in rows.iterrows()}).T
            else:
                mod = _module('18_cc_mask.py' if name == 'cc_mask' else '20_q_score.py')
                M = mod.deposited_cross(rows)
            new += [dict(protein=p, model_emdb=i, map_emdb=j, metric=name, value=M.loc[i, j])
                    for i in M.index for j in M.columns]
        have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
        have.to_csv(f, index=False)                    # saved after each metric
    return have


def bar_check(cross, metrics):
    """For each metric: best value of a wrong-state deposited model, next to the bar."""
    rows = []
    for name in metrics:
        m = METRICS[name]
        c = cross[cross.metric == name].copy()
        c['same'] = [STATES[p][a] == STATES[p][b] for p, a, b in zip(c.protein, c.model_emdb, c.map_emdb)]
        own = c[c.model_emdb == c.map_emdb].set_index('map_emdb').value
        if m['higher'] and name != 'density_ratio':
            c['value'] = c.value / c.map_emdb.map(own)          # as a ratio to the own model
        for p, g in c.groupby('protein'):
            wrong, right = g[~g.same].value, g[g.same & (g.model_emdb != g.map_emdb)].value
            rows.append(dict(metric=m['label'], protein=p, bar=m['bar'],
                             wrong_state_best=wrong.max() if m['higher'] else wrong.min(),
                             wrong_state_worst=wrong.min() if m['higher'] else wrong.max(),
                             same_state_other_map=(f'{right.min():.2f} to {right.max():.2f}' if len(right) else ''),
                             bar_is_stricter=bool(m['bar'] > wrong.max() if m['higher'] else m['bar'] < wrong.min())))
    return pd.DataFrame(rows)


# ── the four results ─────────────────────────────────────────────────────────
def result12(T, metrics):
    rows = []
    for label, t in T.items():
        for (p, e), g in t.groupby(['protein', 'emdb_id'], sort=False):
            g = g.reset_index(drop=True)
            best_dr = g.density_ratio.idxmax()
            for name in metrics:
                if name == 'density_ratio':
                    continue
                m = METRICS[name]
                v = g[m['col']] if m['higher'] else -g[m['col']]
                best = v.idxmax()
                rows.append(dict(metric=m['label'], method=label, protein=p, emdb_id=e, n_frames=len(g),
                                 spearman_with_density_ratio=spearmanr(g[m['col']], g.density_ratio)[0],
                                 same_best_frame=bool(best == best_dr),
                                 rank_of_best_frame_by_density_ratio=int((g.density_ratio > g.density_ratio[best]).sum() + 1)))
    return pd.DataFrame(rows)


def state_scores(t, protein, col):
    g = t[t.protein == protein]
    return g.groupby(['frame', 'state'])[col].mean().unstack()[STATE_ORDER[protein]]


def result3(T, metrics):
    rows = []
    for name in metrics:
        m = METRICS[name]
        for label, t in T.items():
            for p in STATE_ORDER:
                w = state_scores(t, p, m['col'])
                best = w.idxmax(axis=1) if m['higher'] else w.idxmin(axis=1)
                for bar in (m['bar'],) + m['alt']:
                    ok = (w >= bar) if m['higher'] else (w <= bar)
                    for s in w.columns:
                        rows.append(dict(metric=m['label'], bar=bar, main_bar=bar == m['bar'], protein=p, state=s,
                                         method=label, n_frames=len(w),
                                         percent=100 * (ok[s] & (best == s)).mean(),
                                         percent_without_best_state_rule=100 * ok[s].mean(),
                                         frames_that_pass_for_two_states=int((ok.sum(axis=1) >= 2).sum())))
    return pd.DataFrame(rows)


def best_values(T, metrics):
    """Best frame for each state (proteins with one state: all maps together)."""
    rows = []
    for name in metrics:
        m = METRICS[name]
        for label, t in T.items():
            t = t.assign(group=[s if p in STATE_ORDER else 'all maps' for p, s in zip(t.protein, t.state)])
            for (p, s), g in t.groupby(['protein', 'group'], sort=False):
                v = g[m['col']]
                rows.append(dict(metric=m['label'], protein=p, state=s, method=label,
                                 best=v.max() if m['higher'] else v.min(),
                                 worst=v.min() if m['higher'] else v.max(),
                                 median=v.median()))
    return pd.DataFrame(rows)


def method_comparison(T, metrics, top_frac=0.05):
    """For each map: which method has the best frame, the best top-5 % mean and the best
    median. Values are compared after rounding (ratios 2 decimals, RMSD 1 decimal);
    equal values are a tie."""
    rows = []
    for name in metrics:
        m = METRICS[name]
        dec = 2 if m['higher'] else 1
        maps = next(iter(T.values()))[['protein', 'emdb_id']].drop_duplicates().values
        for p, e in maps:
            stat = {}
            for label, t in T.items():
                v = t[(t.protein == p) & (t.emdb_id == e)][m['col']]
                v = v if m['higher'] else -v
                k = max(1, int(round(top_frac * len(v))))
                stat[label] = dict(best=v.max(), top5=v.nlargest(k).mean(), median=v.median())
            for s in ('best', 'top5', 'median'):
                vals = pd.Series({lab: round(stat[lab][s], dec) for lab in stat})
                top = vals[vals == vals.max()].index.tolist()
                rows.append(dict(metric=m['label'], protein=p, emdb_id=e, statistic=s,
                                 winner=top[0] if len(top) == 1 else 'tie',
                                 **{lab: (v if m['higher'] else -v) for lab, v in vals.items()}))
    return pd.DataFrame(rows)


def wide(d, values, index=('protein', 'state')):
    return d.pivot_table(index=['metric'] + list(index), columns='method', values=values, sort=False)[list(METHODS.values())]


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option('display.width', 250)
    df = pd.read_excel(sf.DATASET)
    T = {lab: per_frame(k) for k, lab in METHODS.items()}
    metrics = [n for n, m in METRICS.items() if all(m['col'] in t.columns for t in T.values())]
    for k, lab in METHODS.items():
        T[lab].to_csv(f'{OUT}/per_frame_metrics_{k}.csv', index=False)

    cross = deposited_cross(df, metrics)
    bars = bar_check(cross, metrics)
    r12, r3 = result12(T, metrics), result3(T, metrics)
    bv = best_values(T, metrics)
    mc = method_comparison(T, metrics)
    with pd.ExcelWriter(f'{OUT}/metric_summary.xlsx') as xl:
        bars.to_excel(xl, 'bars', index=False)
        r12.to_excel(xl, 'result1_2_by_map', index=False)
        bv.to_excel(xl, 'result2_best_frame_value', index=False)
        mc.to_excel(xl, 'result2_best_method_by_map', index=False)
        r3[r3.main_bar].to_excel(xl, 'result3_states', index=False)
        r3.to_excel(xl, 'result3_other_bars', index=False)
        cross.to_excel(xl, 'deposited_cross', index=False)

    print('BARS (each bar must be stricter than the best wrong-state deposited model)')
    print(bars.round(2).to_string(index=False))
    print('\nRESULT 1  Spearman with the density ratio, over the 15 maps')
    print(r12.groupby(['metric', 'method'], sort=False).spearman_with_density_ratio.agg(['median', 'min', 'max']).round(2).to_string())
    print('\nRESULT 2  best frame = best frame by density ratio (number of maps, of 15)')
    print(r12.groupby(['metric', 'method'], sort=False).same_best_frame.sum().unstack()[list(METHODS.values())].to_string())
    print('\n          value of the best frame')
    print(wide(bv, 'best').round(2).to_string())
    print('\n          number of maps (of 15) where the method is the best')
    print(mc.groupby(['metric', 'statistic', 'winner'], sort=False).size().unstack(fill_value=0).to_string())
    print('\nRESULT 3  percent of frames that represent each state')
    print(wide(r3[r3.main_bar], 'percent').round(0).to_string())
    print('\n          frames that pass the bar for two states (before the best-state rule)')
    print(r3[r3.main_bar].drop_duplicates(['metric', 'protein', 'method']).pivot_table(
        index=['metric', 'protein'], columns='method', values='frames_that_pass_for_two_states', sort=False)[list(METHODS.values())].to_string())


if __name__ == '__main__':
    main()
