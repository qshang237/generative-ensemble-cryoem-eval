"""
Statistical comparison of the three ensemble-generation methods on density ratio.

The single best frame per map is an extreme-value statistic: it depends on
ensemble size (BioEmu yields 335-490 frames, AlphaFlow and AF2-MSA 500) and
has no uncertainty attached. This script reports, for every (protein, map,
method):

  max        best single frame (as in the manuscript)
  max_eqn    expected best frame when every method is subsampled to the
             smallest ensemble size for that protein (without replacement)
  top1       mean of the top 1% of frames (ceil(0.01 n))
  top5       mean of the top 5% of frames (ceil(0.05 n))
  median     median over all frames

with 95% percentile bootstrap intervals for top1, top5 and median (max and
max_eqn are point values only, because the bootstrap is not valid for a
sample maximum; frames resampled with replacement;
the same resampled frames are used for every map of a protein, because each
frame is scored against all of that protein's maps).

Pairwise method differences are reported per map and per protein (mean over
that protein's maps, each protein weighted equally in the overall summary).
Intervals reflect frame-level sampling only: each method was run once per
protein, so run-to-run variation of the generators is not captured, and with
five proteins no protein-level significance test is meaningful.

Outputs: results/method_comparison/
"""
import os
import numpy as np
import pandas as pd

OUT = 'results/method_comparison'
METHODS = ['bioemu', 'alphaflow', 'msa']
LABEL = {'bioemu': 'BioEmu', 'alphaflow': 'AlphaFlow', 'msa': 'AF2-MSA'}
PAIRS = [('bioemu', 'alphaflow'), ('bioemu', 'msa'), ('alphaflow', 'msa')]
STATS = ['max', 'max_eqn', 'top1', 'top5', 'median']
NO_CI = ('max', 'max_eqn')
B = 5000
B_EQN = 2000
SEED = 20261113


def topk_mean(x, frac, axis):
    """Mean of the top ceil(frac * n) values along `axis`."""
    n = x.shape[axis]
    k = int(np.ceil(frac * n))
    part = np.partition(x, n - k, axis=axis)
    return np.take(part, np.arange(n - k, n), axis=axis).mean(axis=axis)


def point_stats(F, n_min, rng):
    """F: frames x maps. Returns dict stat -> array(maps)."""
    n = F.shape[0]
    out = {
        'max': F.max(0),
        'top1': topk_mean(F, 0.01, 0),
        'top5': topk_mean(F, 0.05, 0),
        'median': np.median(F, 0),
    }
    # expected max at a common ensemble size
    keys = rng.random((B_EQN, n)).argsort(1)[:, :n_min]          # B_EQN x n_min
    out['max_eqn_draws'] = F[keys].max(1)                         # B_EQN x maps
    out['max_eqn'] = out['max_eqn_draws'].mean(0)
    return out


def boot_stats(F, rng):
    """Bootstrap draws (B x maps) for top1, top5, median, max."""
    n = F.shape[0]
    idx = rng.integers(0, n, (B, n))
    S = F[idx]                                                    # B x n x maps
    return {
        'max': S.max(1),
        'top1': topk_mean(S, 0.01, 1),
        'top5': topk_mean(S, 0.05, 1),
        'median': np.median(S, 1),
    }


def ci(draws):
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pf = {m: pd.read_excel(f'results/{m}/per_frame_density.xlsx') for m in METHODS}
    proteins = sorted(pf['bioemu'].protein.unique())

    rows, diff_rows, prot_rows = [], [], []
    overall_draws = {s: {p: [] for p in PAIRS} for s in STATS}

    for prot in proteins:
        wide = {m: pf[m][pf[m].protein == prot]
                .pivot(index='frame', columns='emdb_id', values='density_ratio')
                .sort_index(axis=1) for m in METHODS}
        maps = list(wide['bioemu'].columns)
        for m in METHODS:
            assert list(wide[m].columns) == maps and not wide[m].isna().any().any()
        F = {m: wide[m].values for m in METHODS}
        n_min = min(F[m].shape[0] for m in METHODS)

        pt, bs = {}, {}
        for m in METHODS:
            pt[m] = point_stats(F[m], n_min, rng)
            bs[m] = boot_stats(F[m], rng)
            bs[m]['max_eqn'] = pt[m]['max_eqn_draws']             # subsampling spread
            for j, emdb in enumerate(maps):
                r = dict(protein=prot, emdb_id=emdb, method=LABEL[m],
                         n_frames=F[m].shape[0], n_common=n_min)
                for s in STATS:
                    lo, hi = ci(bs[m][s])
                    r[s] = pt[m][s][j]
                    r[f'{s}_lo'], r[f'{s}_hi'] = lo[j], hi[j]
                rows.append(r)

        for a, b in PAIRS:
            for s in STATS:
                d_pt = pt[a][s] - pt[b][s]                         # maps
                if s == 'max_eqn':
                    k = min(len(bs[a][s]), len(bs[b][s]))
                    d_bs = bs[a][s][:k] - bs[b][s][:k]
                else:
                    d_bs = bs[a][s] - bs[b][s]                     # B x maps
                lo, hi = ci(d_bs)
                for j, emdb in enumerate(maps):
                    diff_rows.append(dict(protein=prot, emdb_id=emdb, stat=s,
                                          comparison=f'{LABEL[a]} - {LABEL[b]}',
                                          diff=d_pt[j], lo=lo[j], hi=hi[j],
                                          excludes_zero=bool(lo[j] > 0 or hi[j] < 0)))
                pm = d_bs.mean(1)
                plo, phi = np.percentile(pm, [2.5, 97.5])
                prot_rows.append(dict(protein=prot, stat=s, n_maps=len(maps),
                                      comparison=f'{LABEL[a]} - {LABEL[b]}',
                                      diff=d_pt.mean(), lo=plo, hi=phi,
                                      excludes_zero=bool(plo > 0 or phi < 0)))
                overall_draws[s][(a, b)].append((d_pt.mean(), pm))

    per_map = pd.DataFrame(rows)
    diffs = pd.DataFrame(diff_rows)
    per_prot = pd.DataFrame(prot_rows)

    # The bootstrap is not valid for a sample maximum (a resampled maximum can
    # never exceed the observed one), so max and max_eqn are point values only.
    for s in NO_CI:
        per_map[[f'{s}_lo', f'{s}_hi']] = np.nan
        for df_ in (diffs, per_prot):
            df_.loc[df_.stat == s, ['lo', 'hi', 'excludes_zero']] = np.nan

    overall = []
    for s in STATS:
        for a, b in PAIRS:
            pts = np.array([x[0] for x in overall_draws[s][(a, b)]])
            k = min(len(x[1]) for x in overall_draws[s][(a, b)])
            dr = np.mean([x[1][:k] for x in overall_draws[s][(a, b)]], axis=0)
            lo, hi = np.percentile(dr, [2.5, 97.5])
            if s in NO_CI:
                lo = hi = np.nan
            sub = per_prot[(per_prot.stat == s) & (per_prot.comparison == f'{LABEL[a]} - {LABEL[b]}')]
            overall.append(dict(stat=s, comparison=f'{LABEL[a]} - {LABEL[b]}',
                                mean_over_proteins=pts.mean(), lo=lo, hi=hi,
                                proteins_positive=int((sub['diff'] > 0).sum()),
                                proteins_ci_above_0=int((sub.lo > 0).sum()),
                                proteins_ci_below_0=int((sub.hi < 0).sum())))
    overall = pd.DataFrame(overall)

    # which method ranks first per map and per protein
    rank = []
    for s in STATS:
        w = per_map.pivot_table(index=['protein', 'emdb_id'], columns='method', values=s)
        first_map = w.idxmax(1).value_counts().to_dict()
        first_prot = w.groupby('protein').mean().idxmax(1).value_counts().to_dict()
        rank.append(dict(stat=s, **{f'maps_first_{k}': v for k, v in first_map.items()},
                         **{f'proteins_first_{k}': v for k, v in first_prot.items()}))
    rank = pd.DataFrame(rank).fillna(0)

    with pd.ExcelWriter(f'{OUT}/method_comparison_stats.xlsx') as xw:
        overall.round(4).to_excel(xw, 'overall', index=False)
        rank.to_excel(xw, 'rank_first', index=False)
        per_prot.round(4).to_excel(xw, 'per_protein_diff', index=False)
        diffs.round(4).to_excel(xw, 'per_map_diff', index=False)
        per_map.round(4).to_excel(xw, 'per_map_stats', index=False)

    with open(f'{OUT}/log.txt', 'w') as f:
        f.write(f'B={B}, B_eqn={B_EQN}, seed={SEED}\n\n')
        f.write(rank.to_string(index=False) + '\n\n')
        f.write(overall.round(3).to_string(index=False) + '\n\n')
        f.write(per_prot.round(3).to_string(index=False) + '\n')
    print(open(f'{OUT}/log.txt').read())


if __name__ == '__main__':
    main()
