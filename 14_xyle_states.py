"""
Five states of a real two-domain transporter, for the external validation of the clustering.

Protein   XylE, the E. coli D-xylose:proton symporter (MFS fold, 12 TM helices, N and C
          domains). It is not one of the five benchmark proteins.
Real      three crystal structures (files in download/, from the RCSB PDB):
            4GBY  outward-facing, partly occluded  2.81 A  (Sun et al., Nature 2012)
            4JA3  inward-facing, partly occluded   3.80 A  (Quistgaard et al., NSMB 2013)
            4QIQ  inward-facing, open              3.51 A  (Wisedchaisri et al., Nat Commun 2014)
Designed  two states that open each side further. Pure continuation of the real rotation
          of the C domain relative to the N domain (screw axis of the 4GBY -> 4QIQ change)
          is possible for only a few degrees before the domains touch on the closing side.
          The designed states therefore rotate further along the real motion and let the
          C domain move away from the N domain, along the line joining the domain centres,
          by the smallest amount that keeps the closest N-C Calpha pair no closer than in
          the real structures (rolling contact):
            OUT_OPEN  from 4GBY, outward side opened further
            IN_WIDE   from 4QIQ, inward side opened further
          The extra angle is the one that puts the designed state EXTRA_RMSD from its real
          neighbour. These two states are a model, not data.

Atoms     backbone N, CA, C, O of the residues of the 12 TM helices that are present in
          all three structures (N domain 7-217, C domain 276-463). The intracellular
          helical domain (218-275 and the C-terminus) is left out.
Frame     all states are superposed on the N domain of 4GBY.

Reads the PDB files as plain text with its own parser (run with: python3 -I 14_xyle_states.py).
Outputs: results/xyle/
"""
import os
import sys
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'download')
OUT = os.path.join(HERE, 'results', 'xyle')
N_RANGE, C_RANGE = (7, 217), (276, 463)
BACKBONE = ['N', 'CA', 'C', 'O']
EXTRA_RMSD = 3.0                       # A, designed state to its real neighbour (the largest real neighbour step)
STATES = ['OUT_OPEN', '4GBY', '4JA3', '4QIQ', 'IN_WIDE']
TEXT = {'OUT_OPEN': 'outward, fully open (designed)', '4GBY': 'outward, partly occluded (real)',
        '4JA3': 'inward, partly occluded (real)', '4QIQ': 'inward, open (real)',
        'IN_WIDE': 'inward, wide open (designed)'}


# ── reading ──────────────────────────────────────────────────────────────────
def read_backbone(path, chain='A'):
    """{resSeq: (resName, {atom: xyz})} for one chain, first model, first alternate location."""
    out, helix = {}, []
    for ln in open(path):
        if ln.startswith('HELIX') and ln[19] == chain:
            helix.append((int(ln[21:25]), int(ln[33:37])))
        if ln.startswith('ENDMDL'):
            break
        if ln.startswith('ATOM') and ln[21] == chain and ln[16] in ' A':
            name = ln[12:16].strip()
            if name in BACKBONE:
                r = int(ln[22:26])
                out.setdefault(r, (ln[17:20], {}))[1].setdefault(
                    name, np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return out, helix


def kabsch(a, b):
    """R, t that put a onto b, and the RMSD."""
    ca, cb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    t = cb - R @ ca
    return R, t, float(np.sqrt((((a @ R.T + t) - b) ** 2).sum(1).mean()))


def rot(axis, ang):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def screw(R, t):
    """Axis n, angle (rad, > 0), point p on the axis and shift along the axis of x -> R x + t."""
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    n = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    shift = float(n @ t)
    p = np.linalg.lstsq(np.eye(3) - R, t - shift * n, rcond=None)[0]
    return n, ang, p - (p @ n) * n, shift


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = {k: read_backbone(os.path.join(SRC, f'{k}.pdb')) for k in ('4GBY', '4JA3', '4QIQ')}
    helix = raw['4GBY'][1]
    keep = [r for r in sorted(set.intersection(*[set(v[0]) for v in raw.values()]))
            if (N_RANGE[0] <= r <= N_RANGE[1] or C_RANGE[0] <= r <= C_RANGE[1])
            and len({raw[k][0][r][0] for k in raw}) == 1
            and all(set(BACKBONE) <= set(raw[k][0][r][1]) for k in raw)]
    res = np.array(keep)
    resname = [raw['4GBY'][0][r][0] for r in keep]
    dom = (res >= C_RANGE[0]).astype(int)                     # 0 = N domain, 1 = C domain
    na = len(BACKBONE)
    X = {k: np.array([[raw[k][0][r][1][a] for a in BACKBONE] for r in keep]) for k in raw}   # (res, 4, 3)
    CA = BACKBONE.index('CA')
    print(f'residues kept: {len(keep)} (N domain {int((dom == 0).sum())}, C domain {int((dom == 1).sum())}), '
          f'{len(keep) * na} backbone atoms')

    # common frame: N domain of 4GBY
    for k in ('4JA3', '4QIQ'):
        R, t, _ = kabsch(X[k][dom == 0, CA], X['4GBY'][dom == 0, CA])
        X[k] = X[k] @ R.T + t

    # real rigid-body motion of the C domain, 4GBY -> 4QIQ, in the N-domain frame
    R, t, fit = kabsch(X['4GBY'][dom == 1, CA], X['4QIQ'][dom == 1, CA])
    n, ang, p, shift = screw(R, t)
    print(f'real motion of the C domain, 4GBY -> 4QIQ: rotation {np.degrees(ang):.1f} deg, '
          f'shift along the axis {shift:.2f} A (C domain fits as a rigid body to {fit:.2f} A)')

    def moved(x, phi):
        """C domain of x carried a further angle phi (rad) along the real screw motion."""
        out = x.copy()
        sel = dom == 1
        out[sel] = (x[sel] - p) @ rot(n, phi).T + p + shift * phi / ang * n
        return out

    def ca_rmsd(a, b):
        return kabsch(a[:, CA], b[:, CA])[2]

    def min_gap(x):
        d = np.linalg.norm(x[dom == 0, CA][:, None] - x[dom == 1, CA][None], axis=2)
        return float(d.min())

    real_min = min(min_gap(X[k]) for k in X)
    u = X['4GBY'][dom == 1, CA].mean(0) - X['4GBY'][dom == 0, CA].mean(0)
    u /= np.linalg.norm(u)                                   # from the N domain to the C domain

    def rolled(x, phi):
        """Further rotation phi along the real motion; the C domain is then moved away
        from the N domain, along the line joining the domain centres, by the smallest
        amount that keeps the closest N-C Calpha pair no closer than in the real
        structures (the domains roll on each other instead of passing through)."""
        y = moved(x, phi)
        if min_gap(y) >= real_min:
            return y, 0.0
        lo, hi = 0.0, 30.0
        for _ in range(40):
            mid = (lo + hi) / 2
            z = y.copy(); z[dom == 1] += mid * u
            lo, hi = (lo, mid) if min_gap(z) >= real_min else (mid, hi)
        y[dom == 1] += hi * u
        return y, hi

    design = {}
    for name, base, sense in (('OUT_OPEN', '4GBY', -1), ('IN_WIDE', '4QIQ', 1)):
        # 1. how far pure rotation can go before the domains come closer than in any real structure
        phi = 0.0
        while min_gap(moved(X[base], sense * (phi + np.radians(0.1)))) >= real_min and phi < np.radians(90):
            phi += np.radians(0.1)
        print(f'{name}: pure rotation beyond {base} is possible for only {np.degrees(phi):.1f} deg '
              f'({ca_rmsd(X[base], moved(X[base], sense * phi)):.2f} A) before the domains touch')
        # 2. rotation with rolling contact, to EXTRA_RMSD from the real neighbour
        lo, hi = 0.0, np.radians(90)
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ca_rmsd(X[base], rolled(X[base], sense * mid)[0]) < EXTRA_RMSD else (lo, mid)
        X[name], sep = rolled(X[base], sense * hi)
        design[name] = (base, sense * hi, sep)
        print(f'{name}: {base} carried a further {np.degrees(hi):.1f} deg with the domains moved '
              f'{sep:.1f} A apart ({ca_rmsd(X[base], X[name]):.2f} A from {base})')

    # membrane normal from the TM helices of 4GBY, and the opening on each side
    ca_gby = dict(zip(keep, X['4GBY'][:, CA]))
    dirs = [ca_gby[b] - ca_gby[a] for a, b in helix if b - a >= 14 and a in ca_gby and b in ca_gby]
    dirs = np.array([d / np.linalg.norm(d) for d in dirs])
    ref = np.linalg.svd(dirs, full_matrices=False)[2][0]
    normal = (dirs * np.sign(dirs @ ref)[:, None]).mean(0)
    normal /= np.linalg.norm(normal)
    z = (X['4GBY'][:, CA] - X['4GBY'][:, CA].mean(0)) @ normal
    top, bot = z > np.percentile(z, 67), z < np.percentile(z, 33)

    def opening(x, slab):
        return float(np.linalg.norm(x[slab & (dom == 0), CA].mean(0) - x[slab & (dom == 1), CA].mean(0)))

    # which slab is the outward (periplasmic) side: the one wider in 4GBY than in 4QIQ
    if opening(X['4GBY'], top) - opening(X['4QIQ'], top) < opening(X['4GBY'], bot) - opening(X['4QIQ'], bot):
        top, bot = bot, top
    print('\nstate      | distance between the N and C domain ends (A) | closest N-C Calpha pair (A)')
    print('           |   outward side      inward side              |')
    for s in STATES:
        print(f'{s:10s} | {opening(X[s], top):10.1f} {opening(X[s], bot):16.1f}              | {min_gap(X[s]):.2f}   {TEXT[s]}')

    D = np.array([[ca_rmsd(X[a], X[b]) for b in STATES] for a in STATES])
    print('\nCalpha-RMSD between states (A), 12 TM helices:')
    print('           ' + ''.join(f'{s:>10s}' for s in STATES))
    for s, row in zip(STATES, D):
        print(f'{s:10s} ' + ''.join(f'{v:10.2f}' for v in row))

    # write the five states (backbone, real residue names and numbers, helix records of 4GBY)
    kept = set(keep)
    hel = []
    for a, b in helix:
        rs = [r for r in range(a, b + 1) if r in kept]
        if len(rs) >= 4:
            hel.append((rs[0], rs[-1]))
    name_of = dict(zip(keep, resname))
    elem = {'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O'}
    for s in STATES:
        with open(os.path.join(OUT, f'XylE_{s}.pdb'), 'w') as f:
            notes = ['XylE (E. coli D-xylose:proton symporter), state: ' + TEXT[s] + '.',
                     'Backbone of the 12 TM helices (residues 7-217 and 276-463) present in '
                     '4GBY, 4JA3 and 4QIQ.']
            if s in design:
                notes.append(f'Designed from {design[s][0]}: C domain carried a further '
                             f'{abs(np.degrees(design[s][1])):.1f} deg along the real 4GBY-4QIQ '
                             f'rigid-body motion and moved {design[s][2]:.1f} A away from the N domain.')
            else:
                notes.append(f'Real coordinates of PDB entry {s}, superposed on the N domain of 4GBY.')
            f.write(('TITLE     XYLE, ' + TEXT[s].upper())[:80] + '\n')
            for note in notes:                      # free-text remarks in valid PDB format
                words, line = note.split(), ''
                for w in words:
                    if len(line) + len(w) + 1 > 68:
                        f.write('REMARK  99 ' + line + '\n'); line = w
                    else:
                        line = (line + ' ' + w).strip()
                f.write('REMARK  99 ' + line + '\n')
            for i, (a, b) in enumerate(hel, 1):
                f.write('HELIX  %3d %3s %3s A %4d  %3s A %4d  1%36d\n'
                        % (i, f'H{i}', name_of[a], a, name_of[b], b, b - a + 1))
            k = 0
            for i, r in enumerate(keep):
                for j, a in enumerate(BACKBONE):
                    k += 1
                    f.write('ATOM  %5d  %-3s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n'
                            % (k, a, resname[i], r, *X[s][i, j], elem[a]))
            f.write('END\n')
    np.savez(os.path.join(OUT, 'XylE_states.npz'), states=np.array(STATES), res=res, dom=dom,
             resname=np.array(resname), xyz=np.array([X[s] for s in STATES]), rmsd=D,
             axis=n, axis_point=p, angle_deg=np.degrees(ang), shift=shift, normal=normal)


if __name__ == '__main__':
    main()
