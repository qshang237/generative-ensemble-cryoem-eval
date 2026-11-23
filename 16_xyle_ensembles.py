"""
Test ensembles with known state labels, for the external validation of the clustering.
They are built on the five XylE states of 14_xyle_states.py (three real crystal structures,
two designed open states).

Noise     anisotropic network model (Atilgan et al. 2001; Calpha, cutoff 15 A) on each
          state. A frame is the state plus a Gaussian displacement along the N_MODES
          lowest-frequency modes, variance proportional to 1/eigenvalue. Low modes are
          collective, so helices stay intact, as in the real generated ensembles; using
          all modes gives jagged local geometry and a narrower RMSD distribution than the
          real ensembles (measured: results/xyle/log.txt). The Gaussian is
          unimodal by construction. Every backbone atom of a residue moves with its
          Calpha. The amplitude is set per state so that the median pairwise Calpha-RMSD
          within the state equals NOISE.

Sets      500 frames each, frames in random order, known state of every frame in
          <name>_labels.csv:
            XylE_uniform   100 / 100 / 100 / 100 / 100
            XylE_main      inward-open 225, outward partly occluded 100, inward partly
                           occluded 100, inward wide open 50, outward fully open 25
            XylE_skewed    350 / 60 / 40 / 30 / 20 in the same order
            XylE_one_state 500 frames of the inward-open state (no classes)

Formats   as the real runs: bioemu_format/<name>/samples.xtc + topology.pdb,
          alphaflow_format/<name>.pdb (one MODEL per frame),
          msa_format/<name>/<name>_unrelaxed_rank_NNN.pdb (one file per frame).
          Backbone atoms N, CA, C, O with the real residue names and numbers.

This script runs no clustering method.
Outputs: synthetic_ensembles/ (ensembles) and results/xyle/ (figure, log)
"""
import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
XY = os.path.join(HERE, 'results', 'xyle')
ENS = os.path.join(HERE, 'synthetic_ensembles')
SEED = 20261114
ANM_CUTOFF, N_MODES, NOISE = 15.0, 30, 1.0
BACKBONE, ELEMENT, CA = ['N', 'CA', 'C', 'O'], ['N', 'C', 'C', 'O'], 1
STATES = ['OUT_OPEN', '4GBY', '4JA3', '4QIQ', 'IN_WIDE']
TEXT = {'OUT_OPEN': 'outward, fully open (designed)', '4GBY': 'outward, partly occluded (4GBY)',
        '4JA3': 'inward, partly occluded (4JA3)', '4QIQ': 'inward, open (4QIQ)',
        'IN_WIDE': 'inward, wide open (designed)'}
SETS = {'XylE_uniform': [100, 100, 100, 100, 100],     # frames per state, in the order of STATES
        'XylE_main': [25, 100, 100, 225, 50],
        'XylE_skewed': [20, 60, 40, 350, 30],
        'XylE_one_state': [0, 0, 0, 500, 0]}


# ── noise model ──────────────────────────────────────────────────────────────
def anm_modes(ca):
    n = len(ca)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = ca[i + 1:] - ca[i]
        r2 = (dv ** 2).sum(1)
        for k in np.where(r2 <= ANM_CUTOFF ** 2)[0]:
            j = i + 1 + k
            blk = -np.outer(dv[k], dv[k]) / r2[k]
            H[3*i:3*i+3, 3*j:3*j+3] = blk; H[3*j:3*j+3, 3*i:3*i+3] = blk
            H[3*i:3*i+3, 3*i:3*i+3] -= blk; H[3*j:3*j+3, 3*j:3*j+3] -= blk
    w, v = np.linalg.eigh(H)
    assert w[5] < 1e-8 < w[6], 'expected exactly six zero modes'
    w, v = w[6:6 + N_MODES], v[:, 6:6 + N_MODES]
    # The sign of an eigenvector is arbitrary and can change with the linear-algebra library or
    # the number of threads. Fix it (largest component positive), so that one seed gives the
    # same ensemble on every machine.
    v = v * np.sign(v[np.abs(v).argmax(0), np.arange(v.shape[1])])
    return w, v


def ca_traj(ca):
    top = md.Topology()
    ch = top.add_chain()
    for _ in range(ca.shape[1]):
        top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    return md.Trajectory(ca / 10.0, top)


def pairwise_rmsd_fit(ca):
    t = ca_traj(ca)
    t.center_coordinates()
    R = np.array([md.rmsd(t, t, i, precentered=True) * 10 for i in range(t.n_frames)])
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0)
    return R


class Model:
    """States, ANM modes and noise scales; sample(state, n) returns backbone frames."""

    def __init__(self, noise=NOISE, seed=SEED):
        z = np.load(os.path.join(XY, 'XylE_states.npz'))
        assert [str(s) for s in z['states']] == STATES
        self.xyz, self.res, self.dom = z['xyz'], z['res'], z['dom']
        self.resname = [str(s) for s in z['resname']]
        self.rng = np.random.default_rng(seed)
        self.modes = [anm_modes(x[:, CA]) for x in self.xyz]
        self.scale = [1.0] * len(STATES)
        for k in range(len(STATES)):                       # same noise level in every state
            R = pairwise_rmsd_fit(self.sample(k, 300)[:, :, CA])
            self.scale[k] = noise / np.median(R[np.triu_indices(300, 1)])

    def sample(self, state, n, factor=1.0):
        w, v = self.modes[state]
        d = (self.rng.standard_normal((n, len(w))) / np.sqrt(w)) @ v.T
        d = factor * self.scale[state] * d.reshape(n, -1, 3)
        return self.xyz[state][None] + d[:, :, None, :]      # all backbone atoms follow the Calpha

    def ensemble(self, population):
        """Frames in random order and their state index."""
        lab = np.repeat(np.arange(len(STATES)), population)
        x = np.vstack([self.sample(k, m) for k, m in enumerate(population) if m > 0])
        order = self.rng.permutation(len(lab))
        return x[order], lab[order]


# ── writing ──────────────────────────────────────────────────────────────────
def helix_records():
    return [ln for ln in open(os.path.join(XY, 'XylE_4GBY.pdb')) if ln.startswith('HELIX')]


def atom_lines(m, frame):
    out, k = [], 0
    for i, r in enumerate(m.res):
        for j, a in enumerate(BACKBONE):
            k += 1
            out.append('ATOM  %5d  %-3s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n'
                       % (k, a, m.resname[i], r, *frame[i, j], ELEMENT[j]))
    return out


def write_set(m, name, x, lab, single_files=True):
    head = ['TITLE     XYLE BENCHMARK ENSEMBLE, SIMULATED FRAME\n',
            'REMARK  99 Backbone of the 12 TM helices of XylE (residues 7-217, 276-463).\n',
            'REMARK  99 State plus elastic-network noise; see 16_xyle_ensembles.py.\n'] + helix_records()
    d = os.path.join(ENS, 'bioemu_format', name)
    os.makedirs(d, exist_ok=True)
    top = md.Topology()
    ch = top.add_chain()
    for i, r in enumerate(m.res):
        rr = top.add_residue(m.resname[i], ch, resSeq=int(r))
        for a, e in zip(BACKBONE, ELEMENT):
            top.add_atom(a, md.element.get_by_symbol(e), rr)
    md.Trajectory(x.reshape(len(x), -1, 3) / 10.0, top).save_xtc(os.path.join(d, 'samples.xtc'))
    with open(os.path.join(d, 'topology.pdb'), 'w') as f:
        f.writelines(head + atom_lines(m, x[0]) + ['END\n'])
    os.makedirs(os.path.join(ENS, 'alphaflow_format'), exist_ok=True)
    with open(os.path.join(ENS, 'alphaflow_format', f'{name}.pdb'), 'w') as f:
        f.writelines(head)
        for i, fr in enumerate(x):
            f.write('MODEL     %4d\n' % (i + 1))
            f.writelines(atom_lines(m, fr))
            f.write('ENDMDL\n')
        f.write('END\n')
    if single_files:
        d = os.path.join(ENS, 'msa_format', name)
        os.makedirs(d, exist_ok=True)
        for i, fr in enumerate(x):
            with open(os.path.join(d, f'{name}_unrelaxed_rank_{i + 1:03d}.pdb'), 'w') as f:
                f.writelines(head + atom_lines(m, fr) + ['END\n'])
    with open(os.path.join(ENS, f'{name}_labels.csv'), 'w') as f:
        f.write('frame,state,description\n')
        f.writelines(f'{i + 1},{STATES[l]},"{TEXT[STATES[l]]}"\n' for i, l in enumerate(lab))


def mds(R, k=2):
    n = len(R); J = np.eye(n) - 1 / n
    w, v = np.linalg.eigh(-0.5 * J @ (R ** 2) @ J)
    o = np.argsort(w)[::-1]
    return v[:, o[:k]] * np.sqrt(np.clip(w[o[:k]], 0, None)), w[o] / w[w > 0].sum()


def main():
    log = open(os.path.join(XY, 'log.txt'), 'w')

    def say(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True); log.write(s + '\n')

    m = Model()
    adj = np.diff(m.res) == 1
    say(f'noise: lowest {N_MODES} ANM modes, median pairwise Calpha-RMSD within a state {NOISE} A')
    say('state      softest mode eigenvalue   5th / 50th / 95th pct of within-state RMSD (A)   '
        's.d. of CA(i)-CA(i+1)   s.d. of C(i)-N(i+1)')
    for k, s in enumerate(STATES):
        fr = m.sample(k, 300)
        v = pairwise_rmsd_fit(fr[:, :, CA])[np.triu_indices(300, 1)]
        cc = np.linalg.norm(fr[:, 1:, CA] - fr[:, :-1, CA], axis=2)[:, adj].std(0).mean()
        cn = np.linalg.norm(fr[:, 1:, 0] - fr[:, :-1, 2], axis=2)[:, adj].std(0).mean()
        p = np.percentile(v, [5, 50, 95])
        say(f'{s:10s} {m.modes[k][0][0]:10.3f}                 {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}'
            f'  ({p[0]/p[1]:.2f} and {p[2]/p[1]:.2f} x median)        {cc:.2f} A                 {cn:.2f} A')

    fig, axs = plt.subplots(2, len(SETS), figsize=(5.2 * len(SETS), 9.6))
    cols = plt.cm.tab10(np.arange(len(STATES)))
    for c, (name, pop) in enumerate(SETS.items()):
        x, lab = m.ensemble(pop)
        write_set(m, name, x, lab, single_files=(name != 'XylE_one_state'))
        R = pairwise_rmsd_fit(x[:, :, CA])
        np.savez_compressed(os.path.join(XY, f'{name}_rmsd.npz'), R=R, labels=lab)
        say(f'\n{name}: frames per state ' + ', '.join(f'{STATES[k]} {n}' for k, n in enumerate(pop) if n))
        present = [k for k, n in enumerate(pop) if n]
        if len(present) > 1:
            say('  median pairwise Calpha-RMSD between frames of two states (A); diagonal = within a state')
            say('            ' + ''.join(f'{STATES[k]:>10s}' for k in present))
            for a in present:
                row = [np.median(R[np.ix_(lab == a, lab == b)][np.triu_indices((lab == a).sum(), 1)]) if a == b
                       else np.median(R[np.ix_(lab == a, lab == b)]) for b in present]
                say(f'  {STATES[a]:10s}' + ''.join(f'{v:10.2f}' for v in row))
        o = np.argsort(lab, kind='stable')
        ax = axs[0, c]
        im = ax.imshow(R[np.ix_(o, o)], cmap='viridis', vmin=0, vmax=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{name}\nRMSD matrix, frames sorted by KNOWN state', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label='A')
        ax = axs[1, c]
        P, var = mds(R)
        for k in present:
            ax.scatter(P[lab == k, 0], P[lab == k, 1], s=6, alpha=0.5, color=cols[k],
                       label=f'{TEXT[STATES[k]]} (n={int((lab == k).sum())})')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel(f'PC1 ({var[0]*100:.0f}%), A'); ax.set_ylabel(f'PC2 ({var[1]*100:.0f}%), A')
        ax.set_title('PCA of the RMSD matrix, coloured by KNOWN state', fontsize=10)
        ax.legend(fontsize=6.5, frameon=False, loc='best')
    fig.suptitle(f'XylE test ensembles: 500 frames each, noise {NOISE} A (lowest {N_MODES} ANM modes). '
                 'Colours are the known states, not a clustering result.', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(XY, 'XylE_ensembles_overview.png'), dpi=100)
    log.close()


if __name__ == '__main__':
    main()
