# fermat-gpc

Hodge and Grothendieck Period Conjectures on Fermat fourfolds.
Reproduces the baseline enumeration, the RL training curves, and the
deterministic-vs-RL comparison across the Jumagulov range d <= 199.

## Setup

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Reproduce central results

    bash scripts/reproduce.sh

This writes:

* `data/baseline.csv` — exact tuple counts and decomposability rates per degree.
* `data/rl_metrics.csv` — final reward, certificate rate, loss, and ceiling
  fraction for the trained DeepSets+CEM agent at each degree.
* `data/comparison.csv` — merge of the two tables with wall-clock time ratio.

## What the comparison shows

For d <= 13 the deterministic baseline finishes in < 1 s. The RL agent needs
~15 epochs of 512 rollouts (≈ 7,700 reward evaluations) to reach ~77% of the
theoretical ceiling. The ratio (RL time) / (baseline time) grows roughly
quadratically in d, because enumeration is O(d^5) and RL requires O(d^5)
states to cover the space even though each state is sampled independently.

## Central modules

* `fermat_gpc.enumeration` — streaming generator of middle-Hodge tuples.
* `fermat_gpc.certificates` — O(k) full-pair and weak-pair tests.
* `fermat_gpc.rl` — bandit environment, equivariant policy, CEM training.
* `fermat_gpc.drb` — de Rham–Betti locus via PSLQ.
## Notebooks
`GpcNb.ipyBb(1).ipynb` shows the RL construction for d=13, `D13Case.ipynb` contains a conditional proof of GPC on this case.
