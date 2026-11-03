# Study `tdstdp-baseline`: the TD-STDP baseline for the paper

**Pre-registered 2026-10-23, 22:37 EDT, before any run of this study (launched 22:38).** Design and gates below are fixed. Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/` and `summary.*` come only from `python -m experiments.report`.

## What the study is

- It is the reported baseline: Chung & Kozma 2020 [CK] (arXiv:2008.13044), feed-forward TD-STDP actor-critic, on the authors'-code path of iteration 013 (`docs/iterations/013-tdstdp-backbone.md`, §3.1).
- The paper's comparison will be action columns plus the map term, without Adam, against these groups.
- The development results of iteration 013 (E1′, E3′, E4′, S1–S3, in `experiments/results/`) are not part of this study. They used other seeds, block sizes and, in part, no `--fast`.

## Design

- **Code path:** the CK-faithful default of `python -m experiments.tdstdp`: the `PRESETS`, `CKRunner` and `block_learn_ck_fast`.
- **Speed settings** on every run: block sum **B = 20** (`--block 20`) and `--fast`. Both passed their gates in iteration 013 §11 (S1, S2).
- **Seeds:** 1000–1019 (CartPole) and 1000–1009 (LunarLander). These are fresh; no earlier run used seeds at or above 1000.
- **Engines:**
  - `lif`: BindsNET's deterministic LIF, W in mV (`--engine lif`).
  - `branching`: the SuperNeuroMAT branching step in numpy, with escape noise (the default engine).

| Group | Task | Optimiser | Engine | Seeds | Episodes |
| --- | --- | --- | --- | --- | --- |
| `CartPole_noadam_lif` (**reference**, paper config) | CartPole-v1 | none (paper config) | lif | 1000–1019 | 400 |
| `CartPole_noadam_branching` | CartPole-v1 | none | branching | 1000–1019 | 400 |
| `CartPole_adam_lif` | CartPole-v1 | Adam (`--adam`) | lif | 1000–1019 | 400 |
| `CartPole_adam_branching` | CartPole-v1 | Adam | branching | 1000–1019 | 400 |
| `LunarLander_adam_lif` (**reference**, paper config) | LunarLander-v3 | Adam (paper config) | lif | 1000–1009 | pooled, min 3000, max 10000 |
| `LunarLander_adam_branching` | LunarLander-v3 | Adam | branching | 1000–1009 | as above |
| `LunarLander_sgd_lif` | LunarLander-v3 | none, η from the E4′ rule (`--sgd`) | lif | 1000–1009 | as above |
| `LunarLander_sgd_branching` | LunarLander-v3 | none, η from the E4′ rule | branching | 1000–1009 | as above |

- **CartPole details:**
  - The authors' `config_cp.ini`, with one env copy.
  - A run is 400 episodes.
  - Adam, where used, is the block Adam with per-ms-matched moments (record §3.1 item 7), with β = (0.995, 0.99995) and ε as in the preset.
- **LunarLander details:**
  - The authors' `config_ll.ini`: 16 lockstep copies, with episodes pooled over them.
  - A run stops when it is solved with at least 3000 pooled episodes, or at 10000.
  - **E4′ η rule** (record §9), applied per engine on that engine: per population, $\eta = \nu_{Adam}/\mathrm{RMS}(g)$, measured at B = 1 with the initial W frozen.
- **Criteria** (CK's):
  - CartPole is solved when 100 consecutive episodes score 500.
  - LunarLander is solved when the trailing-100 pooled mean reaches at least 200.
  - $T_f$ is the first episode at the perfect score, and $T_s$ is the last episode of the first solved window.
  - $T_m$ (secondary) is the first episode whose trailing-100 mean reaches 475 (CartPole) or 200 (LunarLander).
- **Statistics:**
  - Per group: mean ± sd (ddof 0) over the runs that reach each time, the median $T_s$ with unsolved runs counted as infinite, and the last-100 return as mean ± sd over seeds.
  - Every other group of a task is compared with that task's reference group by a two-sided Mann–Whitney test on $T_f$, $T_s$ (never reached = episode cap + 1) and the last-100 return.
  - The statistics function is `experiments.tdstdp.group_stats`, the same one `--summary` uses.

## Gates

| Gate | Group | Pass if |
| --- | --- | --- |
| G1 | `CartPole_noadam_lif` | at least 18/20 solved |
| G2 | `LunarLander_adam_lif` | at least 6/10 solved |

All other groups are reported, not gated. `summary.md` prints pass or fail per gate. A gate on an incomplete group reads "incomplete".

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP_NUM_THREADS=1` and 40 workers. No other jobs of ours are running.
- **Order:** the CartPole groups are queued first, then LunarLander.
- **Checkpoints:** each run checkpoints every 50 pooled episodes (`runs/<group>/s<seed>.ckpt.npz`, gitignored, deleted on completion). A relaunch resumes it. A resumed run is not bit-identical, and the count is recorded in its `resumes` field and in `summary.csv`.
- **One code state:**
  - `manifest.json` records the git HEAD and the sha256 of the uncommitted code in `mapstdp/` and `experiments/`, and `code.patch` holds that code.
  - The runner refuses to relaunch if the code differs from the manifest.
  - If code a run executes has to change after launch, the whole study is deleted and relaunched. Code versions are never mixed.

## Commands

```bash
python -m experiments.report run tdstdp-baseline --workers 40   # launch, or resume, every declared run; then summarize
python -m experiments.report summarize tdstdp-baseline          # rebuild summary.csv and summary.md from runs/
```

The exact per-run commands are in `manifest.json` (`groups.<group>.commands`). For example:

```bash
python -m experiments.tdstdp --task CartPole --seed 1000 --engine lif --block 20 --fast --out reports/tdstdp-baseline/runs/CartPole_noadam_lif/s1000.json
python -m experiments.tdstdp --task LunarLander --seed 1000 --sgd --block 20 --fast --out reports/tdstdp-baseline/runs/LunarLander_sgd_branching/s1000.json
```

## How to regenerate

1. Check out `git_head` from `manifest.json`.
2. Apply `code.patch` (`git apply code.patch`).
3. Create the `map-stdp` env with the versions in `manifest.json`.
4. Delete `runs/`, `manifest.json` and `code.patch`.
5. Run the first command above.

The runs are seeded, so an uninterrupted rerun reproduces the same returns (up to floating-point summation order across BLAS builds).

Per-run logs go to `logs/report/tdstdp-baseline/`, which is gitignored.

## Amendment (2026-10-24, after the runs): CartPole Adam groups dropped

- `CartPole_adam_lif` and `CartPole_adam_branching` were declared and run, then dropped by the user's decision. Both solved 0/20 (last-100 returns 15.3 and 11.4).
- **Why.** The authors' Adam step is the per-ms SGD η. For CartPole that is 40–160× LunarLander's step, and it collapses the policy from the first episodes. The same collapse occurs at B = 1 without `--fast`, so it is the configuration, not the speed settings. CK report no CartPole Adam configuration, and CartPole needs no Adam (`CartPole_noadam_lif`, 20/20).
- **What changed.** The groups are marked `dropped` in `experiments/report.py`. Their run files stay in `runs/`, and `summarize` leaves them out of `summary.csv` and the tables, listing them under "Dropped groups". No other group, gate or statistic changed.
- `report.py` was edited after the study ran, so `manifest.json`'s code hash no longer matches the working tree and `run` would refuse a relaunch. The run code (`mapstdp/`, `experiments/tdstdp.py`) is unchanged.

## Amendment (2026-10-24, after the runs): `sgd` groups shown as `noadam`

- **What:** the LunarLander groups `LunarLander_sgd_<engine>` are shown in `summary.md` and `summary.csv` as `LunarLander_noadam_<engine>`, by the study's `labels` in `experiments/report.py`. The run directories (`runs/LunarLander_sgd_<engine>/`), the manifest and its recorded commands (`--sgd`) keep the old name.
- **Why:** "SGD" reads as backprop's stochastic gradient descent, a non-local algorithm. The arm is the same local three-factor rule as the Adam arm, with Adam removed and a fixed η per population (critic, actor) calibrated once before training (the E4′ rule). The CLI flag is now `--eta-cal`; `--sgd` stays as a hidden alias so the recorded commands still run.
- **Effect on results:** none. Only names change; `summary.*` was rebuilt from the same `runs/`.
