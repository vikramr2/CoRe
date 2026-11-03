# Study `tdstdp-baseline-flow`: the flow-engine arms of the TD-STDP baseline

**Pre-registered 2026-10-24, 15:20 EDT, before any run of this study.** Design and reading rules below are fixed. Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/` and `summary.*` come only from `python -m experiments.report`.

## What the study is

- It is a companion to `reports/tdstdp-baseline/`. It adds the third engine, **flow**, to the reported TD-STDP baseline (Chung & Kozma 2020 [CK], authors'-code path, iteration 013).
- Flow (`--engine flow`, `tdstdp.flow_step`) is the branching step's noise-free limit. Rates replace spikes, and $z$ gets the input autocorrelation term in closed form. Map-STDP's derivation is exact on this engine (`docs/neuron-variants.md` §3), so it is the engine on which the columns-plus-map-term comparison is cleanest.
- It is a separate study because a study uses one code state. `tdstdp-baseline` ran at git `675e49e`. This one runs at the current HEAD, plus the runner change that adds the flow engine and cross-study references (recorded in `code.patch`). The code path the flow runs execute (`mapstdp/tdstdp.py`, `experiments/tdstdp.py`) is unchanged since `675e49e`.
- The Mann–Whitney tests compare against the baseline's reference groups (`tdstdp-baseline/CartPole_noadam_lif`, `tdstdp-baseline/LunarLander_adam_lif`), read from that study's `runs/`. They are not rerun.

## Development evidence (not part of this study)

On the dev seeds 60–79 (`experiments/results/tdstdp_CartPole_ck_flow_s*.json`, block 1; scratchpad smoke runs at B = 20):

- At block 1, all 20 seeds reach $T_s$.
- At B = 20, 16/20 reach $T_s$. The other 4 (seeds 61, 64, 72, 79) collapse by episode 100–150 to a near-deterministic policy (entropy about 0.06) with a return of about 10.
- The block-sum gate (iteration 013 §11, S1) was passed on branching only. Flow has no escape noise, so it may be more sensitive to freezing W over a 20 ms block. The `CartPole_noadam_flow_b1` group tests that on fresh seeds.
- LunarLander flow at B = 20 learns with both Adam and the SGD calibration (smoke, seed 60: trailing-50 return 26 after 500 pooled episodes, 4.6 min).

## Design

- **Code path and settings:** as in `tdstdp-baseline`. The CK-faithful default of `python -m experiments.tdstdp`, with `--fast`, the same `PRESETS`, criteria, episode counting and statistics. `--engine flow` on every run.
- **Seeds:** the same as the baseline, 1000–1019 (CartPole) and 1000–1009 (LunarLander), so engines are compared seed for seed. Flow has never been run on these seeds.

| Group | Task | Optimiser | Block | Seeds | Episodes |
| --- | --- | --- | --- | --- | --- |
| `CartPole_noadam_flow` | CartPole-v1 | none (paper config) | B = 20 | 1000–1019 | 400 |
| `CartPole_noadam_flow_b1` (block-sum control) | CartPole-v1 | none | B = 1 (per-ms rule) | 1000–1019 | 400 |
| `LunarLander_adam_flow` | LunarLander-v3 | Adam (paper config) | B = 20 | 1000–1009 | pooled, min 3000, max 10000 |
| `LunarLander_sgd_flow` | LunarLander-v3 | none, η from the E4′ rule (`--sgd`), measured on flow | B = 20 | 1000–1009 | as above |

- LunarLander has no B = 1 group: at block 1 a run takes many hours (about 7× the B = 20 time).

## Reading rules (pre-registered)

There are no pass/fail gates. Every group is reported.

1. **Engine comparison.** Each group is compared with the baseline's reference group of its task (Mann–Whitney on $T_f$, $T_s$ and last-100 return, in `summary.md`).
2. **Block sum on flow.** `CartPole_noadam_flow` (B = 20) is read against `CartPole_noadam_flow_b1` on the solved count (two-sided Fisher exact test, computed from `summary.csv`).
   - If B = 20 solves significantly fewer (p < 0.05), the B = 20 flow numbers are reported as a lower bound of the flow engine, the CartPole flow entry of the baseline is `CartPole_noadam_flow_b1`, and the LunarLander flow groups carry that caveat.
   - Otherwise `CartPole_noadam_flow` is the CartPole flow entry, matching the block size of every other baseline group.

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP_NUM_THREADS=1` and 40 workers.
- **Order:** CartPole groups first, then LunarLander.
- **Checkpoints:** as in `tdstdp-baseline` (every 50 pooled episodes; a resume is counted in `resumes`).
- **One code state:** the runner refuses to relaunch if the code differs from `manifest.json`.

## Commands

```bash
python -m experiments.report run tdstdp-baseline-flow --workers 40   # launch, or resume; then summarize
python -m experiments.report summarize tdstdp-baseline-flow          # rebuild summary.csv and summary.md from runs/
```

The exact per-run commands are in `manifest.json` (`groups.<group>.commands`), for example:

```bash
python -m experiments.tdstdp --task CartPole --seed 1000 --engine flow --block 20 --fast --out reports/tdstdp-baseline-flow/runs/CartPole_noadam_flow/s1000.json
```

## Amendment (2026-10-24, after the runs): `sgd` groups shown as `noadam`

- **What:** the LunarLander groups `LunarLander_sgd_<engine>` are shown in `summary.md` and `summary.csv` as `LunarLander_noadam_<engine>`, by the study's `labels` in `experiments/report.py`. The run directories (`runs/LunarLander_sgd_<engine>/`), the manifest and its recorded commands (`--sgd`) keep the old name.
- **Why:** "SGD" reads as backprop's stochastic gradient descent, a non-local algorithm. The arm is the same local three-factor rule as the Adam arm, with Adam removed and a fixed η per population (critic, actor) calibrated once before training (the E4′ rule). The CLI flag is now `--eta-cal`; `--sgd` stays as a hidden alias so the recorded commands still run.
- **Effect on results:** none. Only names change; `summary.*` was rebuilt from the same `runs/`.
