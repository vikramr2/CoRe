# Final models of the reported studies

The best run of each arm of the race-readout studies on the TD-STDP backbone (iteration 016 §20–21; the recipe in
`docs/columns-recipe.md`) and of the Map-STDP study `emergent-v4` (model v4, the emergent rule on CartPole), one `.npz`
per arm, plus the scripts that reproduce, verify and run the race models.

## The model code

The two main models live here as importable modules, next to their saved weights. Run them from the repo root:

| Module | Model | Studies | Run |
| --- | --- | --- | --- |
| `basic.py` | Map-STDP: models v3/v4 and the emergent rule (task registry, `Config`, `Agent`, `run`, the per-task CLI) | `emergent-v4` and the M2/M3 runs | `python -m models.basic --task CartPole ...` |
| `tdstdp.py` | The TD-STDP backbone of Chung & Kozma 2020 with the race readout and recurrent action columns | the race studies, `cross-sweep`, `tdstdp-baseline`, `hybrid-columns*` | `python -m models.tdstdp --task CartPole ...` |

Both write to `experiments/results/` by default. Until 2026-11-02 they were `experiments/tasks.py`, `experiments/m2.py`,
`experiments/m3.py` and `experiments/tdstdp.py`; the commands in older manifests and iteration records use those paths
(`experiments.m2` and `experiments.m3` are now `models.basic`, `experiments.tdstdp` is `models.tdstdp`).

**Every model is in `archive/models/`** (gitignored, 316 MB, to be kept on a server): all 320 race runs, one `.npz`
per run in the same `<study>/<arm>/s<seed>.npz` layout, and a copy of the 240 emergent-v4 models. `reproduce.sh` writes
there; this directory keeps the copies of the best runs only.

| Study | Task | Arms (one directory each) | Seeds | Reported result |
| --- | --- | --- | --- | --- |
| `race-cartpole` | CartPole | `CartPole_base_ck`, `CartPole_base_nl`, `CartPole_base_race_rp`, `CartPole_cols_lam0_race_rp`, `CartPole_cols_on_race_rp`, `CartPole_cols_sealed_race_rp` | 7000–7019 | `reports/race-cartpole/summary.md` |
| `race-acrobot` | Acrobot | `Acrobot_base_ck`, `Acrobot_base_race_h2`, `Acrobot_cols_lam0_race_h2`, `Acrobot_cols_on_race_h2`, `Acrobot_cols_sealed_race_h2` | 7100–7119 | `reports/race-acrobot/summary.md` |
| `race-lunarlander` | LunarLander | `LunarLander_base_ck`, `LunarLander_base_race_e`, `LunarLander_cols_lam0_race_e`, `LunarLander_cols_on_race_e`, `LunarLander_cols_sealed_race_e` | 7200–7219 | `reports/race-lunarlander/summary.md` |

Each study's `README.md` in `reports/<study>/` is its pre-registration (arms, flags, settings and how they were chosen); `summary.md` and `pairs.csv` there are the reported tables.

## Best run per arm of the race studies (the files kept here)

Best = the highest last-100 return of the reported run (the policy the final weights encode), ties broken by the mean
return over the whole run (all CartPole arms tie at 500), then the lower seed. Returns are from `reports/<study>/runs/`.

| Study | Arm | Model | Last-100 return | Mean return |
| --- | --- | --- | --- | --- |
| `race-cartpole` | `CartPole_base_ck` | `s7012.npz` | 500.0 | 388.2 |
| `race-cartpole` | `CartPole_base_nl` | `s7012.npz` | 500.0 | 381.2 |
| `race-cartpole` | `CartPole_base_race_rp` | `s7007.npz` | 500.0 | 425.7 |
| `race-cartpole` | `CartPole_cols_lam0_race_rp` | `s7008.npz` | 500.0 | 409.8 |
| `race-cartpole` | `CartPole_cols_on_race_rp` | `s7010.npz` | 500.0 | 414.1 |
| `race-cartpole` | `CartPole_cols_sealed_race_rp` | `s7005.npz` | 500.0 | 421.1 |
| `race-acrobot` | `Acrobot_base_ck` | `s7108.npz` | -107.2 | -156.6 |
| `race-acrobot` | `Acrobot_base_race_h2` | `s7108.npz` | -113.7 | -173.9 |
| `race-acrobot` | `Acrobot_cols_lam0_race_h2` | `s7118.npz` | -113.2 | -182.4 |
| `race-acrobot` | `Acrobot_cols_on_race_h2` | `s7119.npz` | -112.2 | -182.1 |
| `race-acrobot` | `Acrobot_cols_sealed_race_h2` | `s7116.npz` | -110.0 | -163.1 |
| `race-lunarlander` | `LunarLander_base_ck` | `s7211.npz` | 87.5 | 4.4 |
| `race-lunarlander` | `LunarLander_base_race_e` | `s7215.npz` | 111.2 | 22.5 |
| `race-lunarlander` | `LunarLander_cols_lam0_race_e` | `s7219.npz` | 152.1 | 48.5 |
| `race-lunarlander` | `LunarLander_cols_on_race_e` | `s7208.npz` | 120.3 | 25.8 |
| `race-lunarlander` | `LunarLander_cols_sealed_race_e` | `s7203.npz` | 131.6 | 27.9 |

## Best run per arm of `emergent-v4` (`emergent-v4/`)

The 240 models of `emergent-v4` (four engines × map term on, off and sealed × seeds 2000–2019) are committed in
`reports/emergent-v4/models/`, which `latent-ablation` reads; `emergent-v4/` here holds byte copies of the best one per
arm. Best = the highest **frozen** return (the mean of the 20 end-of-run episodes the training run played with the final
weights frozen, `frozen_returns` in `reports/emergent-v4/runs/`), because a v4 model is the final snapshot and its
last-100 return was played while the weights still moved (one QIF run with last-100 293 scores 10 frozen). Ties (most
flow runs at 500) are broken by the last-100 return, then the mean return over the run, then the lower seed. The
picks hold on the 50 independent frozen episodes of `latent-ablation`'s intact condition, where each non-flow pick is
first in its arm, except QIF sealed, where it is second.

| Arm | Model | Frozen return | Last-100 return | Mean return |
| --- | --- | --- | --- | --- |
| `CartPole_branching_off` | `s2001.npz` | 469.3 | 155.3 | 102.0 |
| `CartPole_branching_on` | `s2005.npz` | 471.2 | 229.3 | 210.4 |
| `CartPole_branching_sealed` | `s2011.npz` | 461.9 | 347.4 | 180.6 |
| `CartPole_flow_off` | `s2007.npz` | 500.0 | 500.0 | 418.1 |
| `CartPole_flow_on` | `s2004.npz` | 500.0 | 500.0 | 455.6 |
| `CartPole_flow_sealed` | `s2005.npz` | 500.0 | 500.0 | 464.5 |
| `CartPole_lif_off` | `s2013.npz` | 335.0 | 318.4 | 196.0 |
| `CartPole_lif_on` | `s2017.npz` | 466.4 | 428.1 | 281.8 |
| `CartPole_lif_sealed` | `s2014.npz` | 448.6 | 440.0 | 281.4 |
| `CartPole_qif_off` | `s2005.npz` | 235.3 | 312.8 | 120.2 |
| `CartPole_qif_on` | `s2004.npz` | 269.6 | 211.7 | 162.1 |
| `CartPole_qif_sealed` | `s2011.npz` | 242.2 | 222.0 | 191.7 |

A v4 model holds `W_ep<k>` (the weight snapshots; the last is the final model), `mask`, `m`, `roles`, `config` and `u`.
Run one with plasticity off through the ablation tool, which reads the final β from the training run's record:

```bash
python -m experiments.ablate --family v4 --model models/emergent-v4/CartPole_lif_on/s2017.npz --run reports/emergent-v4/runs/CartPole_lif_on/s2017.json --out X.json
```

## What a race model file holds (`save_ck_model`, `models/tdstdp.py`)

- `W`: the final input→post weights (n_post × n_in, CK units: mV before the trace scaling), rows = critic E, critic I, then per action the actor E (and I) posts; `sign`: +1 / −1 per post.
- `config`: the run's full `Config` as JSON after calibration, so `s_in`, `alpha_a`, `alpha_c`, `eta_c`, `eta_a` (after `--eta-cal`), `race_beta`, `race_theta`, `freeze`, `rate_floor`, `c_t` and the column flags are the values the run used; `n_in`, `n_ce`, `n_ci`, `n_ae`, `n_ai`, `K`, `n_lat`; `episodes_run`; `beta` (the race surrogate's gain at the end); `eta_sgd_mV`, `s_cal`, `rec_cal` (the calibration records).
- With recurrence (`cols_*` arms): `R` (the recurrent E→E weights among the actor posts), `mask` (which synapses exist; sealed columns have none across columns), `m`, `roles`, `ridx`, and the R snapshots `R_ep<n>` at `episodes` (what `experiments/structure.py` reads).

## Scripts (run from the repo root in the `map-stdp` env)

```bash
models/reproduce.sh race-cartpole 20      # rerun every reported run of the study with --save-model (deterministic per seed)
python models/verify.py race-cartpole     # the reproduced returns equal the reported ones bit for bit; every .npz in archive/, kept copies equal
python models/evaluate.py models/race-cartpole/CartPole_cols_sealed_race_rp/s7005.npz --episodes 20 --seed 1
```

`reproduce.sh` reruns the exact commands of `reports/<study>/manifest.json` through `experiments.report.study_command` with `--save-model` (for `race-cartpole` also `--race-fit-gate inf`: that study ran at `23cc0b9`, before the β re-probe fit gate was added at `74f3dd1`, so its runs accepted every re-probe; the gate-free setting reproduces them, which `verify.py` confirms); each run is deterministic given its seed and code state, and the code paths these studies use are byte-identical between the study commits and the commit that saved the models (`test_credit_default_path`'s fingerprints), which `verify.py` confirms on the returns. `evaluate.py` loads a model into a frozen runner (`load_ck_model`: no plasticity) and runs its policy in the environment; the policy is stochastic (noisy LIF, the race), so returns vary with the evaluation seed.

## Not here

- The softmax-readout studies (`hybrid-columns*`, `tdstdp-baseline*`) have no saved models; their reported tables are in `reports/`.
- Every emergent-v4 model is in `reports/emergent-v4/models/` (committed) and `archive/models/emergent-v4/`.
