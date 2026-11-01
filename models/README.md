# Final models of the reported race studies

The best run of each arm of the race-readout studies on the TD-STDP backbone (iteration 016 §20–21; the recipe in
`docs/columns-recipe.md`), one `.npz` per arm, plus the scripts that reproduce, verify and run them.

**Every model is in `archive/models/`** (gitignored, 316 MB, to be kept on a server): all 320 race runs, one `.npz`
per run in the same `<study>/<arm>/s<seed>.npz` layout, and a copy of the 240 emergent-v4 models. `reproduce.sh` writes
there; this directory keeps the copies of the best runs only.

| Study | Task | Arms (one directory each) | Seeds | Reported result |
| --- | --- | --- | --- | --- |
| `race-cartpole` | CartPole | `CartPole_base_ck`, `CartPole_base_nl`, `CartPole_base_race_rp`, `CartPole_cols_lam0_race_rp`, `CartPole_cols_on_race_rp`, `CartPole_cols_sealed_race_rp` | 7000–7019 | `reports/race-cartpole/summary.md` |
| `race-acrobot` | Acrobot | `Acrobot_base_ck`, `Acrobot_base_race_h2`, `Acrobot_cols_lam0_race_h2`, `Acrobot_cols_on_race_h2`, `Acrobot_cols_sealed_race_h2` | 7100–7119 | `reports/race-acrobot/summary.md` |
| `race-lunarlander` | LunarLander | `LunarLander_base_ck`, `LunarLander_base_race_e`, `LunarLander_cols_lam0_race_e`, `LunarLander_cols_on_race_e`, `LunarLander_cols_sealed_race_e` | 7200–7219 | `reports/race-lunarlander/summary.md` |

Each study's `README.md` in `reports/<study>/` is its pre-registration (arms, flags, settings and how they were chosen); `summary.md` and `pairs.csv` there are the reported tables.

## Best run per arm (the files kept here)

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

## What a model file holds (`save_ck_model`, `experiments/tdstdp.py`)

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
- Map-STDP's own models (the emergent-walk v4 study) are in `reports/emergent-v4/models/`.
