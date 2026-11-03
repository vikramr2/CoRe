# Study `emergent-v4`: does the map term help on CartPole, on four engines?

**Pre-registered 2026-10-27 14:20 EDT, before any run of this study** (approved by the user; code at git `086996f`, branch `emergent-v4-study`). Design and tests below are fixed.

Nothing in this directory is written by hand except this README:
- `manifest.json`, `code.patch`, `runs/`, `models/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`;
- `structure.*` comes only from `python -m experiments.structure`.

## What the study is

- **Model:** Map-STDP with the walk observed from spikes (`docs/derivation.md`; `--rule emergent`), on model v4's flags, on four engines: flow, branching, QIF and LIF (derivation §5).
- **The question:** in this minimal v4 backbone, does the map term (λ = 0.05) improve task performance and lower description length, compared with the same agent without it (λ = 0)?
- **A third arm, sealed,** asks whether the task benefit comes from the structure itself or from the process that learns it (iteration 014 §7): λ = 0 on a network built without the cross-module synapses the map term removes.
- **Task:** CartPole only. LunarLander moved to the TD-STDP backbone study (iteration 014 §10, user decision 2026-10-27).
- **Development results** are not part of this study. They come from the two pilots of iteration 014 (`docs/iterations/014-emergent-v4.md`; `experiments/results/pilot014/` and `pilot014b/`; seeds 0–5), which set the settings below.

## Design

- **Common flags:** `--rule emergent --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --eta 5`. These are v4:
  - TD-normalised η;
  - no controller→controller weights;
  - adaptive β;
  - la_beta 0.3 (the default).
  - The routing pairs are exempt from the map term, as in v4.
- **Engines** (`report.M3_ENGINE_FLAG`, plus the flags shown):

| Engine | Flags | Walk and counter |
| --- | --- | --- |
| `flow` | `--backend flow --beta-max 16` | flow walk, exact (Eq. 8 with no chance) |
| `branching` | `--backend spiking --engine numpy --heads 8 --beta-max 16 --readout diffbound --match-scales` | lag-1 pair counts, exact counter (Eq. 11a) |
| `qif` | `--backend spiking --engine qif --heads 8 --beta-max 16 --readout diffbound --match-scales` | window pair counts, counter (11b), gain regressed on the injected noise; fixed inhibitory pools outside W |
| `lif` | `--backend spiking --engine lif --heads 8 --beta-max 1.6 --readout diffbound --match-scales` | noisy LIF (τ_m 20 ms), 60 ms window, counter (11b) with Siegert gain (11c) |

- **8 heads.** The second pilot gave the same returns as 16 heads at about half the cost (iteration 014 §10). The diffbound tables for 8 heads (`m2.BOUND_CAL[8]`, `BOUND_CAL_QIF[8]`, `BOUND_CAL_LIF[8]`) come from `experiments.bound_cal` on the first pilot's flow CartPole models.
- **LIF's lower β cap** (1.6, not 16) keeps its readout inside its calibrated range (iteration 014, §5, decision 1).
- **`--match-scales`:** each spiking engine's λ and η are scaled so that its RMS updates at the initial W match flow's (a fixed rule, seed 5521, no tuning).
- **Arms:** identical except for:

| Arm | Flags |
| --- | --- |
| `on` | `--lam 0.05` |
| `off` | `--lam 0` |
| `sealed` | `--lam 0 --p-cross 0` (no cross-module synapses outside the routing pairs; same SBM draws, only the mask differs) |

- **Groups:** `CartPole_<engine>_<arm>`: 4 engines × 3 arms = 12 groups, 20 seeds each, 240 runs, 1500 episodes per run.
- **Seeds:** 2000–2019. They are fresh: no earlier run used seeds at or above 2000.

## Pre-registered tests

Two-sided Mann–Whitney per engine. Holm correction is applied within each family, separately for each metric.

| Test | Contrast (family) | Metric |
| --- | --- | --- |
| **P1: the map term improves performance** | on vs off, 4 engines (`primary`) | AUC = mean return over all 1500 training episodes, per run |
| **P2: the map term lowers description length** | on vs off, 4 engines (`primary`) | $L_{obs}$ = held-out map equation (bits) of the engine's observed walk at the last evaluation (episode 1500), per run; for flow this is $D$ |
| **S: learned vs prescribed structure** | on vs sealed, 4 engines (`sealed`) | the same AUC and $L_{obs}$ |

- **Support criteria:**
  - An engine **supports** P1 if its Holm-adjusted p < 0.05 and the on arm's mean AUC is higher.
  - An engine **supports** P2 if its Holm-adjusted p < 0.05 and the on arm's mean $L_{obs}$ is lower.
  - A significant difference in the other direction is reported as a **reversal**.
- **Reading S** (iteration 014 §7):
  - on ≈ sealed (no significant difference): the task benefit is the structure, and the map term is a local rule that finds it.
  - on > sealed: the map term contributes beyond cutting links.
  - sealed > on: having the structure from the first episode matters more than learning it.
  - The sealed arm starts at low $L$ by construction, so S on $L_{obs}$ asks whether the map term shapes the walk beyond sealing. The second pilot suggests it does on the spiking engines.
- Every engine is reported, pass or fail.

**Secondary (reported, not tested for the claims):**
- the last-100 return, the frozen-W return (20 episodes with learning off at the end of the run, `m2.frozen_returns`), and the flow-walk $D$ at the last evaluation, with the same pairs (`pairs.csv`);
- the number of runs whose trailing-100 mean reaches 475, and the first such episode;
- the monitors from the result files: ρ(Λ̂^K), causal share, M*, silent-module frames and clip rate;
- network structure from the saved models (`structure.md`, `structure_*.png`): weight share per module block, the module weight table, and the engine-free $L$ of W's own walk, compared across the three arms;
- descriptive comparisons with the TD-STDP baseline (`reports/tdstdp-baseline*`) and with random policies.

**Not done:** no settings change after launch, and no seeds are added or dropped. A run that crashes is relaunched from its checkpoint. If a group still has fewer than 20 runs, it is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), 32 workers, with `OMP`, `MKL`, `OPENBLAS` and `NUMBA` threads set to 1 (`report.run`).
- **Code state:** the study runs from a git worktree of a commit made for it, so the main checkout can change for iteration 015.
- **Projected time:** about 6–10 h in total. Under the second pilot's 48-way load, ms per frame was:
  - flow ≈ 1.8;
  - branching ≈ 4.7 and LIF ≈ 5.3 (8 heads);
  - QIF ≈ 30 (16 heads; less at 8).
  - Episodes lengthen as the policy improves, so the later episodes cost the most.
- **Checkpoints:** each run checkpoints every 50 episodes (`runs/<group>/s<seed>.ckpt.npz`, gitignored, deleted on completion), and a relaunch resumes it. A resumed run is not bit-identical; the count is in its `resumes` field and in `summary.csv`.
- **Models:** `models/<group>/s<seed>.npz`, about 22 KB each and about 5 MB in total. Each holds W at episode 0, every 250 episodes and the end, plus the mask, partition, roles, config and critic.
- **One code state:** `manifest.json` records the git HEAD and the sha256 of the uncommitted code in `mapstdp/` and `experiments/`, and `code.patch` holds that code. The runner refuses to relaunch if the code differs. If code a run executes has to change after launch, the whole study is deleted and relaunched.

## Commands

```bash
python -m experiments.report run emergent-v4 --workers 32   # launch, or resume, every declared run; then summarize
python -m experiments.report summarize emergent-v4          # rebuild summary.csv, pairs.csv and summary.md from runs/
python -m experiments.structure emergent-v4                 # structure.csv, structure.md, structure_*.png from models/
```

The exact per-run commands are in `manifest.json` (`groups.<group>.commands`). For example:

```bash
python -m experiments.m3 --task CartPole --seed 2000 --backend spiking --engine lif --rule emergent --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --eta 5 --beta-max 1.6 --heads 8 --readout diffbound --match-scales --lam 0 --p-cross 0 --episodes 1500 --out reports/emergent-v4/runs/CartPole_lif_sealed/s2000.json --save-model reports/emergent-v4/models/CartPole_lif_sealed/s2000.npz
python -m experiments.m3 --task CartPole --seed 2019 --backend flow --rule emergent --td-norm 0.04 --p-cc 0 --beta-kappa 0.01 --eta 5 --beta-max 16 --lam 0.05 --episodes 1500 --out reports/emergent-v4/runs/CartPole_flow_on/s2019.json --save-model reports/emergent-v4/models/CartPole_flow_on/s2019.npz
```
