# Study `latent-ablation`: what the latents carry in the final v4 models

**Pre-registered 2026-10-31, before any run of this study** (approved by the user; code at the state recorded in `manifest.json`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `contrasts.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-31):** take the final saved models and ablate their latents, to see exactly what is lost when they are gone: what the latents do, and how that depends on the model.
- **Scope (user decision 2026-10-31):** model v4 only, the 240 CartPole models of `emergent-v4` (four engines × three arms × 20 seeds). The TD-STDP race models have no latent neurons, and making TD-STDP latents that do something meaningful needs its own iteration. `experiments/ablate.py` also has a TD-STDP family (recurrent columns) that this study does not run.
- **The latents:** two modules of 16 neurons (L1, L2) between the controller (36) and the two action modules (16 each). The `emergent-v4` structure table shows two possible functions:
  - With the map term on, and sealed, the latents keep 10–28 % of the total weight as controller → latent routing but at most 1.3 % on latent ↔ action synapses (sealed: none).
  - With the map term off they keep 4–8 % latent ↔ action and 8–14 % into the controller.
  - So the latents may **relay** what the policy uses, or act as a **sink** that takes drive from the actions.
- **No training:** every model runs with plasticity off. One run is one saved model evaluated under every condition, on common random numbers (each condition restarts the same evaluation stream and faces the same initial states), so each model is its own control.
- **Design record:** `docs/iterations/017-ablation.md`; code `experiments/ablate.py` (family v4).

## Design

**How a cut is made.** Every engine divides a neuron's outgoing weights by their sum (T = W / d). So a cut either **loses** the drive (a lesion: the cut synapses' targets lose their input, nobody else's d changes) or **reroutes** it (a removal: cutting some of a neuron's targets renormalises its other outgoing weights).

**Models:** `reports/emergent-v4/models/CartPole_<engine>_<arm>/s<seed>.npz`, the final snapshot (episode 1500), with the run's final adaptive β (its last evaluation). Groups `v4_CartPole_<engine>_<arm>`, engines flow, branching, QIF, LIF, arms on (λ 0.05), off (λ 0) and sealed; seeds 2000–2019.

| Condition | Cut | Asks |
| --- | --- | --- |
| `intact` | none | the reference |
| `lesion_L` | the latents' outgoing synapses (W[:, L] = 0); the controller's drive into them is lost | do the latents **relay** anything the policy uses? |
| `lesion_L1`, `lesion_L2` | the same, one latent module | is the relay split between the modules? |
| `remove_L` | the latents disconnected (W[:, L] = W[L, :] = 0); the controller's outflow renormalises onto the actions | do the latents act as a **sink** that sets the actions' share of the controller's drive? |
| `latent_path_only` | controller → action synapses cut (W[A, C] = 0); the outflow renormalises onto the latents | can the latents alone carry the policy? |

On the flow engine a latent left without outgoing weight keeps a 1e-12 self-loop, because `core.flow` divides by d unguarded. The flow that enters it stays there, which leaves the controller's and the actions' flow, hence the policy, as if it were discarded. A frame with no action flow (flow) or no decision (spiking) takes a uniform action, as in training.

**Episodes:** 50 per condition. **Evaluation seed** 0 with `ABL_SEED` 4721. **Runs:** 12 groups × 20 models = **240**.

**Monitors (reported, not tested):**
- the silent-frame share;
- the action and latent modules' flow or spike shares;
- spikes per head and frame (spiking engines);
- the policy shift on the 100 held-out stimuli: total variation from intact, 10 frames per stimulus on spiking engines, with an intact-vs-intact noise floor;
- the intact frozen-return check against the training record (flow, branching and LIF reproduce it bit for bit; QIF's warm state is not saved).

## Pre-registered tests

**Primary metric:** per model, Δ = mean return under the condition − mean return intact (paired).

| Family | Tests | What |
| --- | --- | --- |
| `v4` | 12 groups × 5 conditions = 60 | two-sided Wilcoxon signed-rank of Δ over the 20 models, Holm within the family; one-sample Hodges–Lehmann shift with its 95 % CI |
| `v4-arm` (contrast) | 4 engines × {`lesion_L`, `remove_L`} × {off vs on, sealed vs on} = 16 | Δ of one arm against the on arm's: two-sided Mann–Whitney, Hodges–Lehmann, Holm within the family |

**Reading:** a condition **loses** if Holm p < 0.05 with a negative shift, **gains** if Holm p < 0.05 with a positive shift, and is **without material change** if the 95 % CI lies inside ±25 return. Otherwise it is inconclusive. Also reported: the normalised loss (intact − ablated) / (intact − random), the median over the models whose intact return clears the random policy by at least 10 % of the distance to 475. Models that collapsed in training (intact near random, e.g. several branching and QIF map-term-off models) carry no information on what was lost and drop out of this measure.

**Pre-registered expectations** (from the structure table and a smoke test on development models, `pilot014b` seeds 3–5; expectations, not results):
- **`lesion_L`:** no material change on map-term-on and sealed models, whose latents have little or no synapse onto the actions (a sealed LIF dev model was bit-identical). Map-term-off models keep 4–8 % latent ↔ action weight, so a loss is possible there; no direction is predicted.
- **`lesion_L1`, `lesion_L2`:** may differ from `lesion_L` where the two modules project unevenly onto the two actions (a flow dev model lost 90–180 under one module's lesion and nothing under both).
- **`remove_L`:** a loss on map-term-on models of flow and LIF (dev: flow about 490 → 150, LIF 150–320 → 110–150), the latents acting as a sink that holds the actions' share of the controller's drive (dev: the action share rose from about 0.33 to 0.8); sealed models are mixed.
- **`latent_path_only`:** a collapse to about random (≈ 22) on map-term-on and sealed models of every engine, which have no latent → action path.

**Not done:** no condition, episode count or seed changes after launch. A crashed run is relaunched; a group with fewer than 20 runs is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), threads set to 1, 60 workers. **Projected time:** under 1 h (dev: LIF 65–100 s per model at 10 episodes per condition, QIF about twice that).
- **Inputs:** the committed `reports/emergent-v4/models/`; each run records the sha256 of the model file it read.

## Commands

```bash
python -m experiments.report run latent-ablation --workers 60   # launch or resume every declared run; then summarize
python -m experiments.report summarize latent-ablation          # summary.csv, contrasts.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.ablate --family v4 --model reports/emergent-v4/models/CartPole_lif_on/s2000.npz --out reports/latent-ablation/runs/v4_CartPole_lif_on/s2000.json
```
