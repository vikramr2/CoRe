# Iteration 017: ablating the final models

## 1. Problem

The user's last experiment (2026-10-31): take the final saved models, v4 on CartPole and the TD-STDP backbones on the three tasks, ablate their latents, and see exactly what is lost. The aim is to learn what latent structure does and how that depends on the task and the model.

## 2. What there is to ablate

- **Model v4** (`reports/emergent-v4/models/`, 240 models): modules C (36), L1, L2 (16 each), A1, A2 (16 each). The study's structure table (`reports/emergent-v4/structure.md`) already shows the latents' place in the trained networks:
  - With the map term on, and sealed, the latents keep 10–28 % of the total weight as controller → latent routing but at most 1.3 % on latent ↔ action synapses (sealed: none), and almost nothing back to the controller.
  - With the map term off they keep 4–8 % latent ↔ action and 8–14 % into the controller.
  - So two functions are possible: a **relay** (what the latents pass on) and a **sink** (the drive they take from the actions).
- **The TD-STDP race models** (`archive/models/race-*/`, 320 models; the best run per arm in `models/`) have **no latent neurons** (`n_lat = 0` in every file). Their only structure beyond CK's feed-forward layer is the recurrent action columns. TD-STDP latent modules (`--rec lat`) exist in the code but were trained only in `hybrid-columns` (softmax, CartPole) without saved weights. This iteration ablates the columns' recurrence as the TD-STDP counterpart; ablating TD-STDP latents would need new training runs.
- **Sealed columns never learn their recurrence.** In every sealed race model R equals its initial `R_ep0`: exactly on LunarLander, to 1e-16 on Acrobot. One module-level modulator per column scales a neuron's outgoing kicks together, and the per-neuron normalisation undoes that. Sealed recurrence is therefore fixed random within-column recurrence; what sealed models learn is in W.

## 3. Method (`experiments/ablate.py`)

- **Lesion vs removal.** Every engine divides a neuron's outgoing weights by their sum, so a cut either loses the drive (a lesion) or reroutes it (a removal). See the conditions in the module docstring and `reports/latent-ablation/README.md`.
  - v4: `lesion_L`, `lesion_L1`, `lesion_L2`, `remove_L`, `latent_path_only`.
  - TD-STDP: `rec_off`, `cross_lesion`, `within_lesion` (lesions with the intact d), and the control `rec_untrained` (R_ep0), which keeps the recurrent drive and removes what the recurrence learned.
- **Evaluation:** plasticity off, every condition on the same evaluation stream (v4: environment resets from their own stream, so the same initial states).
- **Loading:** v4 loads the final snapshot with the run's last evaluated β. The intact model reproduces the training run's recorded frozen returns bit for bit on flow, branching and LIF, under both the study's code (`../map-stdp-ev4`) and the current code. QIF differs slightly (203.3 vs 197.2 on one model) because its warm state is not saved.
- **Flow detail:** a latent left without outgoing weight keeps a 1e-12 self-loop (`core.flow` divides by d unguarded). The flow that enters it stays there, which leaves the controller's and the actions' flow, hence the policy, unchanged.
- **Study runner:** `report.py` runner `ablate` and `summarize_ablate`. Paired Wilcoxon signed-rank on Δ = ablated − intact per model, a one-sample Hodges–Lehmann CI, Holm within each family, and declared checks excluded. Contrasts of Δ between arms use Mann–Whitney.
- **Tests:** `tests/test_ablate.py`.

## 4. Development smoke test (dev models only; not results)

v4, `pilot014b` (seeds 3–5, 10 episodes per condition). Mean return per condition:

| Model | intact | lesion_L | lesion_L1 | lesion_L2 | remove_L | latent_path_only |
| --- | --- | --- | --- | --- | --- | --- |
| flow on s3 | 495.7 | 495.6 | 405.4 | 494.0 | 140.4 | 31.1 |
| flow on s5 | 491.2 | 500.0 | 311.2 | 432.4 | 155.7 | 14.1 |
| LIF on s3 | 321.1 | 291.1 | 329.9 | 328.4 | 154.0 | 26.8 |
| LIF sealed s3 | 391.1 | 391.1 | 391.1 | 391.1 | 350.3 | 22.5 |
| LIF sealed s4 | 297.6 | 297.6 | 297.6 | 297.6 | 445.9 | 23.7 |

TD-STDP, dev seed 20 (CartPole and Acrobot full length; LunarLander 160 episodes only):

| Model | intact | rec_off | cross_lesion | within_lesion | rec_untrained |
| --- | --- | --- | --- | --- | --- |
| CartPole learned | 500.0 | 500.0 | 500.0 | 500.0 | 500.0 |
| Acrobot implicit | −171.8 | −500.0 | −338.9 | −478.9 | −500.0 |
| LunarLander sealed (short) | −227.9 | −361.1 | −227.9 | −361.1 | −227.9 |

## 5. Study

`latent-ablation` (user decision 2026-10-31): **v4 only**, 240 runs, pre-registered in `reports/latent-ablation/README.md`. TD-STDP latents need their own iteration to make latents that do something; the TD-STDP family of `experiments/ablate.py` stays available for the recurrent columns.

## 6. Results (`reports/latent-ablation/`, finished 2026-10-31 15:23 EDT, 240/240, none failed)

The intact check reproduced the training run's frozen returns on all 180 flow, branching and LIF models; QIF does not (warm state not saved), as declared. Δ = ablated − intact return per model, one-sample Hodges–Lehmann shift; Holm over the 60 tests. Numbers are in `reports/latent-ablation/summary.md` and `contrasts.csv`.

- **The latents cannot carry the policy.** `latent_path_only` collapses every group to about random (10–32; Holm ≤ 0.004, normalised loss 1.0). The policy runs through the direct controller → action routing.
- **Their main function is a sink.** `remove_L` loses in most groups. Flow: on −206, off −287, sealed −254. Branching on: −118. LIF: on −112, sealed −80. All Holm ≤ 0.002. Normalised loss 0.3–0.9. When the latents are gone the controller's whole outflow reaches the actions; their flow share rises from 0.2–0.3 to about 0.8 (flow), and the policy shifts on the held-out stimuli. The latents set the action modules' share of the controller's drive, and the trained policy depends on that share. Not significant: branching off and sealed, every QIF arm, LIF off.
- **Their output is not a relay the policy needs.**
  - Sealed: `lesion_L` is exactly zero on every engine (no latent → action synapses).
  - On: no material change on QIF and LIF; inconclusive on flow (HL −3 [−94, +4]).
  - Where the latents still drive the actions, cutting their output helps: branching on +47 (Holm 0.003), LIF off +38 (Holm 0.003); single-module lesions +6 to +18.
  - Flow on is the exception: one module's lesion loses (L2 −117, Holm 0.018; L1 −77, Holm 0.10) while both together do not. The two modules' small outputs onto the actions balance each other, so removing one biases the policy.
- **Arm contrasts (`v4-arm`):** under LIF, map-term-off models gain more from the lesion (+28, Holm 0.043) and lose less from removal (+71, Holm 0.043) than map-term-on models. Under branching, on models gain more from the lesion than sealed (+36, Holm 0.008) and lose more from removal (+72, Holm 0.005). The other 12 contrasts are not significant.
- **Against the expectations:**
  - `latent_path_only`: collapse, as expected.
  - `remove_L`: losses on flow and LIF map-term-on, as expected, and on most other groups too.
  - `lesion_L`: sealed exact and on QIF/LIF unchanged, as expected; but a gain on branching on and LIF off, which was not predicted.
