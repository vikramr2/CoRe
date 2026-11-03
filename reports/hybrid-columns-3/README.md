# Study `hybrid-columns-3`: Map-STDP columns in the TD-STDP backbone on three tasks, matched starts, normalised scores

**Pre-registered 2026-10-28 18:22 EDT, before any run of this study** (approved by the user; code at commit `4bdbe45`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-28):** across CartPole, LunarLander and Acrobot, do action columns in the TD-STDP backbone keep or improve performance against the same backbone without them, and which regime of modularity (implicit, learned, enforced) does best, once every arm starts from the same gain and keeps it?
- **What came before:**
  - `reports/hybrid-columns` (iteration 015): CartPole and LunarLander, unmatched starts. No regime improved; sealed was non-inferior on both tasks, the others on LunarLander. Its early LunarLander speed-up was mostly a starting-gain effect (iteration 016 §10–11).
  - `reports/hybrid-columns-2` (iteration 016): Acrobot, matched starts. Implicit and enforced columns were worse than base. The cause was the entropy bonus silencing an E-only actor, a drift the within-column loop doubled (016 §13); not a cost of the columns.
  - Development pilots (016 §13, dev seeds 0–15): with the normalised action scores the drift is gone and the column arms are level with base on Acrobot.
- **This study** reruns the three regimes on all three tasks under one fair readout. It is the backbone result set. Its base is its own reference; the CK-faithful bases are those of `hybrid-columns` and `hybrid-columns-2`.
- **Backbone, engine and arms:** as `hybrid-columns`: the TD-STDP actor-critic of Chung & Kozma 2020 without Adam, on noisy LIF (`--engine nlif`, dt 1 ms, τ_m 20 ms, τ_ref 2 ms) at an initial post rate of 40 Hz. Each action's actor E population is one column with recurrent E→E synapses R.
- **Design record:** `docs/iterations/016-columns-more-tasks.md` §10–15. The pilots (`experiments/results/pilot016/`) set the frozen settings below. They are not part of the study.

## Design

- **Common flags, every arm:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --ent-match-rec --norm-logits`; LunarLander and Acrobot add `--eta-cal`.
  - `--eta-cal`: the calibrated local step (η = the authors' Adam step / RMS of the update at init) on LunarLander and Acrobot, as in the earlier studies. CartPole keeps CK's preset η, as in `hybrid-columns`: its preset η are SGD learning rates, not Adam steps, and calibrating them gives η of 8.7 mV, which silences the actor (pilot, 016 §15).
  - `--rate-match-rec`: s_in calibrated with the arm's R at init, so every arm starts at 40 Hz (016 §10).
  - `--ent-match-rec`: α_a scaled per arm so the initial policy entropy equals the base run's (016 §11; the factors are 0.95–1.05 under the normalisation).
  - `--norm-logits`: the action scores α_a r_k r₀ / max(r̄, ε), normalised by the pooled actor rate against the pre-set 40 Hz, with the exact gradient through the normalisation for the actor eligibility and the entropy bonus (016 §13). It makes the policy invariant to a common change of all actor rates, which removes the entropy bonus's silencing drift and the loop's common-mode gain from the policy. ε = α_a r₀ / (τ_n n_ae), the pooled rate at which one spike moves a score by one nat. Base gets it too: the readout is the same in every arm.
  - This is the third gain-control choice (E-only drift → opponent I posts, withdrawn → normalisation) made after seeing Acrobot returns on dev seeds 0–15. It is recorded as a post-hoc readout change; this study's seeds are fresh.
- **Task settings:** the presets of `experiments/tdstdp.py`: CartPole's and LunarLander's are the authors' configs; Acrobot's follows the pre-set rules of 016 §4 (CartPole's populations and α_a, LunarLander's encoder and entropy bonus, r_scale 0.02, β_c −1.2).
- **Arms** (identical except for):

| Arm | Flags | What it is |
| --- | --- | --- |
| `base` | – | no recurrence (R = 0): the matched no-Adam baseline, normalised scores |
| `cols_lam0` | `--chi-rec 0.3 --rec cols --lam 0` | implicit: recurrent action columns, task term on R only |
| `cols_on` | `--chi-rec 0.3 --rec cols --lam 1` | learned: columns plus the map term (r = 1) |
| `cols_sealed` | `--chi-rec 0.3 --rec cols --rec-seal --lam 0` | enforced: the same draws of R without its cross-column synapses (once sealed, R gets no task credit: a fixed random within-column amplifier) |

- **Frozen settings:** χ_rec 0.3 and r = 1, as in `hybrid-columns`. Every arm of a seed shares the low-rate scaling k and η_rec / λ_rec (measured on the unsealed draws); s_in and α_a are per arm by the matching rules.
- **Episodes:** as the earlier studies. CartPole 400. LunarLander pooled over 16 copies, at least 3000 and at most 10000, stopping once solved (CK's criterion) after 3000. Acrobot 500, no early stop.
- **Groups:** `<task>_<arm>`: 3 tasks × 4 arms = 12 groups × 20 seeds = 240 runs.
- **Seeds:** 5000–5019, fresh. No earlier run used seeds at or above 5000.

## Pre-registered tests

**Primary metric:** LunarLander last-100 return; AUC (the mean return over the fixed episodes) on CartPole and Acrobot.

**Contrasts:** two-sided Mann–Whitney on the primary metric, Holm-adjusted within each family per task. Effect size: Hodges–Lehmann shift with a 95% CI.

| Family (per task) | Contrast | Asks |
| --- | --- | --- |
| **headline** (`primary`, 3 tests) | `cols_lam0` vs `base` | does implicit modularity keep or improve performance? |
| | `cols_on` vs `base` | does learned modularity (Map-STDP) keep or improve performance? |
| | `cols_sealed` vs `base` | does enforced modularity keep or improve performance? |
| secondary (`secondary`, 3 tests) | `cols_on` vs `cols_lam0` | what does the map term add over implicit structure? |
| | `cols_on` vs `cols_sealed` | learned vs enforced structure |
| | `cols_lam0` vs `cols_sealed` | do cross-column links matter? |

For each headline contrast:
- **Improves:** Holm-adjusted p < 0.05 and a positive shift.
- **Keeps performance (non-inferiority):** the lower end of the 95% CI of the shift is above −Δ, with **Δ = 25** (25 points of last-100 return on LunarLander, 25 points of AUC on CartPole and Acrobot; as in the earlier studies, confirmed by the user).
- **Worse:** a significant negative shift.

Every contrast is reported, pass or fail. The claim the study can support is non-inferiority of columns across three tasks at matched gain, and which regime, if any, differs.

**What the pilots suggest** (development, dev seeds, so these are expectations, not results; 016 §13 and §15):
- **Acrobot** (seeds 0–9): column arms level with base (AUC shifts +13, +4, −2; p ≥ 0.39), no collapsed run (0/30), base median AUC −186. Rates hold at 40–50 Hz through 500 episodes.
- **CartPole** (seeds 0–4): every arm solved 5/5. AUC shifts vs base −16 (`cols_lam0`), −1 (`cols_on`), −43 (`cols_sealed`, p 0.03 at n = 5: sealed columns solved later, T_s 256–356 vs 221–269). The matching held (f 0.95–0.99, entropies matched to 0.01). So "worse" or "inconclusive" for sealed columns on CartPole is a live possibility.
- **LunarLander** (seeds 10–12, 1000 episodes only): rates hold (0.040–0.045 per ms), entropies level; AUC over 1000 episodes level with base (−1, −17, −7; n = 3). Too short to predict the primary metric.
- Two development bugs found and fixed by these pilots (016 §15): `--eta-cal` misapplied to CartPole (its preset η are SGD rates), and a NaN in the entropy probe on tasks with short episodes. Neither affects the earlier studies.

**Secondary (reported, not tested for the claims):**
- solved count, T_s, last-100 return; LunarLander AUC over the first 3000 episodes; Acrobot and CartPole last-100;
- collapsed runs (Acrobot last-100 = −500);
- the matching record per run (`low_rate`: `r_got_hz`, `ent_scale`, `h_base`, `h_got`);
- the module weight table of R per snapshot (`rec.RK`) and the within-column share;
- **the stimulus-conditioned walk** (`rec.RKc`, new): per snapshot and per executed action, the module-level pair-count table of R as a share of that class's pairs, with the class's count. It asks whether the walk between columns depends on the state at all, which a stimulus-conditioned map term (016 §15) would need. A monitor only: the same integer pair counts, split by class; it changes no update and draws nothing;
- stability monitors (max Re λ, τ_loop, χG), rates, policy entropy, action repeat vs chance;
- ms per frame.

**Not done:** no settings change after launch, and no seeds are added or dropped. A crashed run is relaunched from its checkpoint. If a group still has fewer than 20 runs, it is reported with its n.

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP`, `MKL`, `OPENBLAS` and `NUMBA` threads set to 1, 64 workers.
- **Code state:** git worktree `../map-stdp-hc3` (branch `hybrid-columns-3-study`) at commit `4bdbe45`.
- **Projected time:** 12–16 h, set by LunarLander (0.4–0.9 h per 3000 episodes, unsolved runs to 10000). CartPole runs take 5–10 min, Acrobot 2–6 min.
- **One code state:** `manifest.json` records the git HEAD and the sha256 of the uncommitted code; the runner refuses to relaunch if it differs.

## Commands

```bash
python -m experiments.report run hybrid-columns-3 --workers 64   # launch or resume every declared run; then summarize
python -m experiments.report summarize hybrid-columns-3          # summary.csv, pairs.csv and summary.md from runs/
```

For example:

```bash
python -m experiments.tdstdp --task LunarLander --seed 5000 --engine nlif --rate-target 40 --block 20 --fast --eta-cal --rate-match-rec --ent-match-rec --norm-logits --chi-rec 0.3 --rec cols --lam 1 --out reports/hybrid-columns-3/runs/LunarLander_cols_on/s5000.json
```
