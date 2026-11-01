# Changelog

## 2026-10-31

### Added: the walk replay of the best models, `experiments/viz.py` and `viz_template.html` (user request)

- **User decisions:** all 28 best models; flow shown as sampled walkers; an snn-expert review of the cause rule before implementation.
- `viz.py` records one frozen episode per model (the best of 10 on environment seeds 1000+k). It captures the spikes by wrapping module functions for the duration of the recording. No random draw moves: `tests/test_viz.py` checks that recorded and unrecorded episodes match on v4 flow, v4 branching and race CartPole.
- Each displayed spike (head 0 of v4, copy 0 of a race model) gets one sampled cause: a parent spike (a walk step), the stimulus or background. The draw uses the parent posterior of a linear (Hawkes) cluster process.
  - **Branching:** exact (parent ∝ T_ij at t−1).
  - **LIF, QIF and noisy LIF:** linear-response approximations with pre-set response kernels and background rates measured on one model each (preliminary). The hop counts reproduce the counter c of Eq. 11b, not the observed walk κ. Earlier presynaptic spikes erased by a reset are excluded, and on race models only R→R parents are walk steps.
  - **Flow:** 32 walkers per frame sampled from Eq. 1.
- **Review findings (snn-expert, preliminary):**
  - A cause rule without a background cause, using the membrane kernel, would put 19–31% of L/A outflow into C, against 0.1–0.3% in c.
  - With the null cause and the measured kernels, per-synapse hops correlate 0.97 with c.
- **Data:** 28 files, 39 MB, race input spikes stored as per-window counts. Spike causes per episode:
  - v4 LIF: about 60% walk steps, 12% stimulus, 28–33% background.
  - Race models: 0–1% walk steps on CartPole, 6–9% on Acrobot and LunarLander column arms; the rest is input or background.
- `viz_template.html`: task and model selectors; the network laid out by module role; CartPole, Acrobot and LunarLander drawings. Published on `viz` as `index.html` with `data/`; the v3 page moves to `v3-replay.html`.

### Changed: the replay page moves to branch `viz` for GitHub Pages (user request)

- New orphan branch `viz` holds only the page, as `index.html`, plus `.nojekyll` and a README. Once Pages is enabled on that branch, it is served at https://vikramr2.github.io/map-stdp/.
- `replay.html` is removed from `dev` and gitignored. The generator `experiments/m2_viz.py` and its template stay on `dev`; `m2_viz page --seed 4 --la-beta 0.3` rebuilds the published page byte for byte.

### Added: `README.md` (user request)

- A short top-level README covering setup, how to run the trained models in `models/`, and how to train the v4 and TD-STDP race models (commands taken from the study manifests, and smoke-tested). The working branch `columns-tdstdp` is now `dev`.

### Added: `models/emergent-v4/`, the best v4 model per arm (user request)

- 12 byte copies (628 KB) from `reports/emergent-v4/models/`, one per engine × arm. **Best = the highest frozen return** (20 end-of-run episodes with the final weights frozen), not last-100: a v4 model is the final snapshot, and its last-100 return was played while the weights still moved (QIF on: one run with last-100 293 scores 10 frozen). Ties go to last-100, then mean return, then the lower seed. The picks hold on `latent-ablation`'s 50 independent intact episodes, where each non-flow pick is first in its arm, except QIF sealed, where it is second; every flow pick plays at about 500. Table in `models/README.md`.

### Changed: `models/` keeps the best run per arm; every model moves to `archive/models/` (gitignored) (user request)

- **Decision (user):** the repo keeps only the best model of each arm; the full set goes to a server later.
  - `models/<study>/<arm>/` now holds one `.npz` per arm, 16 files, 16 MB. Best means the highest last-100 return of the reported run; ties (every CartPole arm at 500) are broken by the mean return over the run, then the lower seed. The table is in `models/README.md`.
  - `archive/models/` (gitignored) holds every saved model: the 320 race runs in the same layout, and a copy of the 240 emergent-v4 models, which stay committed in `reports/emergent-v4/models/` because `latent-ablation` reads them.
- `models/reproduce.sh` now saves to `archive/models/`. `models/verify.py` reads the archive and checks that each kept copy equals its archived file. All three studies verify: 120/120, 100/100 and 100/100 bit for bit.

### Merged: `hybrid-columns-3-ck-study`, `hybrid-columns-4-study` and `race-lunarlander-study` into `columns-tdstdp` (`fc412a2`)

- `race-lunarlander`'s results were committed on their study branch first (`7b0768a`). Every study branch is now in `columns-tdstdp`.

### Finished: study `latent-ablation` (iteration 017 §6)

- 240/240 runs, 25 min, none failed. **What the v4 latents do:** they are a sink, not a relay.
  - Removing them reroutes the controller's whole outflow onto the actions and loses return in most groups (flow −206 to −287, branching on −118, LIF on −112 and sealed −80; Holm ≤ 0.002).
  - Cutting only their output changes nothing where they have no path to the actions (sealed: exactly zero), and helps where they still drive the actions (branching on +47, LIF off +38).
  - With the direct routing cut, the latents alone cannot carry the policy (every group to random).
- The intact check reproduced the training record on all 180 flow, branching and LIF models.

### Added: ablation of the final models, `experiments/ablate.py` and study `latent-ablation` (iteration 017; user request)

- `experiments/ablate.py` evaluates a saved model with plasticity off under every condition of its family, on common random numbers.
  - **v4** (emergent-v4 models): the latents lesioned (all, or one module), removed (rerouting the controller's outflow onto the actions), or left as the only path to the actions.
  - **ck** (race-study models): the recurrence off, the cross-column or within-column kicks lesioned, or the recurrence at its untrained weights (`R_ep0`).
- `report.py`: runner `ablate`, `summarize_ablate` (paired Wilcoxon on ablated − intact per model, one-sample Hodges–Lehmann, Holm within a family, declared checks excluded; contrasts between arms by Mann–Whitney), and the study `latent-ablation`. **Decision (user):** the study runs the v4 family only (240 runs); TD-STDP latents need their own iteration before there is anything meaningful to ablate. Pre-registered and launched 2026-10-31 (`reports/latent-ablation/README.md`). `tests/test_ablate.py`; the existing suite and golden fingerprints are unchanged.
- **Finding:** the TD-STDP race models have no latent neurons (`n_lat = 0` in all 320), so the study ablates their recurrent columns as the counterpart; ablating TD-STDP latents would need new training runs.
- **Finding:** sealed columns never learn their recurrence; R stays at `R_ep0` (exactly on LunarLander, to 1e-16 on Acrobot), because one module-level modulator per column is undone by the per-neuron normalisation.
- **Finding:** a saved v4 model with the run's final β reproduces the training run's frozen returns bit for bit on flow, branching and LIF; QIF's warm state is not saved.

## 2026-10-30

### Finished: study `race-lunarlander` (iteration 016 §21)

- 100/100 runs, 17.4 h, none failed (manifest `431403b`). Every race arm beats CK's softmax on last-100 after Holm: the race alone +23.8 [+12.1, +32.3], implicit columns +41.1, learned +37.6, sealed +35.9 (Holm ≤ 1e-4); early learning level; about 30 % of the actor spikes per decision. Columns beat the race without them (learned, sealed: Holm 0.005, 0.011); the regimes do not differ. Results in `../map-stdp-racl/reports/race-lunarlander/`, uncommitted on branch `race-lunarlander-study`; weight-saving reproduction of all 100 runs launched.

### Added: `models/`, the final weights of the reported race studies; `save_ck_model` / `load_ck_model` (user request 2026-10-30)

- `--save-model` on a CK run now writes the final model: W, the calibrated config as JSON, the Layout sizes and signs, the race's β and the calibration records, and with recurrence R, its mask and the R snapshots (the keys `experiments.structure` read before). `load_ck_model` rebuilds a frozen runner from it. `test_save_and_load_ck_model`. Default path unchanged.
- `models/README.md`, `models/reproduce.sh` (reruns a study's reported runs with `--save-model`; deterministic per seed), `models/verify.py` (bit-for-bit returns against `reports/<study>/runs/`), `models/evaluate.py` (runs a saved model with plasticity off). **All 220 reported runs reproduced bit for bit with their final models saved**: `race-cartpole` 120/120 (12 MB), `race-acrobot` 100/100 (30 MB), in `models/<study>/<arm>/s<seed>.npz`.
- `race-lunarlander`'s 100 runs reproduced with `--save-model` (user request; 2026-10-30 20:14 to 2026-10-31 13:39 EDT, one wave at 100 workers): **100/100 bit for bit with their models saved** (275 MB, `models/race-lunarlander/`); `models/verify.py` takes an optional reports root for a study whose results are still on its branch.
- `--race-fit-gate` (default 0.05, unchanged; `inf` accepts every re-probe, token `fg<v>`; `test_race_fit_gate_option`): `race-cartpole` ran at `23cc0b9`, before the re-probe fit gate (`74f3dd1`), so its race arms reproduce only with the gate off (with the gate on, 6 of 20 no-column seeds matched); `models/reproduce.sh` passes `inf` for that study. Acrobot uses no re-probe. 84 tests, golden unchanged.

### Merged: `race-cartpole-study` and `race-acrobot-study` into `columns-tdstdp` (`e1812f3`)

- The reported tables and runs of both race studies are on the branch (`reports/race-cartpole/`, `reports/race-acrobot/`).

## 2026-10-29

### Result: LunarLander pilot with the E-rate surrogate; the race with columns beats the backbone on dev seeds (iteration 016 §21)

- 30/30 full-length dev runs, none failed or collapsed. Last-100 vs CK's softmax (62.3): sealed columns 93.8 (+31.2 [+13.9, +48.6], p 0.0008), learned 79.8 (+16.9 [+7.4, +26.9], p 0.005), the race without columns 69.0 (+6.4, n.s.); early learning level (AUC-3000 within ± 8); 30 % of the softmax's actor spikes to decision at a held 38 Hz. Columns beat the race without them (p 0.01, 0.03); sealed highest, the third task in a row. The old-surrogate ablation (9 runs) complete: no columns 108, sealed 72, learned 51, with the 40–60-point early deficit. `race-lunarlander` README's expectations updated from these numbers; awaiting approval and commit.

### Drafted: study `race-lunarlander` (iteration 016 §21)

- Declared in `report.py` (seeds 7200–7219, 100 runs, primary last-100; CK's softmax at the matched start as the reference on the same seeds; the race with the E-rate surrogate, the band, θ 26 at a 20-ms decision held at the preset's 40-ms control rate, on no / implicit / learned / sealed columns); README drafted for approval. Dev pilot (30 full-length runs) in progress: at episode 5700 every race arm 20–25 last-100 points above the backbone with no collapse; three old-surrogate no-column runs at the cap: last-100 82–128 vs the backbone's 62.

### Fixed: the race's surrogate reads the E column rate; policy credit on E rows; the calibration's actor RMS over the credited rows (iteration 016 §21)

- `CKRunner.MaE` (`Ma` with its −1/n_ai entries zeroed) replaces `Ma` wherever the race's surrogate reads column rates (window-end `prk`, the module scores, the contrast probe, the re-probe, the `H(s)` monitor); under the race `c` and `kap` are zero on I actor rows. Bit-identical where n_ai = 0 (CartPole, Acrobot: `MaE` is `Ma`; fingerprints unchanged; `race-cartpole` and `race-acrobot` untouched); on LunarLander the probe now gives β 12 with fit 0.02 (was 1.4, fit 0.24). `calibrate_ck` under the race takes the actor RMS over the credited E rows (otherwise the silent I rows dilute the E4' rule's RMS and inflate η_a by √2 where n_ai > 0; a pre-set, no-op where n_ai = 0). Tests `test_race_surrogate_e_rate`, `test_race_credit_e_rows`, `test_race_e_rate_invariance_n_ai0`; `test_credit_score_run`'s I-row assertion flipped (I rows now stay at init under the surrogate too). 82 tests, golden unchanged. The `Config.freeze` comment restates the hold rule (decide within 20 ms; hold for the preset's freeze / 20 steps).
- LunarLander full-length pilot launched (`pilot016_ll2.sh`, dev seeds 20–29 × base / sealed / learned, 30 runs, 30 workers; patch `ll2_code.patch` in the scratchpad).

### Diagnosed: the race's late start on LunarLander; the surrogate read the net rate (iteration 016 §21)

- The race's LunarLander climb matches the backbone's but starts about 1000 episodes later (at 4000: +15 vs +15 / +25). Cause (snn-expert): under the race the surrogate p̃ read CK's *net* column rate E − I (`Layout.Ma`), while the race tallies E counts (§18); with LunarLander's n_ai = 32 the net rate sits below the surrogate's floor 67 % of the time, the credit is mostly dead and β = 1.4 is a floor artefact (the probe's fit 0.21 at init). Reading the E column rate gives β 10–12, fit ≤ 0.03. With that and the hold at the backbone's own control rate (the preset's 40-ms freeze = 2 env steps; the race deciding within 20 ms at θ 26) the race is level with the backbone at 1500 episodes (last-50 −46 vs −52; AUC −131 vs −124) at 177 vs 370 actor spikes to decision; neither change alone suffices. The mean-rate pin, θ, the entropy bonus, the step and the copies were rejected as causes. Decision: the surrogate reads what the race tallies (E rows get the policy credit; a no-op where n_ai = 0, so `race-cartpole` and `race-acrobot` are untouched), and the hold rule is restated as "the race decides within 20 ms and holds for the preset's freeze / 20 steps". Implementation by the snn-expert in progress; the full-length LunarLander pilot (`pilot016_ll2.sh`, 30 runs) follows.

### Finished: study `race-acrobot` (iteration 016 §21)

- 100/100 runs, 12 min, none failed (manifest `74f3dd1`). Race + hold + sealed columns non-inferior to CK's softmax at β_c −1.5 (+13.0 AUC [−5.0, +38.6]; last-100 −132 vs −147; 2/20 solved vs 0/20); learned −5.0 (inconclusive by 2.7 at Δ = 25); implicit −25, no columns −35 (inconclusive). **Regime matters after Holm:** sealed vs none +53.6 (0.0027), sealed vs implicit +42.2 (0.011). Energy: the race spends more actor spikes per decision here (30–33 vs 26; the band pins 40 Hz where the softmax's rate falls). Results in `../map-stdp-racr/reports/race-acrobot/`, uncommitted on branch `race-acrobot-study`.

### Result: hold pilot on Acrobot; study `race-acrobot` drafted (iteration 016 §21)

- 40/40 dev runs through the real flags: with the hold and the band, sealed columns are level with CK's softmax at β_c −1.5 (AUC −174 vs −174, +3 [−27, +25]; last-100 −132 vs −150; 2/10 solved vs 0/10; fewer spikes), learned columns level (−4), implicit −43 and no columns −37 (worse). No floor runs. Ordering sealed ≈ learned > implicit > none. `race-acrobot` declared (seeds 7100–7119, 100 runs, reference CK's softmax at β_c −1.5, families headline / secondary); README drafted for approval. LunarLander full-length pilot running.

### Added: `--freeze`, the re-probe fit gate, `action_repeat_probability` (iteration 016 §21; neuroscientist review of the hold)

- `--freeze MS` sets the decision window (token `fz<ms>` when it differs from the preset); with `--race-theta` from the 20-ms rule it is the decision hold (`--freeze 40 --race-theta 16`: decide within 20 ms, hold 2 env steps). `reprobe_step`: a re-probed β is accepted only when the probe's fit is within `RACE_FIT_GATE` = 0.05 (rejected records marked `accepted: False`). Monitor `action_repeat_probability` (the race's own persistence at resamples, held steps excluded) in `race_log`. 79 tests, golden unchanged; the hold smoke run is bit-identical to the screen's. Neuroscientist: the hold is action persistence (temporally correlated exploration variability), impl / abstraction with a supported class; the leaky race counter a Proposal; `action_repeat_probability` its monitor; recipe doc §2 rows added.
- Hold pilot launched (dev seeds; Acrobot hold 2 × four regimes with the band and β_c −1.5, 40 runs; LunarLander full length, θ 26 at 20 ms, base / sealed / learned, 9 runs, hours).

### Result: the decision hold fixes the race on Acrobot; LunarLander's θ rule (iteration 016 §21; user decision to keep fixing)

- Diagnosis (snn-expert, dev seeds): the race's Acrobot failure is temporal inconsistency of exploration (a fresh decision every 20 ms never pumps the pendulum; a uniform random policy terminates 0/100 episodes per step, 21/100 when each action is held 2 steps); the softmax's persistence is a filtered-noise sample read at high gain, not state knowledge. Every knob that sharpens the race (longer window at θ by the rule, more neurons, less noise) collapses. **Hold 2** (decide within 20 ms at θ = 16, hold the decision 40 ms): 0/10 floor seeds, and with sealed or learned columns level with CK's base (AUC −174 vs −174, +3 [−27, +25]; 2/10 solved vs 0/10; fewer spikes); the hold does nothing for the softmax. LunarLander: no temporal gap; θ = 26 at a 20-ms decision matches the backbone's AUC-1000 at half the spikes; the β re-probe is broken there (probe fit 0.24–0.64, β at the floor). Pre-set rule: n_hold = the smallest hold at which a uniform random policy terminates ≥ 20 % of episodes (Acrobot 2, others 1); re-probed β accepted only within the 0.05 fit gate. Code (`--freeze`, the gate) and the neuroscientist's review of the hold in progress; next pilots scripted (`pilot016_hold.sh`: Acrobot hold 2 × four regimes, LunarLander full length × three).

### Result: contrast-floor pilot; Acrobot remains the race's limit, LunarLander trails at 1000 episodes (iteration 016 §20)

- 72/72 dev runs. Acrobot with the band: no silent columns (minima 17–40 Hz), entropy held at 0.7 of max, yet 5–10/10 runs at the floor: the policy is near uniform, and uniform is −500; one surrogate run solved. Per the user's rule, Acrobot is reported as the race readout's limit on a sparse-reward floor (diagnosis in §20; Proposals not built). LunarLander at 1000 episodes: the band holds the race steady at −170 vs the softmax's −103 with a near-uniform policy; the re-probe alone is unstable. Open: a full-length LunarLander pilot or reporting it as trailing.

### Finished: study `race-cartpole` (iteration 016 §20); the floor pilot relaunched

- 120/120 runs, 5 min, none failed. Race + β re-probe on columns beats CK's softmax after Holm in every regime: sealed +45.2 AUC [+29.4, +62.9] (Holm 9e-5, 20/20 solved, T_s 194 vs 293, 12.9 vs 36.0 actor spikes to decision), implicit +29.2 (0.013), learned +25.0 (0.041); the race without columns +14.5 (n.s., one run unsolved); the normalised softmax non-inferior. Sealed vs no columns +27.8 (Holm 0.033). The first reported improvement over the backbone. Results in `../map-stdp-rcp/reports/race-cartpole/`, uncommitted on branch `race-cartpole-study`.
- The floor pilot's first launch merged commands (a trailing blank on lines without column flags is a line continuation for `xargs -L 1`); stopped by process group, fixed (`sed 's/ *$//'`), relaunched 18:17 EDT.

### Added: `--rate-floor`, the one-sided per-column rate floor (iteration 016 §20; neuroscientist review; user decision)

- Per ms, column k's actor-E rows receive +c_t (ρ_min − ρ_k)_+ with ρ_k the column's mean E rate and ρ_min = `rate_floor` × r₀ (pre-set 0.5); surrogate arms on the coincidence trace, score arms rank-one; requires `--rate-cost` (the same coefficient and effective step); with the mean term it forms a band (copy mean at r₀, each column ≥ r₀/2). Token `rf<frac>`; monitors `col_rate_min` (min over columns of the bin-averaged column rate) and `col_rate_min_ms` in `race_log` for every race run. Default off, nothing drawn; 76 tests, golden unchanged, the default-path fingerprint unchanged. Smoke (Acrobot seed 20, 100 episodes): the surrogate race with the band leaves the floor (T_f 78) with the column minimum at 19–25 Hz; under the score credit the floor never binds (columns ≥ 32 Hz while the realised entropy halves): there the perseveration is a small persistent contrast, not a deprived column.
- Floor pilot (72 runs: Acrobot β_c −1.5, band × {surrogate, surrogate + β re-probe, exact score ×3} on base and sealed columns; LunarLander 1000 episodes: β re-probe alone and the three band variants) queued behind `race-cartpole`.

### Started: study `race-cartpole` (iteration 016 §20)

- README pre-registered 2026-10-29 18:02 EDT (user approved); user committed `23cc0b9`; the stamp committed on branch `race-cartpole-study` (`93f1880`, worktree `../map-stdp-rcp`); 73/73 tests there; launched 18:08 EDT, 120 runs, 60 workers.

### Decided: CartPole's reported result is the race with the β re-probe (study `race-cartpole` declared); Acrobot and LunarLander continue (iteration 016 §20; user decision 2026-10-29)

- User: the six CartPole arms of the dev table (CK softmax, normalised softmax, race + β re-probe on no / implicit / learned / sealed columns) are the reported set, with the settings stated as selected on dev seeds. Declared as `race-cartpole` in `report.py` (seeds 7000–7019, 120 runs, primary AUC, families headline / readout / secondary; README drafted). LunarLander pilot: the race trails the softmax by 40–60 points at 1000 episodes with a near-uniform policy; the score credit slower still. Next: the one-sided per-column rate floor (contrast floor; neuroscientist review in progress) piloted with the re-probe and rate-term variants on Acrobot and LunarLander.

### Result: per-task grid pilot (iteration 016 §20)

- 130/130 dev runs. CartPole: the exact score at 3× the RMS-matched actor step is level with CK's base (AUC 359 vs 364, 8/10 solved; ×1: 235; ×6: 306, 10/10 solved): the CartPole setting for score arms. Acrobot with β_c = −1.5: the critic now survives in every race run (43–52 Hz), but the race still collapses into perseveration (4–8/10 runs at every θ ∈ {16, 24, 32}; entropy 0.4–0.6 of ln 3 at a pinned mean rate): the race needs a floor on the contrast between columns, not on the mean. LunarLander race pilot (1000 episodes, 3 seeds, surrogate + rate term and exact score ×3) launched.

### Added: `--beta-c` and `--eta-mult-a`; per-task grid pilot launched (iteration 016 §20; user go-ahead)

- `--beta-c` overrides the preset's critic bias (token `bc<v>`); `--eta-mult-a` multiplies the calibrated actor step under `--eta-cal` (token `em<m>`, applied before the rate term's effective step is computed). Defaults reproduce the presets; `test_beta_c_and_eta_mult_flags`. Both are per-task settings for the declared grids (user decision: per-task tuning on dev seeds, reported on fresh seeds).
- Grid pilot (dev seeds 20–29, main checkout, patch `grid_code.patch` in the scratchpad): Acrobot with β_c = −1.5 (the 1.5 V_rand rule from the critic diagnosis): `base_ck` (new reference), the surrogate race with the rate term and the exact-score race, each at θ ∈ {16, 24, 32}; CartPole: the exact-score race at actor-step multipliers {1, 3, 6}, base and sealed columns. 130 runs.

### Result: CartPole score re-pilot after the η rule fix; the exact score learns CartPole, slowly (iteration 016 §20)

- 80/80 dev runs. `--credit noise`: every arm near the ceiling at episode 400 (last-100 483–490), rates held, no collapse; AUC 187–235 vs 364 for CK's base and 380–403 for the stabilised surrogate race: about 2.5× slower, as the measured signal fraction predicts. `--credit hazard`: 5× slower again. Regimes equal. Acrobot critic death diagnosed (TD(0) bias from the critic's own spikes puts the floor's value below β_c = −1.2; the race's uniform initial policy never terminates an episode, the softmax's state-locked one does): recommended β_c = 1.5 V_rand (−1.5 on Acrobot; `base_ck` re-run as the reference), then a θ grid for the race's initial state-dependence. Nothing changed yet.

### Fixed: `--eta-cal` on CartPole calibrates against the authors' Adam steps (iteration 016 §20; user go-ahead)

- `Config.adam_step_c / adam_step_a` (default 0 = use `eta_c` / `eta_a`); CartPole's preset carries the LunarLander config's Adam steps (1.25e-4 / 6.25e-5). `calibrate_ck` divides the Adam step, not the preset's SGD rate, by the per-ms update RMS, so CartPole's `--eta-cal` gives η_c ≈ 0.02 mV instead of 0.8 (the §15 trap that killed the score pilot's CartPole arms in episode 1). No reported study used `--eta-cal` on CartPole; the other presets are unchanged (`test_eta_cal_cartpole_uses_adam_steps`). CartPole score re-pilot (80 runs, seeds 20–29, main checkout, uncommitted patch saved as `score_cp_code.patch` in the scratchpad) launched.

### Result: score pilot; both spike-score credits fail to learn on both tasks (iteration 016 §20)

- 160/160 dev runs (seeds 20–29). CartPole: every `--credit noise` / `hazard` arm at or below the random policy (AUC 9–16 vs CK's base 364), realised policy entropy falling; Acrobot: every arm at the floor. Rates held, clip 0, no NaN. Diagnosis (snn-expert, same day): not the credit. CartPole: `--eta-cal` with CartPole's SGD-rate preset (the §15 trap) gave a 68–280× step, the critic died in episode 1 and the actor random-walked; with the other tasks' η rule the exact score **solves CartPole** (last-100 496–497, slower than the surrogate). Acrobot: the critic dies at the floor in every race arm, the surrogate's included (β_c below the floor policy's value; an absorbing silent state), upstream of the credit. The live-path finite differences of the score eligibility match (0.96–1.24). Nothing changed; the η rule fix and the critic fix are the next steps.

### Result: surrogate re-pilot of the race readout; no variant passes the gates on both tasks (iteration 016 §18)

- 240/240 dev runs (seeds 20–29). CartPole: every stabilised race arm (rate term, β re-probe, both) solves 10/10 and is +23 to +41 AUC above CK's base, nine of twelve CIs above zero, with fewer actor spikes to the decision (13–27 vs 36): the first race result above the published backbone. Acrobot: the rate term holds the mean rate and collapses 4–9/10 runs (silent columns under a pinned mean, the neuroscientist's issue 3); the β re-probe fails the rate gate (97–110 Hz). The first pilot's structural signal on CartPole (+57) does not survive the fixes (−3 / +18 / −2): it was carried by the drift. Nothing changed; the score pilot (§20) is next.

### Added: `--credit {hazard,noise}`, the per-neuron spike score as the race readout's credit (iteration 016 §20; derivation §8.1 Proposal, Eq. 13, Appendix A H1–H3)

- `mapstdp/lif.py`: the kernel exports the pre-noise membrane and the noise draw per ms behind `score=True` (no extra draw; default path bit-identical); `post_scores` (hazard form, clipped at h = 1e-3; noise form, the exact observed-membrane score); self-check (h). `mapstdp/tdstdp.py`: `simulate_nlif(score=)`; self-check h4, the eligibility as the learner computes it vs finite differences (ratios 0.997–1.009).
- `experiments/tdstdp.py`: `Config.credit`, CLI `--credit`, tokens `sch`/`scn`; two learner calls per block under a score credit (critic rows unchanged; actor E rows with Y as the post factor and **x(t−1)** as the pre factor, truncated at the bound, no entropy term); the rank-one rate term with its effective step held (c_t η_a_ref/η_a); R's `hs` from the column's summed score with G = 1/τ_eff on the task path; monitors `score_abs_mean`, `score_clip_frac`, `race_entropy` (`race_log`). Decision: the pre factor is x(t−1), not the 20-ms pre trace, which over-counts the gradient 1.7–1.9× on the toy. 71 tests, golden unchanged, default path fingerprint-identical to `4a74319`.
- Docs: derivation §8.1 Proposal paragraph with Eq. 13 and Appendix A rows H1–H3; references Fiete & Seung 2006, Pfister et al. 2006; recipe doc rows to impl; 016 §20 implementation record.

### Finished: study `hybrid-columns-3-ck` (iteration 016 §19)

- 60/60 runs, 2.9 h, none failed. Against CK's own readout at the matched start (same seeds as `hybrid-columns-3`): LunarLander columns **worse** by 22–25 last-100 (Holm ≤ 1e-4) and the normalised base worse by 33 (CK's base reaches 62.3, the normalised 30.1): §17's LunarLander gain was a partial recovery of the normalisation's own loss. CartPole and Acrobot: the normalised base level with CK's; regimes non-inferior or inconclusive, never significantly worse at n = 20; Acrobot last-100 (secondary) better in every normalised arm (+9 to +18). Closes the softmax branch of the hybrid. Results in `../map-stdp-hc3ck/reports/hybrid-columns-3-ck/`, uncommitted on branch `hybrid-columns-3-ck-study`.
- Surrogate re-pilot (§18) launched 15:20 EDT from a detached worktree `../map-stdp-rc` at `4a74319` (the committed code, isolated from the score-credit edits in the main checkout), 60 workers.

### Added: `docs/columns-recipe.md`, the insertion recipe (user decision 2026-10-29), with a pointer in SPEC §11

- Current-state recipe for converting a feed-forward spiking actor into recurrent action columns on the TD-STDP backbone: columns and loop-gain calibration, the race readout, the spike-score credit (spec), the rate term, structure control, what is not covered, the studies. Status tags as `model.md`. Updated as the score credit lands and the Stage B pilot and study report.

### Decided: Stage B reframed as the insertion recipe; the spike score replaces the rate surrogate as the race's credit (iteration 016 §20; user decision, two expert reviews)

- Lesson: the hybrid works when the walk makes the decision (v4 vs §17/§19: the softmax hides structure). Stage B = columns + race readout + **per-neuron spike score as the credit** (`--credit noise` headline, the exact observed-membrane score, within 2 % of finite differences; `--credit hazard` secondary, the escape-noise score at the realised membrane, biased 0.2–0.3× for this neuron) + CK's rate term (rank-one, no per-synapse state, effective step held) + structure control; no entropy bonus; realised race entropy gated; truncation at the bound; E actor posts only; R credited at the module level (per-synapse version a Proposal). Recipe doc `docs/columns-recipe.md` to follow. Implementation by the snn-expert in progress.
- Surrogate re-pilot (§18) unchanged, runs as the stopgap comparison.

### Finished: study `hybrid-columns-4` (iteration 016 §19)

- 200/200 runs, 14 min, none failed. Against the normalised base all three regimes are non-inferior at n = 40 (CIs −22/−14/−17 to +4/+10/+10), settling §17's inconclusives. Against the published backbone (`base_ck`) implicit columns are **worse** (−14.7 AUC [−25.7, −3.9], Holm 0.041; the one failed expectation), learned and sealed columns non-inferior, the normalised base non-inferior (−5.7 [−18.6, +6.6]). Regimes do not differ. Post-solve actor-rate decay (→ 4–6 Hz) in every normalised arm, not in `base_ck`. Results in `../map-stdp-hc4/reports/hybrid-columns-4/`, uncommitted on branch `hybrid-columns-4-study`. `hybrid-columns-3-ck` launched 12:20 EDT.

### Started: companion studies `hybrid-columns-4` and `hybrid-columns-3-ck` (iteration 016 §19); race re-pilot pre-registered (§18)

- READMEs pre-registered 2026-10-29 12:03 EDT (user approved); user committed `0025470`. Worktrees `../map-stdp-hc4` (branch `hybrid-columns-4-study`) and `../map-stdp-hc3ck` (`hybrid-columns-3-ck-study`) at that commit, 63/63 tests in each. `hybrid-columns-4` launched 12:06 EDT, 200 runs, 50 workers; `hybrid-columns-3-ck` (60 runs, 60 workers) launches automatically when it finishes (detached launcher), about 7.5 h.
- Re-pilot of the race readout pre-registered in §18 (user's selection: three variants per regime, `--rate-cost 5e-4`, `--race-reprobe 50`, both; 240 dev runs, seeds 20–29; gates as the first pilot plus the surrogate fit at every informative re-probe; energy read as `actor_spikes_to_decision`). Waits for the user's go-ahead to run on the idle cores during `hybrid-columns-3-ck`'s LunarLander phase.
- §18's addenda (neuroscientist's review, sizing, fix (a)) moved to their place before §19.

### Added: `--race-reprobe N`, the surrogate gain β re-estimated on a schedule (iteration 016 §18 fix a; user decision, neuroscientist's recommendation)

- `Config.race_reprobe` (default 0, token `rp<N>`): with `--readout race`, once every N pooled episodes `reprobe_race` re-runs `calibrate_race`'s 1-s contrast probe at the run's current W (and R with recurrence), its calibrated s_in and alphas, on the probe's own stream `(seed, CAL_SEED, episode)`; the new β replaces the surrogate's from the next window on. **Pre-set N = 50**, one re-probe per log bin on CartPole and Acrobot, fixed before the re-pilot. The run's `rng` and `nrng` draw nothing for the probe (same draw sequence with and without it); the probe is frozen (W unchanged).
- `CKRunner.beta` holds the gain (`surrogate`, `race_factor` read it, not `cfg.race_beta`); `race_cal_history` (records with `race_cal`'s keys plus `episode`) and the current β are in the checkpoint and the result (`race_cal_history`, `race_beta_final`); the log prints `reprobe ep N beta X fit Y` (fit = max |win0 − win0_surrogate|). `calibrate_race` shares the probe body (`_race_probe`) with `reprobe_race`; its init call is byte-identical (same runner, same stream; `test_race_reprobe` checks the hand-off, and the smoke run reproduces the pilot's init β 7.9403 on Acrobot seed 20). `CKRunner(rng_seed=...)` accepts a tuple of ints (an int is unchanged). Default off: nothing drawn, golden unchanged. Test `test_race_reprobe`.

### Added: `--rate-cost`, CK's target-rate term in `CKRunner` (iteration 016 §18 fix b; neuroscientist review, snn-expert sizing)

- Per ms, on the copy's actor E rows, `kap -= c_t (mean actor-E rate − ρ̂)`, ρ̂ = `rate_target` by default (`rho_hat` overrides); the `c_t = 0` assert is lifted, token `rc<c_t>`. Default off: the broadcast `kap` is unchanged and nothing is drawn (golden bit-identical). Test `test_rate_cost_term`.
- Pre-set value for the race re-pilot, fixed before any run: **c_t = 5e-4**, the smallest power-of-ten multiple of the authors' `targ_firing_r` (5e-6, `config_cp_lfr.ini`; no unit conversion, like `c_e`) at which the restoring term exceeds the pilot's observed drift (+8 Hz / −11 Hz per 100 episodes) at the gate edge on both tasks; implied relaxation 19–33 episodes.
- Neuroscientist (review-only): a coincidence-gated rate homeostasis on the population mean (a rapid compensatory process, Zenke et al. 2013), not synaptic scaling; crossbar-native; stabilises the common mode only, cannot revive a silent column, and makes `actor_spikes_per_decision` a constant across arms (only `actor_spikes_to_decision` remains an energy measure). Recommends fix (a), a rescheduled surrogate gain β, with (b) alongside, not instead; the user decides the re-pilot's arms.

### Added: companion studies `hybrid-columns-3-ck` and `hybrid-columns-4` declared and pre-registered (iteration 016 §19; user decision)

- `hybrid-columns-3-ck`: the CK-faithful base (softmax on the unnormalised scores, matched start) on the three tasks, seeds 5000–5019 as `hybrid-columns-3` (a stated exception: same initial W and first environment seed per contrast; the arm never ran on them). Pairs across studies: each regime of `hybrid-columns-3` vs `base_ck` (Holm per task) and its base vs `base_ck`. `report.summarize` now loads a pair member written `<study>/<group>` from that study's `runs/`, as it did for a cross-study reference; `test_summarize_cross_study_pairs`. Existing summaries byte-identical.
- `hybrid-columns-4`: CartPole at n = 40 (seeds 6000–6039, fresh), `hybrid-columns-3`'s four arms plus `base_ck`; families headline / `ck` / secondary. The power the §17 inconclusives lacked.
- Both READMEs are drafts awaiting the user's approval; launch after the user commits. Noted: `hybrid-columns-3`'s `pairs.csv` Holm is over the nine headline contrasts, not per task as its README says; the reading is unchanged (per-task LunarLander 0.051 ×3).

### Merged: the four study result sets into `columns-tdstdp` (`cf72b9e`)

- `hybrid-columns`, `hybrid-columns-2`, `hybrid-columns-3` and `emergent-v4`: each study's `manifest.json`, `runs/`, `summary.*`, `pairs.csv` (and `emergent-v4`'s models and structure comparison), committed on the worktree branches and merged; `summarize` reproduces every committed summary from `runs/`.

### Result: Stage B pilot, the race readout fails its gate on Acrobot (iteration 016 §18)

- 120/120 dev runs. `base_race` is worse than CK's base on Acrobot (−84 AUC [−277, −4]; actor rate climbs 46 → 82 Hz, 2/10 collapsed) and inconclusive on CartPole (−11 [−56, +10]; rate falls 41 → 10 Hz). Every race column arm is worse on Acrobot. Hypothesis: a common-mode rate drift of task-dependent sign from the mismatch between the race's realised choices and the surrogate, through the 1/r_j factor and the TD error's sign. The structural contrast opened on CartPole (sealed − implicit +57 AUC, p 0.04, n = 10), the first regime contrast to move. Nothing changed; reported to the user with three pre-set candidate fixes (β on a schedule, CK's rate-cost term, the surrogate's sample as the choice).

### Changed: the race tallies E counts; added `actor_spikes_to_decision`; Stage B pilot pre-registered (iteration 016 §18)

- User decision 2026-10-29 (LunarLander θ): the race counts each column's E posts only (`CKRunner.Mcnt`; the I posts shape the dynamics but are not tallied), so θ = round(n_ae r₀ T) means the same on every task; CartPole and Acrobot (no I posts) are unchanged. A noted deviation from the §14.1 net-count tally. Test: an I-post raster moves no count.
- `actor_spikes_to_decision`: actor spikes up to the bound ms (or the whole window at a deadline decision) per copy-decision, the energy a loop stopped at the bound would spend; checkpoints carry `ebnd`. Race only; the softmax path and the golden test are unchanged.
- Pilot (development, dev seeds 20–29): CartPole and Acrobot × `base_ck` (CK's softmax), `base_nl`, `base_race` and the three column regimes under the race; co-primary AUC vs `base_ck` and spikes to decision; stability gates only; `cols_sealed_race` vs `cols_lam0_race` is the structural question.

### Finished: study `hybrid-columns-3` (iteration 016 §17)

- 240/240 runs, 10.3 h, none failed. No column arm is worse than base on any task; none improves after Holm. Non-inferior (Δ = 25): `cols_lam0` on CartPole, all three regimes on LunarLander (shifts +7 to +11, CIs above zero before Holm), `cols_sealed` on Acrobot; the other four contrasts inconclusive. Regimes never differ. Acrobot's deficit of `hybrid-columns-2` (−35, −31) is gone under the normalised scores (−9, −2), as §13 predicted.
- Cross-study, not a test: the normalised readout's LunarLander base plateaus after episode 3000 (last-100 30 vs 58 for 015's CK-faithful base, same AUC-3000); the column arms recover part of it. Recorded as an open gain-control question for Stage B.
- The study's `rec.RKc` (raw pairs) is chance-dominated and uninformative, as expected; the results of all three column studies are still untracked in their worktrees.

## 2026-10-28

### Added: `--readout race`, Stage B of iteration 016 §14.1 (§16; user decision; pilot held)

- The race readout on the columns' net counts (Eq. 1h; first to θ, leader at the deadline, ties by one draw), θ = round(n_ae r₀ T_freeze); the population surrogate p̃ = softmax(β log r) with its exact gradient as the learning signal (finite-difference test; no silencing drift); β by a contrast (psychometric) probe at init, `calibrate_race`, after the entropy-matching rule proved ill-conditioned; `actor_spikes_per_decision` in every result. Default off; softmax path byte-identical; golden unchanged. Implemented by the snn-expert; six tests.

### Fixed: the `RKc` monitor records the causal counter per class; end-snapshot overwrite (iteration 016 §15)

- `class_table` adds `cshare` / `cn`: the module-level counter (Eq. 8's causal transitions) split by executed action; the raw pair split (`share`) was chance-dominated (0.506 cross-column in every arm, sealed included). The end-of-run snapshot no longer overwrites the `SNAP_EVERY` snapshot of the same episode (which lost the last phase). The running study `hybrid-columns-3` carries the old monitor; its `RKc` is raw-only and lacks the last phase (recorded in §15). Interim CartPole look: every arm 20/20 solved, shifts vs base −4 to −13, none significant.

### Started: study `hybrid-columns-3` (iteration 016 §15)

- README pre-registered 2026-10-28 18:22 EDT (user approved; Δ = 25 confirmed). Launched from worktree `../map-stdp-hc3` at `4bdbe45`: CartPole, LunarLander, Acrobot × 4 arms × seeds 5000–5019, 64 workers.

### Added: the stimulus-conditioned walk monitor, `rec.RKc` (iteration 016 §15; user decision)

- `HybridRec.window(…, cls)` splits the window's pair counts by the copies' executed action; `class_table()` reduces them per snapshot to module-level shares per class; results carry `rec.RKc`, checkpoints `rec_kc` / `snaps_kc`. The totals are unchanged (integer counts split per head), no update or draw is affected. It measures whether the cross-column walk depends on the state, the prerequisite for a stimulus-conditioned map term. Test `test_class_pair_counts`.

### Fixed: `probe_entropy` NaN on CartPole; CartPole keeps CK's preset η (iteration 016 §15)

- `probe_entropy` pooled over live copies of all ticks instead of a per-tick mean, which was NaN on ticks with no live copy (CartPole's one copy in reset) and drove `match_entropy` to its floor (f = 0.125: the column arms' policy stayed uniform). Test added. Acrobot and LunarLander were unaffected.
- `hybrid-columns-3`: CartPole without `--eta-cal` (its preset η are SGD rates; calibrating them gave η_a 8.7 mV and a silent actor), as in `hybrid-columns`.

### Declared: study `hybrid-columns-3` (iteration 016 §15; user decision on the sequence)

- Three tasks × the three regimes vs base, seeds 5000–5019, every arm with `--eta-cal --rate-match-rec --ent-match-rec --norm-logits`. README draft written; CartPole and LunarLander stability pilots under the normalised scores run before pre-registration. Test `test_hybrid_columns_3_declared`.
- Sequence (user): this study, then the race readout (§14.1 Stage B), then a stimulus-conditioned map term informed by the results. The conditioned map term is feasible and cheap (S × K × K), but the softmax readout cannot see R's structure, so it waits for a readout where links act at the decision.

### Pilot: `--norm-logits` removes the Acrobot column deficit (iteration 016 §13)

- Acrobot, dev seeds 0–9, every arm normalised: rates hold (0.04–0.05 per ms through episode 500), no column run collapsed or slow (0/30), and the column arms are level with base (HL +13, +4, −2, p ≥ 0.39). All stability gates passed. The §12 deficit was the silencing drift acting through the loop. Base under normalisation is −186 vs −163 unnormalised (its late re-flattening was the drift).

### Added: `--norm-logits`, divisive normalisation of the action scores (iteration 016 §13; user decision)

- Policy scores α_a r_k r₀ / max(r̄, ε): r̄ the pooled actor rate, r₀ = `rate_target`, ε = α_a r₀ / (τ_n n_ae) (the neuroscientist's rule: one spike moves a score by at most one nat). The actor eligibility and the entropy bonus use the exact gradient through the normalisation (`CKRunner.scores`, `norm_factors`; two broadcast scalars per copy), which cancels the entropy bonus's silencing drift. Default off; the default path's expressions are unchanged; token `nl`. Test `test_norm_logits` (scale invariance, CK's scores at r̄ = r₀, the floor, finite-difference gradients).
- Neuroscientist check: gradient and locality confirmed; pooled-over-options normalisation of action value has precedent in LIP (Louie, Grattan & Glimcher 2011). The pilot (Acrobot, 4 arms × dev seeds 0–9) is a stability pilot; this is the third gain-control choice made after seeing dev returns, recorded as such.

### Pilot: the fixed Acrobot preset puts base at floor (iteration 016 §13)

- **Withdrawn (user decision):** `_NEW` is back to `n_ai` 0. Gain control is instead divisive normalisation of the action scores (`--norm-logits`, Proposal in 016 §13), chosen over the authors' α_a pairing and over dropping the entropy bonus.

- With opponent I posts the actor rate holds (the drift is gone) but the policy sharpens early: base collapsed in 3/10 dev runs (AUC median −264 vs −163 E-only), `cols_on` in 7/10. At init the opponent readout's entropy is 0.874 vs 0.969, and its logit learning gain is about doubled. Base at floor: reported to the user before any change.

### Changed: expert review, the entropy-bonus confound and the Acrobot preset (iteration 016 §13)

- Divergent-then-convergent review (neuroscientist, snn-expert) of how to use columns and the map term in the TD-STDP backbone. Shortlist: actor gain control; columns as a competitive readout (inhibition plus a race) giving the map term a function; latent state modules with latent→action exempt from the map term.
- **Finding (snn-expert, development):** the Acrobot actor-rate decline is the entropy bonus on an E-only actor (Σ_k g_k r_k = −α Var(r) ≤ 0), a silencing drift the loop roughly doubles. The `hybrid-columns-2` contrasts are confounded by it.
- **Preset (user decision):** `_NEW` gains `n_ai` 20 = `n_ae`, the authors' pairing of c_e with opponent I posts (`config_ll.ini`). Changed after the study for a mechanistic reason independent of return; `hybrid-columns-2` stands as run. `test_preset_closed_loop` checks the pairing.
- **Softened:** 016 §11 (the amplified logit spread is mostly noise), §12 (the loop's cost is confounded by the drift; P-seq is not a clean sequencing test), 015 §8 (the early LunarLander speed-up was mostly a starting-gain effect; sealed R and late `cols_on` are fixed random amplifiers, not learned structure).
- Pilot of the fixed preset: Acrobot, 4 arms × dev seeds 0–9, both matchings, stability gates only.

### Results: study `hybrid-columns-2` (iteration 016 §12)

- Acrobot, 20 seeds per arm, AUC vs base:
  - `cols_lam0` −35.2 [−59.8, −12.3] and `cols_sealed` −31.0 [−56.9, −9.8]: both **worse** (Holm p 0.014);
  - `cols_on` −16.6 [−43.1, +4.6]: inconclusive.
- Secondary contrasts are null. P-seq is level (−3.1), so the cost comes from the within-column loop, not from cross-column links.

### Started: study `hybrid-columns-2` (iteration 016 §12)

- README pre-registered 2026-10-28 15:25 EDT (user approved; Δ = 25 confirmed). Launched from worktree `../map-stdp-hc2` at `8d0a9e4`: Acrobot, 4 arms × seeds 4000–4019.

### Changed: `hybrid-columns-2` uses both matchings (user decision)

- Every group of `STUDIES["hybrid-columns-2"]` now carries `--rate-match-rec --ent-match-rec` (no-ops for base). The pre-registration draft `reports/hybrid-columns-2/README.md` states the matched starts, the updated pilot expectations, and a reworded P-seq reading. Δ = 25 still awaits the user's confirmation.

### Results: entropy-matched pilots (iteration 016 §11)

- **Acrobot:** entropy matching removed the collapses (0/30 vs 3/30 column runs), but the column arms stay below base (HL −32, −47, −51). It is not perseveration; the cause is open.
- **LunarLander:** with rate and entropy matched, the early AUC gain falls to +3 to +6 (n = 6, not significant). Most of 015's early speed-up was a gain effect.

### Added: per-arm entropy matching, `--ent-match-rec` (iteration 016 §11)

- At a matched mean rate the column arms' initial policy is sharper: the within-column loop amplifies rate differences between columns (Acrobot H₀ 0.92 vs 0.98, logit spread +25 to +37%).
- `--ent-match-rec` scales α_a per arm (`probe_entropy`, `match_entropy` in `experiments/tdstdp.py`) so the arm's initial policy entropy equals the base run's. Default off; token `em`; golden path unchanged. Test `test_ent_match_rec`.
- Pilot: Acrobot column arms, seeds 0–9; LunarLander `cols_lam0`, `cols_sealed`, seeds 10–15, 3000 episodes.

### Results: rate-matched pilots (iteration 016 §10)

- **Acrobot** (seeds 0–9): rate matching equalised the actor rate, but each column arm still had 1/10 collapsed runs (base 0/10), lower entropy at episode 50 and AUC below base (HL −73, −61, −22 for `cols_lam0`, `cols_on`, `cols_sealed`). The starting rate is not the cause of the collapse.
- **LunarLander** (seeds 10–15, 3000 episodes): rate matching roughly halves the early AUC gain of the column arms (+11 → +6, +13 → +5); with n = 6 the remainder is not significant. Part of 015's early speed-up is a starting-rate effect.

### Added: per-arm rate matching, `--rate-match-rec` (iteration 016 §10)

- **Diagnosis:** column arms start at a higher post rate (46 vs 41 Hz on Acrobot), because s_in is calibrated without recurrence. That gives a sharper initial policy and occasional collapse to a deterministic corner, where (A − p) and the entropy gradient vanish.
- **The flag:** `Config.rate_match_rec` makes `probe_rate` run with the arm's R at init. Default off; existing runs and the golden test are unchanged.
- **Test:** `test_tasks016::test_rate_match_rec`.
- **Pilots running:**
  - Acrobot, rate-matched, dev seeds 0–9;
  - LunarLander, matched vs unmatched, dev seeds 10–15, 3000 episodes. This tests whether 015's early speed-up is a gain effect.

### Decided: iteration 016 pilot v3, Acrobot only (§9 of the record)

- **Acrobot passes the gates.** Base learns in 3 of 3 seeds; column arms learn in 7 of 9.
- **Taxi fails the floor gate:** no deliveries in base. Decision (user): run the study on Acrobot alone; a hierarchical task is rethought separately.
- `STUDIES["hybrid-columns-2"]` is now Acrobot × 4 arms, seeds 4000–4019.
- Draft pre-registration: `reports/hybrid-columns-2/README.md`, not launched.

### Decided: iteration 016 pilot v2, entropy bonus for the new presets (§8 of the record)

- **Finding:** after the β_c fix, Acrobot and Taxi policies collapse before exploring.
  - Acrobot: 3 of 12 runs learn.
  - Taxi: no run delivers a passenger.
- **Decision (user):** the new presets take LunarLander's entropy bonus, c_e = 1e-4. This is the authors' exploration setting.
- **Next:** pilot v3 on the same dev seeds. If base still collapses, report to the user rather than trying further fixes.

### Fixed and decided: iteration 016 pilot v1 (`docs/iterations/016-columns-more-tasks.md` §7)

- **Taxi:** `ck_encoder` returned a numpy int for `n_in`, so writing the result failed. Fixed.
- **Acrobot:** at floor in every arm, because CartPole's critic floor β_c = −0.2 cannot represent Acrobot's negative values.
- **Decision (user):** new presets use β_c = min(−0.2, 1.2·V_rand), where V_rand is the random-policy value r_scale·E[r]·τ_d/20 ms.
  - Acrobot −1.2, Taxi −0.82.
  - Consistent with the authors' LunarLander −2.0 (V_rand −2.31).
  - Tested in `test_tasks016`.
- **Full suite:** 50 passed before the β_c change; `test_tasks016` 8 passed after it.

### Results (reported study `hybrid-columns`, iteration 015 §8)

- All 200 runs completed (0 failed) on branch `hybrid-columns-study` (git `0d2d851`).
- **Headline** (Δ = 25):
  - **LunarLander:** all three regimes are non-inferior to base (last-100 shifts −6.9, −2.0, −0.5). No LunarLander run solved.
  - **CartPole:** sealed is non-inferior (AUC +0.4 [−18.1, +18.8]). `cols_lam0` and `cols_on` are inconclusive: their CIs reach −27.6 and −32.0, and neither is significantly worse.
  - No regime improves the primary metric.
- **Secondary:**
  - every column arm learns LunarLander faster over the first 3000 episodes (AUC +8.8 to +10.7, Holm ≤ 0.0014);
  - CartPole solved 20/20 with columns vs 16/20 for base.

## 2026-10-27

### Added: iteration 016, Acrobot and Taxi in the TD-STDP backbone (code only; nothing run)

- **Decision (user):** the tasks are Acrobot-v1 and Taxi-v4. Taxi replaces the discretised Pendulum: CartPole, Acrobot and Pendulum are all pendulums.
  - Acrobot's actions are exclusive but must be sequenced.
  - Taxi is hierarchical, with two groups of actions.
- **Purpose:** test when each regime of modularity (implicit, learned, enforced) helps. Record: `docs/iterations/016-columns-more-tasks.md`.
- **`experiments/tdstdp.py`:**
  - `PRESETS["Acrobot"]` and `PRESETS["Taxi"]` by pre-set rules, never tuned on return. The rules (§4 of the record):
    - CartPole's populations and α_a;
    - LunarLander's Adam steps for `--eta-cal`;
    - `r_scale` = 0.02 / RMS of the random-policy reward: Acrobot 0.02, Taxi 0.0035;
    - Acrobot uses LunarLander's encoder;
    - Taxi uses a one-hot input with `in_rep` 1.
  - Config fields:
    - `onehot` (one-hot input for a discrete observation in `ck_encoder`);
    - `r_ck_hz`: `calibrate_low`'s k from CartPole's BindsNET rate, `R_CK_CARTPOLE` = 140 Hz.
  - `TD_TASKS` adds Taxi for `CKRunner` only. `tasks.TASKS` is unchanged, so M3 is unaffected.
  - `SOLVE`, `M_THR` and `SNAP_EVERY` entries for the new tasks; `--summary` handles tasks without CK numbers.
- **`experiments/report.py`:**
  - study `hybrid-columns-2` (Acrobot and Taxi × base / `cols_lam0` / `cols_on` / `cols_sealed`, seeds 4000–4019). Declared, not pre-registered.
  - The primary pair metric is AUC on every task except LunarLander. Captions of existing studies are unchanged.
- **Tests:** `tests/test_tasks016.py`.

### Results (exploratory, iteration 014 §12): description length vs SOP energy proxy

- Post hoc, on the saved `emergent-v4` models with W frozen; not pre-registered.
- L_obs correlates with effective SOPs (spikes × synapses above the weight floor) on all three spiking engines:
  - across the final networks, Spearman ρ +0.61 (branching), +0.95 (QIF), +0.91 (LIF);
  - over training, the median per-run ρ is +0.54 to +0.89.
- L_obs does not correlate with raw spike counts, except on QIF.

### Launched: reported study `hybrid-columns` (iteration 015 §7)

- **Pre-registered:** 2026-10-27 20:02 EDT.
- **Code:** git `0d2d851` with an empty patch, worktree `../map-stdp-hc` on branch `hybrid-columns-study`.
- **Runs:** 200, on 64 workers.

### Results (reported study `emergent-v4`, iteration 014 §11)

- All 240 runs completed (0 failed) on branch `emergent-v4-study` (git `086996f`, empty patch). Summary, pairs and structure were written by `experiments.report` and `experiments.structure` in the worktree. They reach `main` only by merging that branch.
- **P1** (AUC, on vs off): supported on branching, QIF and LIF (Holm p 2.7e-7); not supported on flow (+21, p 0.17).
- **P2** (L_obs, on vs off): supported on all 4 engines.
- **S** (on vs sealed): AUC equal on flow, branching and LIF. Sealed beats on for QIF (196 vs 168, Holm 0.0023). The map term reaches a lower L_obs than sealing on all three spiking engines.

### Results (iteration 015, pilot C: LunarLander columns and latents)

- 12 runs, `pilot015/C/`: χ_rec 0.3, seeds 0–2, 3000 episodes, `--eta-cal --rate-target 40`. Recorded in `docs/iterations/015-hybrid-columns.md` §6. Development results.
- **Last-100 return** (mean of 3 seeds):
  - cols λ = 0: 52 (48, 55, 54);
  - base: 26 (pilot B);
  - cols r = 1: 24; cols r = 3: 26; lat r = 1: 31.
  - No run solves.
- **Structure.** With λ = 0, the task term spreads the columns apart (within share 0.77 → 0.46) and builds a hub: the main-engine column receives 73% of the R weight. The map term seals the columns by about episode 200 and removes this pathway. On CartPole, by contrast, the task term seals the columns itself (§5).
- **Gates.** Every run passes the stability gates (max Re λ ≤ 0.25, τ_loop ≤ 27 ms). The latent arm passes its pre-set go criteria, so it stays in the study. However, its latent→action share falls from 0.25 to 0.004: the map term nearly disconnects the latents and the barrier does not hold them.
- **L_obs is uninformative on LunarLander.** The causal share is 0.1–0.2%, and L_obs changes by less than 0.04 bits in every arm.

### Decided (iteration 015: `hybrid-columns` design, user 2026-10-27)

- **Non-inferiority margin Δ = 25:** 25 points of last-100 return on LunarLander, 25 points of AUC on CartPole.
- **Headline:** each regime of modularity against `base`, with both tests ("improves" and "keeps performance"):
  - implicit `cols_lam0`;
  - learned `cols_on`;
  - enforced `cols_sealed`.
  
  Holm applies over these 6 contrasts. The secondary family (`cols_on` vs `cols_lam0`, `cols_on` vs `cols_sealed`, `lat_on` vs `cols_on`) has its own Holm correction.
- **New arm `cols_sealed`.** It adds `--rec-seal` (`Config.rec_seal`, token `seal`, `HybridRec(seal=True)`): the same random draws for R, with the cross-module synapses removed. Default paths are unchanged. New test `test_rec_seal_drops_cross_module_synapses`. The study now has 10 groups and 200 runs.
- **Fixed: `calibrate_rec` with sealed R.**
  - In a sealed network every target of j lies in m(j), so the one-hop credit h_m(i) − h̄_j is exactly 0.
  - The smoke run's calibration therefore divided by rounding noise (RMS(u) 6e-19) and set η_rec ≈ 4e14.
  - `calibrate_rec` now measures on the unsealed draws, so every arm of a seed shares η_rec and λ_rec, as it already shares s_in and the low-rate scaling. The test checks the shared rates.
  - In `cols_sealed`, R therefore has no task learning, by construction: only homeostasis acts on it.

### Added (iteration 015: the `hybrid-columns` study, draft)

- `STUDIES["hybrid-columns"]` (`experiments/report.py`): 2 tasks × {base, cols_lam0, cols_on, lat_on}, nlif at 40 Hz, χ_rec 0.3, r = 1, seeds 3000–3019, with 8 pairs.
- `reports/hybrid-columns/README.md`: the draft pre-registration. The non-inferiority margin is left for the user to fix. Not launched.
- **Pairs for tdstdp-runner studies.** `summarize` writes `pairs.csv` and a "Pairs" section only when the study declares pairs, so existing tdstdp summaries stay byte-identical.
  - Metrics (`TD_PAIR_METRICS`):
    - primary = CartPole AUC / LunarLander last-100 return;
    - AUC over the first `episodes` episodes;
    - last-100 return.
  - `m3_pairs` takes the metrics and adds an optional Hodges–Lehmann shift with its 95% CI (`hodges_lehmann`, Moses interval).
  - New test `test_summarize_tdstdp_pairs`.
- **`experiments/tdstdp.py`:** the result's `rec.RK` holds R's module weight table (shares of total R) at every snapshot, so structure is reported without saving R. The latent checkpoint test checks it.

### Decided (iteration 015)

- New record `docs/iterations/015-hybrid-columns.md`, covering steps 1–3 and the pilots.
- **User decision:** the low-rate nlif regime (`--rate-target 40`) is used for both tasks and every arm.
  - It replaces the CK-matched regime, where LunarLander's recurrent posts lock at 500 Hz with gain ≈ 0.
  - The CartPole baseline is slower in this regime (4 of 5 solved by 400 episodes, T_s 252–370, AUC 322–399, against 154–191 and 393–453 CK-matched). That floor is reported, not tuned away.
- Pilot B launched: CartPole columns grid and the LunarLander low-rate baseline.

### Added (iteration 015, step 3: the low-rate nlif regime, `--rate-target R`; the default stays CK-matched)

- **`mapstdp/lif.py`:**
  - `DT1R` is the dt = 1 ms operating point with the default 2 ms refractory period: μ_bg 0.5778, χ 0.01575 (`calibrate`), rate ceiling 333 Hz.
  - `markov_rate` gains the refractory chain, so `gain_table` and `gain_markov` hold for τ_ref > 0. Its table key includes nref, and the nref = 0 values are unchanged.
  - Self-checks (f) and (g) add DT1R. Markov against simulation: 45.8 vs 45.9 at μ = 1.0, 41.0 vs 41.2 at μ = 1.5, 29.6 vs 29.7 at μ = 3.
- **R = 40 Hz** is the default proposal, set from the gain curve with τ_ref 2 ms:
  - G ≈ 41 Hz/θ at 40 Hz (μ ≈ 1.48), near the curve's peak of about 46 at 15–20 Hz;
  - 12% of the ceiling;
  - 0.8 spikes per neuron per 20 ms window.
- **`experiments/tdstdp.py`:**
  - `Config.rate_target` (Hz, 0 = CK-matched), `--rate-target` and the token `rt<R>`.
  - `calibrate_low` is a pre-set rule, run without recurrence:
    - τ_ref 2 ms;
    - s_in bisected to R by `calibrate_s_in`, which now also returns BindsNET's rate r_CK;
    - k = r_CK / R, with α_a and α_c multiplied by k and β_c kept;
    - without `--eta-cal` or Adam, η_c and η_a are multiplied by RMS(g_CK) / RMS(g_low). Both are the per-ms update RMS at init with W frozen on the CAL stream, g_CK from BindsNET at CK's α and g_low from low-rate nlif at the scaled α.
  - It runs before `--eta-cal` and `calibrate_rec`, so every arm of a seed shares it. The result's `low_rate` and the checkpoint carry the record, and a resume reuses it. τ_n stays at 20 ms.
- **Default path unchanged:**
  - `calibrate_s_in` still gives pilot A's s_in for CartPole seed 0 (0.94140625).
  - `pytest -q tests`: 40 passed (2 new tests), golden bit-identical.
  - `ruff` and both self-checks pass.
- **Smoke** (development, seed 0, plus seeds 1 and 2 for the CartPole base; scratchpad):
  - **CartPole base, 400 episodes:**
    - k 3.50, s_in 0.283, η scale 2.15 / 2.55.
    - Seeds 0 and 2: T_s 252 and 260.
    - Seed 1 does not solve: its last 100 episodes average 499.5, with the last imperfect episode at 303.
    - That is **2 of 3, which misses the step's 3-of-3 criterion**. The CK-matched nlif had T_s 154–179.
  - **Column-rate noise** at init (one observation, 400 copies): the CV of ρ̄_k is 0.093 at 41 Hz vs 0.029 at 143 Hz. The logit noise s.d. is 0.93 vs 0.28, because α_a grows by k while the CV grows by about 3×.
  - **LunarLander `--eta-cal`, 51 episodes:** k 9.12, s_in 0.100, −191 (3.2 ms/frame).
  - **CartPole cols χ 0.3, 100 episodes:**
    - max Re λ 0.23, τ_loop 26 ms, χG 0.23;
    - causal share 0.55%, up from 0.17% at the CK-matched rates;
    - R rate 53 Hz;
    - within-module share 0.92 → 1.00 with λ on (0.92 → 0.93 with λ 0);
    - last-20 return 43 / 38;
    - 2.3 / 2.0 ms/frame vs 0.9–1.1 for the base.
  - **LunarLander lat, 20 episodes:**
    - max Re λ 0.23, causal share 0.11% (from 0.01%);
    - R posts and latents at 51–55 Hz, none near the ceiling;
    - within-module share 0.68 → 0.75, latent output share 0.25 → 0.20;
    - 4.7 ms/frame.
  - **Counter validation, LunarLander at init (54 Hz, μ 1.3–2.6):**
    - Markov: synapse corr 0.887, ratio 0.99. Siegert gives 0.888 and 1.03.
    - The truth's split-half reliability caps the correlation at about 0.88, so the counter reaches the noise ceiling.
    - The gain is about 37 Hz/θ on every post.

### Added (iteration 015, step 2: recurrent columns in the TD-STDP actor, `--rec {cols,lat}`; engine nlif only)

- **Gain estimator: the kernel's own transfer, not Siegert.** The nlif posts run mean-driven (CartPole 140–190 Hz, LunarLander 365–470 Hz). In that regime the Siegert gain is wrong on the 1 ms grid.
  - It stays flat at about 48 Hz/θ. The kernel's gain is 45 at μ = 3 and 31 at μ = 8.
  - Rates mode-lock at 500 Hz (period 2 steps), where the gain is 0.
  - `mapstdp/lif.py` gains `markov_rate`, the stationary rate of the discrete kernel from the transition matrix of V, with the background noise of `p.kernel`. `gain_table` caches it (about 15 s per parameter set), and `gain_markov` interpolates it bilinearly in (μ, per-step input s.d.). Like Siegert, it is an abstraction of what a neuron knows about its own operating point.
  - Self-check (g): it agrees with simulation within 2% at μ = 0.7, 3 and 8 in both noise modes. Siegert gives 48.6 and 48.3 where the kernel has 44.7 and 30.9.
- **Counter validation** (replay truth, open loop with common random numbers, 20 ms window, χ_rec 0.3; scratchpad, development). Synapse correlation and ratio c / marginal:
  - CartPole init: Markov 0.974 / 1.04, Siegert 0.969 / 1.22.
  - CartPole, input rows scaled over U(0.02, 1.3): Markov 0.981 / 1.02, Siegert 0.970 / 1.14.
  - LunarLander init: Markov 0.935 / 1.06, Siegert 0.776 / 2.0.
    - Gain per (copy, post): Markov corr 0.89; the split-half reliability of the truth is 0.66–0.72.
    - Binned by μ, the correlation is 0.996 for Markov and 0.70 for Siegert.
  - Removal is 1.00–1.03 × marginal here, so the response is linear on the scale of one kick.
- **Credit check** (Monte Carlo, scratchpad): E[r(A) Σ_h c^h ψ^h] vs R ⊙ ∂J/∂R by finite differences with common random numbers, on a frozen 2 × 5 column nlif actor at 222 Hz. Here J = E[Σ_a p_a r_a], with CK's softmax readout per 20 ms window.
  - Cosine 0.976: 0.950 within modules, 0.993 across.
  - The finite-difference truth is reproducible to 0.970 across two noise streams.
- **`mapstdp/tdstdp.py`:**
  - `Layout(..., n_lat=0)` adds latent posts at the end: sign +1, zero Mv and Ma, mask `lat`. It is bit-identical at 0.
  - `HybridRec`:
    - R is an E→E matrix among the actor E posts, one module per action, plus `n_lat_mod` latent modules of n_ae posts. Its SBM init (p_same 0.5, p_cross 0.05, U(0.5, 1.5), d_j = 1) comes from its own stream.
    - Kicks are χ_rec R/d_j.
    - Per 20 ms window and copy: G from `gain_markov`, the pair counts (`lif.pair_counts`, tail carried) and the counter c^h = τ_eff b G_h n_hᵀ (Eq. 11b).
    - One-hop credit (Eq. 4b): e_h ← e^{−20/τ_q} e_h + c^h ψ^h. The module score is CK's α_a (A − p)/n_ae, 0 on latents. Credit pairs with the counter of the window whose rates gave p.
    - Task step: η_rec mean_h(Σ_t δ) e_h.
    - Per statistics frame (5 windows), in order:
      - observed walk with an EMA over 20 frames;
      - map modulator, minus the latent-output barrier in the lat arm;
      - structural increment with λ;
      - homeostasis (Eq. 4d).
    - Monitors: max Re λ(Λ̂^K), χ_rec τ_eff G, τ_m/(1 − max Re λ), causal share, within-module weight share, L_obs, silent-module frames, rates, latent rates and output share, latent→action synapses at w_min, and action perseveration against Σp².
  - `check_counter`, self-check (h3): corr 0.933, ratio 1.03, replay mismatch 0.
- **`experiments/tdstdp.py`:**
  - `Config` fields `rec`, `chi_rec`, `lam`, `lam_ratio`, `lam_rec`, `eta_rec`, `stat_windows`, `pair_ms`, `n_lat_mod` and `la_beta`, with flags and tokens after the engine token.
  - `rec` requires nlif, `--block 20` and no Adam.
  - Input→latent W rows are frozen at init: c, κ and the update are zeroed there.
  - `calibrate_rec` is a pre-set, frozen run:
    - η_rec gives R the actor W's relative step per window;
    - λ = lam_ratio · RMS(task step over a frame) / RMS(structural step).
  - `calibrate_s_in` and `calibrate_ck` now run without recurrence, so every arm of a seed shares s_in and η.
  - Checkpoints carry R, e, the tail, the EMA tables, the logs and the snapshots.
  - `--save-model` writes R snapshots (episode 0, every 100 CartPole / 500 LunarLander episodes, end) with m, roles and mask.
  - The result's `rec` holds the calibration and the monitor time series.
  - Default paths are unchanged.
- **`tests/test_hybrid.py`** (+4 tests):
  - h2: rec cols with χ_rec = 0 equals the nlif baseline bit for bit (W, returns, membranes, both streams), for H = 1 and 3;
  - the credit against a per-synapse loop;
  - lat: frozen latent W rows and a checkpoint round trip.
- **Checks:**
  - `pytest -q tests`: all pass, golden bit-identical.
  - `reports/tdstdp-baseline*` summaries are byte-identical.
  - `ruff check .` passes.
- **Smoke** (development, seed 0, scratchpad):
  - **CartPole**, 20 episodes, cols:
    - χ 0.3, λ on: η_rec 0.124, λ 0.80. max Re λ 0.25, so τ_loop is 27 ms. Causal share 0.17%. Within share 0.92 → 0.98 (λ 0: 0.92 → 0.92). L_obs 6.20 → 6.20 bits, chance-dominated. 190 Hz.
    - χ 0.15: max Re λ 0.13, 163 Hz.
    - Cost: 2.4 vs 1.5 ms/frame for nlif on the same loaded node.
  - **LunarLander** `--eta-cal`, lat, 10 episodes:
    - η_rec 0.52, λ 4.4;
    - max Re λ 0.09–0.26, R rates 440–470 Hz with the latents alike, so most posts sit on the 500 Hz lock with gain near 0;
    - within share 0.69 → 0.71, latent output share 0.24 → 0.23;
    - 6.7 vs 4.1 ms/frame.

### Added (iteration 015, step 1: noisy-LIF engine for the TD-STDP backbone, `--engine nlif`; no recurrence yet)

- `mapstdp/lif.py`:
  - `_kernel_ext_py` / `_kernel_ext_nb` and `simulate_ext(Iext, Bt, ridx, st, rng, p)`. The kernel takes a given per-step external current (signed, threshold units) instead of the teleport pulses, plus recurrent kicks on an index set (empty for now). Noise, leak, threshold and reset are the same as `_kernel_nb`. The compiled and interpreted kernels are bit-identical in both noise modes. Existing kernels are unchanged.
  - `DT1`, the dt = 1 ms operating point from `calibrate`: μ_bg 0.5778, χ 0.01551, σ_V 0.2, **no refractory period**.
    - A 2 ms clamp caps the rate at 1/3 per ms. That is below BindsNET's initial LunarLander rate (0.365 per ms), so the rate match was infeasible.
    - On CartPole the clamp also saturated the critic at about 0.34 per ms (V ≤ 0.48, while the preset needs V ≈ 1). In a 2-seed smoke at 2 ms, neither seed solved.
  - Self-check (f): the calibration, Siegert against simulation at dt = 1 ms (rate within 0.7%, gain within 2%), and `simulate_ext` at zero input against Siegert (2.59 vs 2.55 Hz). Earlier self-check output is unchanged.
- `mapstdp/tdstdp.py`:
  - `simulate_nlif`: W in mV as in BindsNET, with current s_in x(t−1)Wᵀ/13 in threshold units, τ_m 20 ms and background noise on every post, the critic included.
  - `check_bindsnet_limit`, self-check (h1): the ext kernel with σ_V = 0, μ_bg = 0, no refractory period, τ = 100 ms and threshold 13 mV reproduces `simulate_bindsnet` bit for bit over 20 ms blocks, raster and membranes, in both mV and threshold units.
- `experiments/tdstdp.py`, engine `nlif` (authors' code path only):
  - W, init and η are in mV as for engine `lif`, and the decision readout, rates and rule are unchanged.
  - Membranes are carried across blocks and set to μ_bg on an episode reset (code 2), with no draw. The neuron noise has its own stream, so the main stream (inputs, actions, env seeds) is consumed as in engine `lif`.
  - `Config.s_in`, `--s-in` and token `si<S>`. If s_in is not given, `calibrate_s_in` sets it before the run by bisection: the mean initial post rate (`probe_rate`, W frozen, CAL_SEED stream, CAL_TICKS ticks) equals engine `lif`'s. It is not tuned on return. The result stores `s_in` and `s_in_cal` (the lif and nlif rates).
  - Checkpoints carry the membranes, the noise stream and s_in, and a resume reuses the stored s_in.
- `experiments/report.py`: `ENGINE_FLAG["nlif"]`. Existing studies are unchanged.
- `tests/test_hybrid.py` (4 tests): compiled vs interpreted ext kernel, h1, and an nlif checkpoint round trip with the episode reset.
- Smoke (development, scratchpad, not committed):
  - **CartPole**, 400 episodes:
    - nlif: s_in 0.94, T_s 154 / 179 / 172 on seeds 0–2;
    - lif: T_s 202 / 150 / 228;
    - 0.67–0.88 vs 0.64–0.68 ms/frame.
  - **LunarLander** `--eta-cal`, 20 episodes:
    - s_in 0.975;
    - mean return −223 (nlif) vs −198 (lif);
    - 1.99 vs 2.02 ms/frame;
    - s_in calibration takes about 20 s.

### Changed (study `emergent-v4`, CartPole only; not launched)

- `experiments/report.py`, `STUDIES["emergent-v4"]`:
  - CartPole only (1500 episodes, seeds 2000–2019);
  - arms on / off / sealed (`--lam 0 --p-cross 0`) on four engines, 12 groups, 240 runs;
  - 8 heads.
- `M3_ENGINE_FLAG` no longer fixes `--heads`; each study passes it.
- Pairs take an optional family, `(a, b, family)`, with Holm applied within a family. On vs off is `primary`; on vs sealed is `sealed`.
- The summary's pair table now has Group / Against / Family columns, and `experiments/structure.py` accepts 3-tuples and labels its pair table generically. The test is updated.
- **Added:** `m2.BOUND_CAL_QIF[8]` (0.67–4.26), from `experiments.bound_cal --engine qif --heads 8` on the pilot014 flow CartPole models. The same models at 16 heads give 1.01–5.11, within 11% of `BOUND_CAL_QIF[16]`. Without it, QIF could not run diffbound at 8 heads.
- `reports/emergent-v4/README.md` was rewritten as a draft pre-registration. It covers:
  - P1 (AUC) and P2 (L_obs), on vs off, Holm over 4 engines;
  - S, on vs sealed, its own Holm family;
  - frozen-W returns and structure as secondary results;
  - 32 workers, run from a frozen worktree.

### Results (iteration 014, second pilot; development)

- `docs/iterations/014-emergent-v4.md` §10: CartPole, 3 seeds × 500 episodes.
  - The map term raises AUC on every spiking engine (branching 94 → 204, QIF 79 → 182, LIF 174 → 235) and lowers L_obs on all four. Flow is unchanged in AUC.
  - Sealed (λ = 0, `--p-cross 0`) matches on for AUC, but on reaches lower L_obs than sealed on the spiking engines.
- **Decided:**
  - 8 heads adopted (same returns at half the cost);
  - LIF 20 ms window harmless, not adopted;
  - QIF homeostasis ×2 dropped.
- **Reversal (user decision):** the LunarLander part of the pilot was cancelled; 7 runs were stopped with checkpoints. The `emergent-v4` study becomes CartPole-only, and LunarLander moves to the TD-STDP backbone study (iteration 015).

### Changed (iteration 014 wording)

- `docs/iterations/014-emergent-v4.md` §7: the sealed control is reframed as learned versus prescribed structure (sealing is the minimum-description structure for the given partition), not structure versus sealing. The readings now cover sealed ≈ on, on > sealed (with candidate mechanisms), sealed > on, and what a sealed fix of the collapse implies, given that routing pairs are exempt in both arms. No change to the pilot or study design.

### Changed (canonical derivation: the emergent walk)

- `docs/derivation.md` is rewritten from the user's emergent-walk drafts (`docs/derivation-guide-emergent.md`, `docs/derivation-details-emergent.md`, PDF). The walk is observed from spikes: pair counts κ set the walk and the baseline ē^obs, a signed counter c sets each synapse's step (Lemma E, Eq. 9; Theorem E, Eq. 10), and homeostasis is −ε_h (d_j − 1) π_j W_ij / d_j (Eq. 4d). Section §5 is now the four engines (flow, branching, LIF, QIF) with their counters (11a–11c), and §7 the E–I stability condition (12). The K-series Proposal (K1–K9) is dropped. The flow engine is the special case a = X = 0 and its rule is unchanged.
- Equation map: 1–7, 4a–4h keep their meaning, rewritten for the observed walk (4b now uses the signed counter; 4d as above); new 8–12.
- Corrections to the drafts: the column-sum stability bound with inhibition is max(χ_E, χ_I) G_max < 1, not the sum (D10 Step 4); homeostasis pulls d_j toward 1 rather than conserving it; the branching counter is bounded by n_j |b_ij| / 2θ rather than proportional to W_ij, so Dale's law holds for small steps, with the w_min clip in code.
- **Open:** the mean counter (11b) and Eq. 4d make updates proportional to W_ij, which conflicts with the old crossbar rule against weight-proportional decay (§6, marked Open).
- Neuroscientist review (accepted with changes, folded in): c is presynaptic arrival × postsynaptic sensitivity (voltage-based plasticity), Siegert gain and homeostasis are tagged abstractions, the module-activity signal is own-module plus global activity (not a neuromodulator), chance caps M* and pulls ē^obs toward 1 − p_m, running estimates of ē^obs and M* over 10–50 frames are allowed.
- Checks (`/derivation-check`, new `fdcheck.py --emergent`, Appendix A F1–F4): Lemma E vs finite differences 3.8 × 10⁻⁷ (excitatory) and 3.7 × 10⁻⁷ (4 of 16 inhibitory); Theorem E cosine 1.0000000; reduction to the old Lemma 1.0 × 10⁻¹⁷; Monte Carlo of the sampled rule 0.999999 / 0.999998.
### Added (code: the emergent-walk rule, behind `rule="emergent"`; default `"v4"` keeps `tests/golden.json` bit-identical)

- `mapstdp/core.py`: `observed_walk` (Eq. 8, module rows only, silent columns fall back to T), `structural_update_emergent` (Eq. 4 with ḡ^obs), `homeostasis_update_emergent` (4d), `eligibility_step_emergent` (4b, one-hop credit from the observed walk), `module_offspring` (Eq. 12 monitor), `mstar_ceiling`, `lemma_e_grad` (Eq. 9); self-check adds Lemma E vs finite differences (4.4 × 10⁻⁷) and Theorem E.
- `mapstdp/spiking.py`: `run_heads(..., return_drive=True)` (no extra draws, raster unchanged) and `branching_counters` (exact counter, Eq. 11a; firing probability clip(u/2θ, 0, 1), forced external spikes contribute 0). Monte Carlo on frozen branching W: cosine 0.999995 with −E_j W ⊙ ∇L (κ as the step: 0.907).
- `experiments/m2.py`: Config `rule`, `lam_scale`, `eta_scale`, `obs_tau` (running observed-walk statistics), ablations `ebar` (`obs` | `module`) and `step_stat` (`c` | `kappa`); per-episode monitors (L_obs, M* and its chance ceiling, ē^obs, ρ(Λ̂^K), causal share, clip rate, silent frames, block weight shares); held-out L_obs in `evaluate`; `match_scales` (step sizes RMS-matched to flow at initial W); checkpoints every 50 episodes with resume (`resumes` in the result); `save_model` (W at episode 0, every 250 and final, float32 npz). Flow under `rule="emergent"` is bit-identical to v4.
- `experiments/m3.py`: `--rule`, `--obs-tau`, `--ebar`, `--step-stat`, `--lam-scale`, `--eta-scale`, `--match-scales`, `--out`, `--save-model`.
- `experiments/report.py`: studies take `runner` (`tdstdp` | `m3`); the m3 runner saves models under `reports/<study>/models/`, `summarize_m3` writes summary.csv, pairs.csv (Mann–Whitney with Holm per metric over declared pairs) and summary.md. tdstdp summaries are byte-identical (tested).
- `tests/test_emergent_rule.py` (10 tests).
- `mapstdp/lif.py` (new): noisy LIF engine, normalised units (θ = 1), τ_m 20 ms, dt 0.5 ms, OU background σ_V 0.2, 100 ms frames, window 60 ms with pairs straddling frames credited once, warm-up of 60 ms at episode start; counters `siegert` (default, Eq. 11b with 11c plus a discrete-time threshold correction), `psi`, `regression`; `siegert_gain` (also used by `tests/test_emergent.py`, whose import of the absent `test_eligibility` it replaces). **Offspring χG = 0.5, not 0.8:** the convex excitatory-only LIF ignites a high-rate state near 0.8. Checks: Siegert rate and gain within 1%; counter vs weight-scaled derivative corr 0.945, ratio 1.04; vs finite removal 0.94, 1.54; pair counts are about 99% chance; 17 / 29 ms per frame (CartPole / LunarLander, 16 heads).
- `mapstdp/qif.py`: emergent counter (Eq. 11b) with raw pair counts in the existing window (pairs straddling frames counted), E neurons only; gain from a running regression of rate on the injected noise current (total input gives the wrong sign, because the fixed pools respond to the neuron's own spikes), shrunk to module and mean-field gains. Checks: counter vs removal corr 0.89, ratio 1.04; gain vs finite differences corr 0.93, ratio 1.01. `--match-scales` should use ≥ 50 frames on QIF.
- `experiments/m2.py`, `m3.py`: `engine="lif"` (`--engine lif`, `--lif-<field>` overrides; requires `rule="emergent"`), burn-in 40 steps (20 ms), per-episode LIF state with any carried gain checkpointed; evaluation streams never touch training state. `match_scales` defaults to 50 frames on spiking engines (CartPole: LIF λ × 1.48, η × 1.09; QIF 0.98, 0.86; branching 1.26, 0.91).
- `experiments/bound_cal.py` (new): the iteration-010 diffbound calibration as a CLI (trained flow models, 20 held-out stimuli × 200 warm frames, β_eff by maximum likelihood to Eq. 1e). It reproduces `BOUND_CAL_QIF` within 6%. New `BOUND_CAL_LIF` (H = 16): β_eff 0.47, 0.61, 0.78, 1.06, 1.63, 2.20, 2.35 at θ = 1–64. **The LIF readout saturates at β ≈ 2.35** (background spikes reach every bound), and at θ = 64 69–84% of frames hit the deadline, so v4's `--beta-max 16` is unreachable on LIF.
- `docs/neuron-variants.md` rewritten for the observed walk: chance as teleportation, why two statistics, one section per engine with measured numbers, LIF and QIF limitations in one place, and a side-by-side table. `CLAUDE.md`: the derivation description, the `rule="emergent"` code map and commands.
- `experiments/structure.py` (new): from saved models, per snapshot the block weight shares, the K × K module weight table and L_W (map equation of the trained W's own walk under uniform teleportation, engine-free); on-vs-off Mann–Whitney per declared pair; heatmaps of the mean final module table per arm and on − off (`structure_<on>.png`). Smoke (flow CartPole, seed 0, 300 episodes, dev): L_W 4.37 bits with the map term against 6.12 without; within-module weight 63% against 46%; latent → action weight about 0.5% with the map term (the neuroscientist's predicted disconnection). Models are about 22 KB each, so they are committed with a study.
- `docs/derivation.md` §5.3–§5.4 and Appendix A: the counter estimates the weight-scaled derivative (finite removal is about 0.65 of it on the v4 LIF network); the ψ counter is exact only without reset; the LIF offspring and chance share; the QIF noise-instrument gain.
- `ruff.toml`: per-file ignores for the user's standalone `tests/test_emergent.py`. Its `test_lif_inhibitory_kick` fails until `siegert_gain` exists (it imported the absent `test_eligibility.py`).

### Results (iteration 014 pilot, development: 56 runs, 2 seeds, 300 episodes, `experiments/results/pilot014/`)

- `docs/iterations/014-emergent-v4.md` (new): pilot design, results, decisions.
- The map term lowers description length on all four engines and both tasks: held-out D 6.5 → 4.0–4.5 bits (CartPole) and 7.4 → 3.8–4.5 (LunarLander), against 5.8–6.5 and 6.6–7.6 without it. L_obs falls too, by only about 0.3 bits on LIF, whose pair counts are about 99% chance. Non-routing cross-module weight falls from 15–37% of the total to under 1.3%.
- Performance is mixed at this length. The map term is ahead in AUC for branching, QIF and LIF on CartPole and for QIF and LIF on LunarLander, and behind or mixed for flow and for branching on LunarLander.
- QIF with the map term runs at ρ(Λ̂^K) ≈ 1.0 (5% silent-module frames on LunarLander).
- Decisions:
  - LIF uses diffbound with β ≤ 1.6. With β ≤ 16, 36–61% of frames hit the deadline, past BOUND_CAL_LIF's 2.35 saturation. Race scores about 30 on CartPole.
  - The rule ablations `--ebar module` and `--step-stat kappa` do not replace the defaults.

### Found (iteration 014 §7: the action collapse)

- In the map-term-off spiking arms, the weights swing for 20–60 episodes at a time to one action. Controller→A1 weight reached 2.7–9× controller→A2, and the flow policy on the same W also collapses, so the readout is not the cause.
- The map term prevents this by sealing the modules (every action module's within-module offspring is 0.80). Part of its task benefit on spiking engines is therefore sealing, not a better abstraction.
- Training returns can hide a collapsed W: branching off seed 0 had a training return of 295 over its last 30 episodes and 10 with frozen weights.

### Added (iteration 014 §7–§8: a sealed control and runtime)

- `m2.Config.p_cross` and `m3 --p-cross`: the SBM's cross-module connection probability outside the routing pairs. The default 0.05 is unchanged. 0 builds the modules sealed: the λ = 0 control for "sealed modules help". The draws are the same; only the mask differs. Results record `p_cross` only when it is not the default.
- `core.build_network`: casts `p_cross` to float. An integer 0 had made the whole probability table integer, so every connection probability truncated to 0.
- `mapstdp/lif.py`:
  - numba loop `_kernel_nb`. `LIFParams.kernel`: `nbf` (default) draws variance-matched Irwin–Hall(4) noise from a xorshift64* generator, `nb` Gaussian noise, `np` the numpy reference.
  - External pulses are drawn as one Poisson count per head and step, split by v. One seed per frame comes from the frame's rng.
  - numba `pair_counts` (identical integer counts).
  - A LIF frame now takes 2.6 ms instead of 19 ms. LIF runs are not bit-identical to the pilot's: the stream differs, the distributions do not (module rates within 0.4%; the self-check passes).
- `mapstdp/spiking.py`: numba `branching_counters` over the synapses j → i only. Equal to the numpy path up to summation order. Per frame 8.4 → 1.3 ms (CartPole) and 12.2 → 2.0 ms (LunarLander).
- `experiments/report.py`: runs set `NUMBA_NUM_THREADS=1`. QIF's `parallel=True` kernel had started up to 72 threads in each of 40 processes.
- `tests/test_emergent_rule.py`: numba paths against numpy, the three LIF kernels' rates, and the sealed build.
- Worker benchmark (iteration 014 §8): throughput peaks at about 60 concurrent runs (3726 frames/s) and is 91% of that at 48. Under load, branching costs 12 ms/frame, LIF 10 and QIF 36, against 53–112 in the pilot.
- `m2.frozen_returns`, called at the end of every `rule="emergent"` run (`frozen_returns` in the result): 20 episodes with W frozen, on a deep copy of the agent, from the stream (seed, `FROZEN_SEED` = 7331). `report.m3_stats` and the pairs gain `frozen` (summary column "Frozen").
- `m2.BOUND_CAL[8]`, `BOUND_CAL_LIF[8]`: diffbound tables for 8 heads, from the pilot014 flow CartPole models (iteration 014 §9).
- Second pilot launched (`experiments/results/pilot014b/`, 116 runs): sealed control, LIF 20 ms window, QIF homeostasis ×2, 8 heads, LunarLander at 3000 episodes.

### Added (study `emergent-v4`, declared, not launched)

- `experiments/report.py`: `STUDIES["emergent-v4"]`: {LunarLander 1000, CartPole 1500 episodes} × {flow, branching, QIF, LIF} × λ {0.05, 0} × seeds 2000–2019 (320 runs, LunarLander queued first), with 8 on/off pairs.
- `reports/emergent-v4/README.md`: pre-registration draft, awaiting the user's approval. P1 is AUC and P2 is end-of-run held-out L_obs, each on vs off by two-sided Mann–Whitney with Holm over the 8 cells.

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-24

### Changed (no-Adam arm renamed: `sgd` → `noadam`, `--sgd` → `--eta-cal`)

- "SGD" read as backprop's stochastic gradient descent, which is non-local. The arm is the local three-factor rule without Adam, with a fixed η per population calibrated once at initialisation (E4′ rule).
- `experiments/tdstdp.py`: the flag is `--eta-cal`; `--sgd` stays as a hidden alias, because manifests and the 013 record contain it. The result-file token stays `sgd`, so existing intermediates keep their names.
- `experiments/report.py`: a study's `labels` map run keys to display names. `summary.*` show `LunarLander_noadam_<engine>`, with a "Display names" section. Run directories and manifests are unchanged.
- Both study READMEs carry an amendment. Both summaries were rebuilt from the same `runs/`, and no number changed.

### Results (`reports/tdstdp-baseline-flow/`, 60 runs, 0 failures)

- CartPole, no Adam: 17/20 solved at B = 20 and 18/20 at B = 1 (Fisher p = 1.0), so B = 20 stands.
- LunarLander: 3/10 solved with Adam; 0/10 without Adam. The no-Adam return peaks at 67–103 and then falls to −24.
- Record 013 §14.

### Added (flow-engine arms of the TD-STDP baseline, `reports/tdstdp-baseline-flow/`)

- New companion study `tdstdp-baseline-flow`, pre-registered in its README: TD-STDP on the flow engine (the branching step's noise-free limit) for CartPole without Adam, and LunarLander with Adam and with SGD, on the baseline's seeds. It is a separate study because the baseline ran at `675e49e` and a study uses one code state.
- A CartPole block-1 group is the control for the block sum, which was gated on branching only. On dev seeds 60–79, flow at B = 20 collapsed 4/20 CartPole runs to a one-action policy, against 0/20 at block 1.
- `experiments/report.py`: `ENGINE_FLAG` gains `flow`; a study's `reference` may name another study's group as `"<study>/<group>"` (`load_runs`). New test `test_summarize_cross_study_reference`.

### Added (explainer `docs/neuron-variants.md` and the LIF walk diagnostic)

- **`docs/neuron-variants.md` (new):** a plain-language guide to Map-STDP on the four engines (flow, branching, LIF, QIF). It covers what the rule assumes (pairings in proportion to link flow, flow matching the walk, subcriticality), how each engine meets those assumptions, the limitations of LIF and QIF, a side-by-side table, and a glossary. It explains the derivation and does not replace it.
- **LIF diagnostic** (frozen weights, `docs/iterations/013-tdstdp-backbone.md` §13; pre-registration, scripts and text outputs archived in `docs/iterations/013-lifdiag/`):
  - Model v4's recurrent network on the TD-STDP LIF fails the walk checks as it is. With input noise of 2–4 mV/√ms it passes subcriticality, the structural-rule direction (cos 0.94–0.95) and settling, but no operating point makes the pairing count proportional (slope 0.5–0.75), so task credit is 20–35% off.
  - On trained, nearly sealed weights the structural rule loses direction even on the branching engine (cos 0.44–0.89), from module-level co-fluctuation in κ on cross-module synapses. This needs a closed-loop check.
- `ruff.toml` excludes `docs/iterations` (archived diagnostic scripts).
- `CLAUDE.md` lists `docs/neuron-variants.md` under Docs.

### Changed (`reports/tdstdp-baseline/`: CartPole Adam groups dropped)

- **Decision (user):** `CartPole_adam_lif` and `CartPole_adam_branching` are dropped from the reported baseline. Both solved 0/20, because the authors' Adam step (the per-ms SGD η) is 40–160× LunarLander's for CartPole and collapses the policy at once. CK report no CartPole Adam configuration, and CartPole solves without Adam (20/20 on `lif`).
- `experiments/report.py`: studies take a `dropped` map (group → reason). `summarize` leaves those groups out of `summary.csv` and the tables, and lists them under "Dropped groups". Their run files stay in `runs/`.
- The study README has a dated amendment, and `summary.csv` and `summary.md` were regenerated. No other group, gate or statistic changed. 9 tests pass and ruff is clean.

## 2026-10-23

### Added (final-results layout and the reported TD-STDP baseline, `reports/tdstdp-baseline/`)

- **Layout (decided):**
  - `experiments/results/` is now, by documentation, the intermediates store: every dev, tuning and diagnostic run, and the default output of every CLI. No file was moved or renamed.
  - New top-level `reports/<study>/` holds only final reported outputs:
    - `README.md`, the pre-registration;
    - `manifest.json`: git HEAD, the sha256 of the uncommitted code in `mapstdp/` and `experiments/` (untracked files included, results excluded), with `code.patch`; versions; per-run commands;
    - `runs/<group>/s<seed>.json`;
    - `summary.csv` and `summary.md`.
  - Checkpoints and partials there are gitignored.
- **Code:**
  - New `experiments/report.py`: `run <study> [--workers N]` (launch, skip finished runs, resume checkpointed ones, refuse on a code-state change) and `summarize <study>` (from `runs/` only). Studies are declared in `STUDIES`.
  - `experiments/tdstdp.py`:
    - `--out PATH` (default unchanged);
    - `group_stats`, factored out of `summary()`. The `--summary` output is byte-identical before and after.
  - New `tests/test_report.py`: `summarize` on a tiny fake study gives the expected csv.
  - Golden is bit-identical, 9 tests pass and ruff is clean.
- **Docs:**
  - `CLAUDE.md`: the results layout, the `report` commands, and the trap "final numbers come only from `reports/`; never hand-copy results there".
  - Record 013: new §12, and a status line. The E1′, E3′, E4′ and S1–S3 numbers stay as development results.
  - Fixed three `\times` in record 013 §5.2 that had been corrupted to a tab followed by `imes`.
- **Study `tdstdp-baseline`** (pre-registered in its README):
  - Settings: the authors'-code path, B = 20, `--fast`, fresh seeds.
  - CartPole: {no Adam, Adam} × {lif, branching}, seeds 1000–1019.
  - LunarLander: {Adam, `--sgd`} × {lif, branching}, seeds 1000–1009.
  - 120 runs, 7.3 h, 0 resumes.
  - **Gates:**
    - CartPole no-Adam lif: 20/20, pass;
    - LunarLander Adam lif: 8/10, pass.
  - Reported:
    - CartPole branching: 5/20.
    - CartPole Adam: 0/20 on both engines. The authors' Adam step equals the SGD η, which collapses CartPole; this is not a code fault, because B = 1 behaves the same.
    - LunarLander Adam branching: 6/10.
    - LunarLander SGD: 0/10 on both engines.
  - The numbers are in `reports/tdstdp-baseline/summary.md`.

### Added (iteration 013: TD-STDP speed-ups and the flow limit, record §10–§11)

- **Code:**
  - `tdstdp.block_learn_ck_fast` with `--fast`: the update is summed over copies in one product. It is equal to the default up to summation order.
  - `tdstdp.flow_step` with `--engine flow`: the branching step's noise-free limit, rates instead of spikes, plus the input-autocorrelation term of z.
  - The defaults are unchanged and bit-identical. This was checked against the committed E1′ (both engines) and the legacy E1.
  - New self-checks:
    - (b″) the fast form equals `block_learn_ck` to $2 \times 10^{-16}$;
    - (e) the flow update against the Monte Carlo mean of the sampled rule: cosine 0.99997.
  - Golden is bit-identical, 8 tests pass and ruff is clean. `CLAUDE.md`'s Commands line lists `--block 20`, `--engine flow` and `--fast`.
- **S1, block sum without Adam** (CartPole, seeds 40–59, B = 20 against B = 1): **passes its gate on both engines.**
  - LIF: solved 20 against 19, Mann–Whitney p 0.27 on $T_s$ and 0.54 on return.
  - Branching: solved 6 against 7, p 0.99 and 0.39.
  - 6.7–7.1× faster.
- **S2, `--fast`** (LunarLander, seeds 100–104, 400 pooled episodes): the return sequences are identical to the default's. It is 1.8× faster per copy-frame (8.26 against 4.48 ms under load). Env stepping is about 1% of a tick, because the copies already share one batched network step.
- **S3, flow limit** (CartPole, seeds 60–79): 20/20 solved, $T_s$ 212.7 ± 51.7 and return 467.7 ± 107.8. Against E1′ branching (4/20), Mann–Whitney p $7.4 \times 10^{-8}$ on $T_s$. Escape noise is what fails branching's strict $T_s$.

### Changed (iteration 013: TD-STDP follows the authors' code; E1′, E3′ and E4′)

- **Code** (`mapstdp/tdstdp.py`, `experiments/tdstdp.py`):
  - `python -m experiments.tdstdp` now defaults to the authors' code (`reference/td_stdp`, BindsNET 0.2.7) wherever it disagrees with the paper. Results carry the token `ck`.
  - `PRESETS` are their `config_cp.ini` and `config_ll.ini` (η in mV). Weights convert to branching units by `MV` = 13, not 1300.
  - New `tdstdp.block_learn_ck`: their per-ms order with the $1/\tau_z$ and $1/\tau_q$ trace scale.
  - New `tdstdp.simulate_bindsnet` (`--engine lif`): their deterministic LIF with W in mV.
  - New `CKRunner`: their loop. The action is sampled from the response to the current state; p is fixed at the sampling probability; $V' := 0$ only on true termination; each episode has 6 presentations of $s_0$; the update is the mean over all copies; episodes are pooled over copies, with their min/max episode stopping rule.
  - Also new: their Fourier encoder and bounds; Adam ε mapped to the unit-trace gradient; `calibrate_ck` (the E4′ η rule); and checkpoints for long runs (`*.ckpt.npz`, now in `.gitignore`).
  - `--legacy` keeps the pre-audit path. It reproduces the committed E1/E2 results bit for bit under their original names.
  - `--summary` separates the two families and counts the authors'-code episodes pooled. It now reports Mann–Whitney p on $T_f$, $T_s$ and return, plus the number of runs with a last-100 return of at least 475.
- **Self-check:**
  - (b′) `block_learn_ck`'s closed form equals a literal per-ms transcription of `FBTDSTDP` to $6 \times 10^{-16}$.
  - (d′) The actor rule in the authors' order on a bandit has cosine 0.984 with the exact gradient, and 1.0000 per row.
  - Golden is bit-identical, 8 tests pass and ruff is clean.
- **Stopped:** the first E3 and E4 (LunarLander) were stopped at the wrong settings: the Adam step was 100× too small, episodes were counted per copy, entropy was 20× too weak, and decay was on. Their partials were moved out of `experiments/results` to the session scratchpad.
- **E1′** (pre-registered in record §9; CartPole, the authors' config, no Adam, B = 1, seeds 20–39):
  - **BindsNET LIF passes the gate:** 20/20 solved, $T_f$ 64.0 ± 8.4, $T_s$ 174.8 ± 18.5 (median 168), against CK's 57.0 and 169.5.
  - **Branching fails it:** 4/20 solved and last-100 306.9 ± 214.7. It has the same $T_f$ (62.6, MW p 0.46), but 7/20 collapse into a constant action after a period of near-perfect play (last-100 against LIF: MW p $6.8 \times 10^{-4}$).
- **E3′** (LunarLander, the authors' config, Adam, branching, B = 20, seeds 0–9): **passes its gate.**
  - 7/10 solved; $T_f$ 390.4 ± 99.8 against CK's 383.8.
  - $T_s$ 6774 ± 1896 pooled episodes against CK's 2575, so 2.6× slower.
- **E4′** (the same without Adam, η by the pre-registered Adam-equivalent rule): 0/10 solved and last-100 −0.9 ± 45.8. Against E3′, Mann–Whitney p ≤ 0.002 on $T_f$, $T_s$ and return. Adam is load-bearing on LunarLander.
- **LIF arm** (reported only): seeds 0–3 solved 3/4. Seed 4 was stopped at 5600 of 10000 pooled episodes; it can be resumed from its checkpoint.
- **Interruption:** all 25 LunarLander runs were killed at about 15:15 by a coordinating agent and resumed from their checkpoints. W, the Adam state, the returns and the rng were restored exactly; fewer than 50 episodes per run and the in-flight episodes were lost (record §5.2).
- **Docs:**
  - `docs/iterations/013-tdstdp-backbone.md` is rewritten to the current state. §3 states the corrected units and timing, lists the paper-against-code disagreements, and keeps the pre-audit choices. The pre-fix E1/E2 results are kept and labelled as such. §6's earlier reading that Adam is load-bearing is withdrawn for CartPole; E4′ shows that it holds for LunarLander.
  - `CLAUDE.md`: the tdstdp Code and Commands lines (`MV` = 13, `--legacy`).
  - `docs/derivation.md`, `docs/model.md` and `docs/SPEC.md` are unchanged.

### Found (iteration 013: audit against the authors' code, post hoc)

- `docs/iterations/013-tdstdp-backbone.md` §8 (new) compares iteration 013 with `reference/td_stdp` and BindsNET 0.2.7. No repo code changed; the diagnostic runs live in the session scratchpad.
- **The E1 failure is a reproduction error.** The cause is the η scale:
  - the authors' traces carry $1/\tau_z$ (critic) and $1/(\tau_z\tau_q)$ (actor), which the paper's Eqs. 72 and 77 omit;
  - BindsNET adds w mV per spike, so weights convert by ÷13, not ÷1300;
  - the code's critic η is 0.005.
  - Our critic step was therefore 10× too small and our actor step 8× too large.
- **Two loop details caused the collapses:** our action is sampled from the previous observation's response, and $V' := 0$ is also applied on truncation, while the authors bootstrap there.
- **Post hoc test, 5 seeds:** with all fixes and their deterministic LIF, the run reproduces CK: 5/5 solved, $T_f$ 65.0 ± 7.4, $T_s$ median 185. On the branching engine the same fixes reach a return of 490.4 ± 9.8 with 1/5 solved.
- **E3 is affected too.** The authors' Adam step is 100× ours, and their episode counts pool the 16 copies.

### Added (reference code of Chung & Kozma 2020)

- `reference/td_stdp/`: a shallow clone of https://github.com/stephen-chung-mh/td_stdp (commit `a5cb7f4`), the paper authors' BindsNET code, used to compare against the iteration 013 implementation. `reference/` is gitignored; reference only, not imported by the project.

### Added (iteration 013 step 1: TD-STDP backbone of Chung & Kozma 2020 on the branching engine)

- **Code:**
  - `mapstdp/tdstdp.py` (new): the feed-forward TD-STDP actor-critic of Chung & Kozma 2020, a baseline and not Map-STDP. It holds the branching step without presynaptic normalisation and a SuperNeuroMAT twin, a true-LIF control (CK Eq. 1, or SuperNeuroMAT's subtractive leak), a closed-form block sum of the z/q traces and updates (block 1 = the per-ms rule), a block Adam with per-ms-matched moments, and a self-check.
  - `experiments/tdstdp.py` (new): closed loop, CLI and `--summary`. Results go to `tdstdp_*.json`.
  - `tests/test_regression.py` now also runs the `mapstdp.tdstdp` self-check.
  - Golden is bit-identical, 8 tests pass and ruff is clean.
- **Self-check:**
  - The numpy step equals SuperNeuroMAT spike for spike, both in branching mode and with a finite leak, and the closed loop on the `snm` engine reproduces the numpy engine's returns and W.
  - The block closed form matches the per-ms loop to $4 \times 10^{-16}$.
  - The STE in branching mode is exact per postsynaptic row (cosine 1.0000) but carries the post rate as a row gain (global cosine 0.9975 critic, 0.9685 actor).
- **Decisions and deviations** (`docs/iterations/013-tdstdp-backbone.md` §3):
  - Weight units: η is divided by 1300, from CK Eq. 1 and Eq. 44. The BindsNET reading of 13 destroyed the network in a debug run.
  - Batch 16 runs as 16 asynchronous env copies with 3000 episodes per copy.
  - LunarLander runs at B = 20.
  - Block Adam was changed to per-ms-matched moments after the first form collapsed 3/10 CartPole seeds. It passed a gate set in advance (MW p 0.14 against per-ms Adam).
  - The E4 rule uses the per-ms update RMS.
  - E3 and E4 were stopped at episode 50 and relaunched under these two fixes.
- **Results** (`python -m experiments.tdstdp --summary`):
  - E1 (CK CartPole, no Adam, 20 seeds) **fails its gate**: 0/20 solved, last-100 15.4 ± 16.4. The true-LIF control with Table 1 verbatim fails the same way (0/10).
  - E2 (+ per-ms Adam) learns: 377.8 ± 83.2, $T_f$ 123.0 ± 30.2 against CK's 57.0, and 1/20 meets CK's strict $T_s$.
  - Post hoc: LIF + Adam gives 464.8 ± 31.4 (branching 369.7 on matched seeds, MW p 0.006). SGD with the Adam-equivalent η learns and then collapses in 6/10 seeds. An init at rate 0.1 does not rescue E1.
  - E3 and E4 (LunarLander) are running locally, started 11:13 EDT, at least 8.5 h.
- **Not changed:** `docs/derivation.md` and `docs/model.md` (TD-STDP is a baseline, not the model). `CLAUDE.md` gains the Code and Commands lines for the new module and script.

### Decided (iteration 013: TD-STDP on the branching engine first; QIF deferred)

- **Result (preliminary, scratch only; no repo code):** the derivation §5 rule on the `mpr` rate backend (`mprK`: pairing count $J^\Lambda$, eligibility $\kappa_{ij} \Phi^K_{m(i)}$ with no presynaptic baseline, pairing-weighted structural baseline, kernel clipped at 0 for $\kappa$ and $T^{eff}$, eligibility scale 0.75 fixed before training) against `mpr` and flow v4, v4 flags, last-100 mean ± sd:

  | Task (seeds) | flow v4 | mpr | mprK | mprK vs mpr (Welch / MWU p) |
  | --- | --- | --- | --- | --- |
  | CartPole (0–19) | 487.6 ± 51.7 | 231.1 ± 103.2 | 190.9 ± 100.8 (n = 18) | 0.23 / 0.63 |
  | Acrobot (0–19) | −107.6 ± 17.5 | −123.9 ± 23.5 | −114.5 ± 27.2 | 0.25 / 0.05 |
  | MountainCar (0–9) | −131.8 ± 2.8 | −166.6 ± 35.1 | −159.0 ± 34.8 | 0.63 / 0.57 |
  | LunarLander (0–9) | 102.5 ± 94.1 | −6.0 ± 57.6 | 4.5 ± 34.4 | 0.63 / 0.62 |

  On frozen weights the new score's cosine with the exact policy gradient is 0.83–0.97 after removing the column rescaling (the old 4b on `mpr`: 0.07–0.42), but in closed loop it closes none of the CartPole gap and is not significantly better than `mpr` on any task. Confounds: the structural and homeostasis terms ran about 1.7× stronger, and `mprK` drives $W$ toward regimes where the rate solver needs fallbacks (5–25× more). CartPole seeds 15 and 17 were stopped at the user's direction (8.6 h, episodes 950 and 600) and are excluded. Scripts and results: session scratchpad `qif/kscore/`.
- **Decision (user):** the QIF migration postpones the hard part (recurrent modules) rather than removing it. Build the TD-STDP backbone [Chung & Kozma 2020] on the SuperNeuroMAT branching engine first, then tune QIF. Derivation §5 stays a Proposal.

### Changed (derivation rewritten: current state only, map equation only, linear-response base)

- **Docs:** `docs/derivation.md` rewritten from 1037 lines (about 19k words) to about 480 (about 9k words), at the user's direction. History lives here and in `docs/iterations/`, not in the derivation.
- **Cut:**
  - Appendix B (corrections) and every dated or "previously/corrected" note.
  - Iteration-by-iteration narrative and result tables; iterations are now cited in one clause where a design choice needs evidence.
  - §5's alternative description lengths (entropy rate, entropy production, cost-weighted map equation, Markov stability, SBM, predictive dissipation, information flow, latent-chain entropy rate). The structural term is the map equation only.
  - The lateral-inhibition score (old Eq. 8, §7).
  - The rejected latent-output alternatives and the withdrawn or removed variants: Eq. 1g′ (β set point), Eq. 4c′ (critic TD(λ)) and Eq. 4i (usage term).
  - The dual routing term (4a) is reduced to a short m1-only ablation, and the routing exemption is the stated choice.
- **Restructured:**
  - §2.1 now derives the walk as the network's linear response (Hawkes kernel $K = \mathrm{diag}(G) B$, $r = Kr + \mu$, walk $T^{eff} = K/m$, escape $1 - m_j$), with branching mode ($K = (1-\alpha)T$, $G = 1$) as the implemented special case that gives Eq. 1.
  - The new §5 (**Proposal**, not implemented) states Lemma K (K6), Theorem K3 (K7: pairing-weighted baseline, preconditioner $m_j$) and the policy score (K8–K9: cascade value $\Phi$, no presynaptic baseline), with their open conditions.
  - §7 is now Inhibition: code the E→E walk, with inhibition entering only through $G$ and $\pi$; the score keeps the pools.
  - Appendix A is compressed to short proofs and one check table. Its EM-split cosines are flow 0.85/0.92/0.55/0.92 and QIF 0.25–0.44, replacing the 0.977 small-network figure.
  - Lemma K and (K8)–(K9) were re-checked independently on a 12-neuron softplus network: FD error $9 \times 10^{-10}$ and $7 \times 10^{-9}$ relative.
- **Equation numbers unchanged** for every surviving equation (1, 1a–1e, 1g, 1h, 2, 3, 3a, 4, 4a–4e, 4c″, 4h, 5–7). The retired numbers (1g′, 4c′, 4i, 8) are not reused. New equations are labelled K1–K9.
- **References:** dropped works no longer cited. Added Hawkes 1971, Hawkes & Oakes 1974, Lambiotte & Rosvall 2012, Montbrió et al. 2015, Pernice et al. 2011, Trousdale et al. 2012, Tsodyks et al. 1997 and Ozeki et al. 2009 (the last two verified this session).
- **Other docs:**
  - CLAUDE.md: dropped the Appendix B sentence and the "seven alternatives" framing.
  - SPEC (user decision: map equation only): dropped the §6 candidate table and its scoring criteria, the "estimators" and "composite modulators" open items, the per-candidate metric and ablation, and milestone M5 (modulator comparison; M6 keeps its ID). The energy-proxy criterion moved into the energy-model open item. Citations of derivation §3.4 now point to §3; the branching mode now cites §2.1.
  - model.md: rows citing Eqs. 1g′, 4c′ and 4i now cite their iterations.
  - `.claude/skills/doc-edit/SKILL.md`: the Appendix B rule is replaced by "no history in the derivation".
  - No code changed.
- **Review fixes (snn-expert and neuroscientist reviews, same day):**
  - Notation: offspring matrix $K \to \Lambda$ (so $J^\Lambda$, $\Lambda^{pop}$), offspring mean $m_j \to \nu_j$, Theorem K3 → Theorem K, latent-output barrier weight $\beta \to \beta_{LA}$ ($D_{LA}$), entropy-target gain and fraction $k_H$, $\phi_H$ (logs in nats). $K$ is now only the number of modules. Equation labels unchanged.
  - Corrections: the exact score is $\kappa_{ij} \Phi_i$ (neuron level); the module table $\Phi^K_{m(i)}$ is an approximation (cosine 0.83–0.97). Appendix A: baseline-as-renormalisation is $0.025 \lambda$ (was $2.5 \lambda$), (4h) MC correlation $\ge 0.9997$ (was 0.9998); §4.6 untrained (4b) cosine 0.98–0.998. "$m_j$ is a continuation probability" is replaced by "$\nu_j$ offspring on average". The Markov-stability remark, the Theorem 2 proof (base-2 logs) and the §6 hardware caveat are reworded; the non sequitur in §9 item 1 is dropped.
  - Added: one line on TD normalisation (model v4, `--td-norm`) after (4c); a note that the (4c'') label is kept.
  - Structure: new §2.5 "Action readout" ((1d), race, (1e), (1g), (1h), policy limits); the world model stays in §2.4. §4.5's second paragraph is split into three.
  - Biology (neuroscientist): the hypothesis that crossing costs more is labelled as a hypothesis; (K1) is the per-neuron QIF transfer "as in" the MPR mean field; the race burn-in cites Frank 2006 as a "hold your horses" STN signal; the per-neuron baselines $\bar{g}_j$, $\bar{h}_j$, $\bar{g}^{eff}_j$ are marked as abstractions (Royer & Paré is input-side only); the action broadcast is "plausible but untested"; $\Phi^K$ is not a plausible neuromodulator payload, so it is treated as a peripheral computation; §6 restores the backward (dendritically normalised) walk sentence and adds astrocyte domains [Bushong et al. 2002] and a presynaptic-locus factor; §7's E→E coding is a **Proposal**, with feed-forward PV inhibition [Gabernet et al. 2005] as the caveat. Added Gabernet 2005 and Bushong 2002 to the references.
  - Pointers: SPEC (controller recurrence) and model.md (Initialisation) now cite §2.5; SPEC's Locality convention cites §6 (Biological locality). Comment and docstring only: `spiking.set_weights` cites "§2.2, constant presynaptic output", `core.modulator` says exempt is the agent's default, and `m3` says "former Eq. 4i". `spiking.pooled` and `spiking.race` cite §2.5 for the race; SPEC §1 cites §2.4–2.5 for the readout.

## 2026-10-22

### Added (iteration 013 prep: QIF engine, Proposal; gate failed)

- **Code:** `mapstdp/qif.py` (new): quadratic integrate-and-fire engine in θ-form (τ 10 ms, dt 0.1 ms, Lorentzian η per head, exponential synapse τ_s 2 ms, delay 1 ms, fixed per-module PV pools, Bernoulli input pulses, warm start, 1 ms raster bins), numba kernel with a numpy twin, Montbrió–Pazó–Roxin mean field (`mpr`), Brian2 reference (`brian_frame`, lazy import) and self-check. `experiments/m2.py` `Config.engine="qif"`, `Config.qif`, `BOUND_CAL_QIF`; `experiments/m3.py` `--engine qif`, `--q-<field>` flags, file token `qif[...]`, fidelity columns `1-corr_bias`, `1-corr_row`, `1-corr_Jhat`, `S_mod`. Default path unchanged: golden bit-identical, 7 tests pass, ruff clean.
- **Env:** `environment-brian2.yml` (env `map-stdp-brian2`, numpy < 2.4, brian2 2.9) for the reference check only. Brian2 2.9.0 does not import under numpy 2.4 (`ndarray.ptp`), and pinning `map-stdp` to numpy 2.3 changes the spiking golden case, so the training env is unchanged.
- **Design** filtered by the neuroscientist; changes made during calibration: Δ 0.25 → 0.05, η truncated at 10, A_ext 3 → 5, fresh noise σ 1, PV pools on by default.
- **Results (preliminary, scratch only):** numba = numpy = Brian2 spike for spike. Fidelity on initial CartPole networks (16 heads): L1(π̂, π) 0.22 and 1 − corr(κ, J) 0.46, against 0.09 and 0.24 for branching mode. v4 smoke (CartPole, 5 seeds × 300 episodes): QIF 20.7–26.0 (random ≈ 22), branching 216–271. With flow π and J, the QIF race learns (63–99); with QIF π̂ and κ, nothing learns, because the structural rule dissolves the modules. Neuron-level MPR predicts the module shares better than the linear flow (L1 0.10 against 0.20).
- **Phase A, tune before rebuild (preliminary, scratch only):** qif-path scale fields `QIFParams.lam_scale`, `eta_scale`, `eps_h_scale` (default 1; flags `--q-lam-scale`, `--q-eta-scale`, `--q-eps-h-scale`, `--q-tau`). Matching update RMS to branching does not help (CartPole 20–25). A fidelity sweep (600 configs) finds no point with L1 ≤ 0.10 and corr(E[κ], J) ≥ 0.9; the best, at 0.19 and 0.80, needs strong fresh noise (σ 3.4, τ 3.5 ms), far from the Montbrió–Pazó–Roxin-exact regime. There, CartPole reaches 44–56 (branching 216–271) and Acrobot −291 to −340 (branching −137 to −162). Decision: QIF needs the rebuild (Phase B: π from rates, κ against G·T·r̂, the 4b score on r̂).
- **Phase B steps 4–5 (preliminary, scratch only):** `qif.rates` (neuron-level MPR fixed point: damped Newton, continuation fallback, returns per-neuron gain G), `qif.MPR_POINT` (σ 0, η̄ −2, Δ 0.5, gain 20, w_ie 15, rate 1, 32 I neurons per pool, τ_s 20), and `m2.Config.backend = "mpr"` (π = r*/Σr*, κ = T·diag(r*), Eq. 1d/1e on module shares). Solver: 6–12 ms per frame, 0 failures, fallbacks below 0.1% of frames. v4 flags, 5 seeds, against flow v4: CartPole 261 vs 500, MountainCar −159.6 vs −132.2, Acrobot −118.2 vs −96.3, LunarLander −24.0 vs 82.4. Gate (within 10%) failed on every task. The cosine of the 4b score with the true policy gradient drops on trained networks from 0.63–0.96 (flow) to 0.07–0.25 (MPR), so the module score, not κ, is the binding problem.
- **Phase B step 4 gates (passed):** L1 of module shares, rates against QIF spikes, is 0.027 / 0.064 / 0.028 / 0.041 on CartPole / MountainCar / Acrobot / LunarLander (per-neuron corr 0.92–0.95). From 10 starts × 20 random W per task, plus trained W, every solve converges to the same fixed point. Self-checks: residual below 1e-9, and G matches finite differences. A gain-aware credit (a K × K module resolvent with module-mean gains, times each neuron's own gain G_i) reaches cosine 0.94–0.97 with the exact gradient, against 0.07–0.25 for 4b (scratch `psi.py`).

### Removed / Changed (iteration 013 prep: prune unadopted variants, `m2.Config`)

- **Removed (not adopted; code, CLI flags and golden cases):** models v1 and v2 and the dual routing term (Eq. 4a) in `experiments/m2.py` (`--model`, `--routing dual`, `Agent.mu`/`q_star`, `EPS`, `ETA_SPK`, `ETA_MU`, `Q_STAR_FRAC`); the β ramp (`--beta-end`, `--beta-ramp`), mean reversion (Eq. 1g′: `--beta-eps`, `--beta-set`, `--beta-decay`), trace signal (`--beta-signal`), step cap (`--beta-step-cap`, `--beta-step-shape`) and score-unit ablation (`--score-unit`); the TD(λ) critic (Eq. 4c′, `--critic-lambda`); the `rich` and `net` critics (`--critic`); the usage term (Eq. 4i: `--c-use`, `--use-tau`, `core.usage_score` and its self-check); the MSPRT readout in `experiments/ceiling.py`. `mapstdp/core.py` keeps `modulator(routing='dual')` for `experiments/m1.py`. Golden cases `flow-dual-0.05-v1`, `flow-none-0.05-v2`, `spiking-exempt-0.0-v1` deleted from `tests/golden.json`; nothing regenerated.
- **Changed:** `m2.Agent(cfg)` takes one frozen `m2.Config` dataclass (shared by the m2 and m3 CLIs, `m3.fidelity`, `ceiling`, `m2_viz`); `m2.run(cfg, episodes, name=None)`. `Agent.episode` split into `task_step` and `track`; the KL readout check moved to `Agent.race_fidelity`; `regulariser` became `entropy_term` (Eq. 4h only).
- **Result files:** names unchanged for every kept configuration. New files no longer carry the pruned keys `mu`, `q_star`, `beta_end`, `beta_ramp`, `score_unit`, `params.eta_mu`, or `use_rms_ratio`/`ubar` in evals. `m2 --summary`, `m3 --summary`, `m3 --ablations` and `ceiling --summary` print byte-identical output over the committed results (v1/v2, dual, MSPRT and pruned-variant files included).
- **Behaviour:** remaining golden cases (`flow-exempt-0.05-v3`, `spiking-exempt-0.05-v3`, `spiking-readout`) bit-identical; v4 LunarLander, iteration-012 MountainCar, spiking 2-head and diffbound runs reproduce the original code's returns exactly. ruff clean; pyright 42 → 33 errors.
- **Docs:** CLAUDE.md `m2.py` bullet updated for `m2.Config` and the pruned model versions; dual routing marked as `m1.py`-only. Decision (user): no no-latent control or new golden fingerprints before the prune; the no-latent ablation is deferred to the latent fixes and sweeps of iteration 013, and code still needed as reference (flow training, ceilings, ablations) is kept until superseded.

## 2026-10-21

### Added (iteration 012: encoders, separate critic, entropy terms)

- **Code** (all default off; golden 9/9 bit-identical; ruff clean): `mapstdp/core.py` Fourier basis (`fourier_coeffs`, `fourier_features`, `fourier_stimulus`), `action_inflow`, `action_entropy`, `entropy_score` (Eq. 4h), `usage_score` (Eq. 4i), with self-checks. `experiments/tasks.py` encoder specs `g<n>[s<σ>]`, `f<order>c<conj>[h][b]`, `FOURIER_BOUNDS`, `with_encoder`, `critic_layer`. `experiments/m2.py` `Agent(encoder, critic_layer, c_ent, c_use, use_tau, ent_target)` and diagnostics. `experiments/m3.py` `--encoder`, `--critic-layer`, `--c-ent`, `--ent-target`, `--c-use`, `--use-tau`; tokens `enc`, `crit`, `ce`, `cu`, `ep<E>`. `experiments/ceiling.py` `--encoder`, `--no-train`, an underflow guard for $p^\beta$.
- **Results:** about 680 `m3_*` runs (pilot seeds 0–4, pre-registered seeds 0–19, fresh seeds 20–39, LunarLander 3000-episode runs) and about 230 `ceiling_*` runs (10 CEM runs per cell).
- **Docs:** `docs/iterations/012-encoders-critic-entropy.md`; derivation §4.6 (Eq. 4c'' separate critic, Eqs. 4h and 4i, Proposals), Appendix A (numerical checks for 4h and 4i: finite differences to $2 \times 10^{-10}$, Monte Carlo correlation ≥ 0.9998, full-resolvent cosine 0.92–0.999), Appendix B item 17; references Chung & Kozma 2020, Foster et al. 2000, Haarnoja et al. 2018, Lansink et al. 2009, O'Doherty et al. 2004, Tervo et al. 2014; model.md rows, parameter and history; SPEC M3 row; CLAUDE.md iteration range and the best MountainCar command.
- **Findings (40 seeds):** MountainCar −100.7, 38/40 at −110 (12 × 12 encoder, 24 × 24 separate critic, entropy target); Acrobot −92.8, 35/40 (separate Fourier critic); CartPole 500, 40/40, median 296 episodes to solve (Fourier encoder and critic); LunarLander unchanged. Confirmed on fresh seeds 20–39.
- **Decision:** no model version bump; the per-task configurations stay Proposals. The entropy term's setting deviates from the neuroscientist's calibration (destructive at 0.1–0.3 of the task RMS); sign-off pending. Next: iteration 013 on a new branch, a TD-STDP backbone with communities.

### Added (iteration 011: flow diagnostics)

- **Docs:** `docs/iterations/011-flow-diagnostics.md` (scratch code only; no repo code or results changed). model.md history row; SPEC M3 row; CLAUDE.md iteration range.
- **Findings (preliminary):** a sharper readout does not help on frozen v4 weights (argmax: MountainCar −133.6, Acrobot −107.6). MountainCar is limited by the encoder and the critic: 12 × 12 controller grid + separate critic on a 24 × 24 RF layer gives −101.9 ± 6.0, 18/20 at −110 (v4 −131.0, 0/40). Acrobot's bad seeds entrench the zero-torque action early; masking it on frozen weights gives −97.4 (26/40 at −100).
- **Corrections:** the 6 × 6 MountainCar encoder's CEM ceiling is about −114 at 10 CEM runs, not −110 (iterations 005, 009, 010 used 3 runs); iteration 009's "the critic is not the MountainCar bottleneck" is withdrawn (009 record, derivation §4.6).
- **Decision (user):** iteration 012 tests a 12 × 12 encoder + separate 24 × 24 critic on MountainCar, a Fourier-basis encoder [Chung & Kozma 2020] and a separate critic on the other tasks, and an entropy bonus on all tasks.

### Added (iteration 010: decision-bound readout)

- **`mapstdp/spiking.py`:** `diffbound()`, a race on count differences to a bound (Eq. 1h), optionally capped at one spike per module, head and step; self-checks (deterministic raster, cap, $K = 2$ gambler's-ruin Monte Carlo). **`experiments/ceiling.py`:** readouts `beta<B>`, `db<T>`, `msprt<T>c<C>`, `--nmax`; existing readouts draw unchanged. **`experiments/m2.py`:** `Agent(readout='diffbound')`, spiking only, $\theta$ from $\beta$ through `BOUND_CAL`. **`experiments/m3.py`:** `--readout`, token `dbc`. All default off; golden 9/9 bit-identical, ruff clean.
- **Results:** level-1 ceilings on MountainCar and Acrobot for Eq. 1e at $\beta$ 4/8/16, (1h) and MSPRT at $\theta$ 2–16 (3 CEM seeds); spiking pilot at 16 heads on CartPole and MountainCar (race, (1h) with adaptive $\beta$, (1h) at fixed $\beta$ 4 and 8; seeds 0–4).
- **Docs:** `docs/iterations/010-decision-bound.md`; derivation §2.4 (Eq. 1h, Proposal, spiking only; $\beta$ read as a decision threshold rather than noradrenergic gain; burst overdispersion $\beta_{eff} \approx \theta / F$), §9 item 14, Appendix A (numerical checks for 1h: $K = 2$ identity to $2 \times 10^{-15}$, expected count to $10^{-12}$, $K = 3$ KL ≤ 0.0065, burst formula), Appendix B item 16; references Bogacz et al. 2006, Bogacz & Gurney 2007, Bogacz et al. 2010, Cavanagh et al. 2011, Feller 1968, Lo & Wang 2006, Usher & McClelland 2001, Wald 1947; model.md row and history; SPEC M3 row; CLAUDE.md iteration range.
- **Decision:** flow unchanged (the gate failed: (1h) and MSPRT ceilings equal Eq. 1e's at matched $\beta$). Spiking (1h) kept as a Proposal, not adopted: MountainCar −132.2 against −177.0 for the race (flow parity), CartPole 153.9 against 298.3. MSPRT and the attractor fallback dropped.
- **Note:** the committed iteration-005 $\beta = 4$ ceiling at seed 0 does not reproduce on this machine (MountainCar −115.2 against −121.6); training return differs too, so the drift is machine level. The committed files are kept.

### Added (iteration 009: credit horizon, negative)

- **`experiments/m2.py`:** `Agent(critic_lambda, gamma)`: the critic runs through a presynaptic trace $z$ (Eq. 4c'), identical to Eq. 4c at $\lambda = 0$ (golden 9/9, full v4 reruns identical). **`experiments/m3.py`:** `--critic-lambda`, `--gamma`; tokens `cl<λ>`, `g<γ>`; `nondefault()` catches `tau_e`.
- **Results:** 280 new `experiments/results/m3_*` runs (MountainCar and Acrobot arms, seeds 0–19; $\tau_e = 10$ also on fresh seeds 20–39 and on CartPole and LunarLander).
- **Docs:** `docs/iterations/009-credit-horizon.md`; derivation §4.6 (Eq. 4c', tested, not adopted), Appendix A (numerical checks for 4c': $\lambda = 0$ bit-identical; forward-view equivalences to $10^{-15}$; update-time normalisation off by up to 12% on Acrobot), Appendix B item 15; references Pan et al. 2005, Sutton 1988; model.md rows; SPEC M3 row.

### Changed (decisions, iteration 009)

- **v4 kept.** The `neuroscientist` filtered the proposals first (critic TD($\lambda$), $\tau_e = 10$, $\gamma = 0.995$ accepted; $\tau_e = 30$ rejected as beyond the dopamine-trace evidence; encoder expansion lowest priority). No arm survived the pre-registered Bonferroni tests, $\tau_e = 10$ did not replicate on fresh seeds and cost CartPole (295.6 against 487.6), and the critic trace with update-time normalisation diverged on Acrobot (20 of 20 seeds).
- **The critic is not the bottleneck** (V correlates 0.99 with the true value on MountainCar). v4 sits 10–25 below the ceiling of its own Eq. 1e readout, so the flow gap is the learned weights. Next, iteration 010: a decision-bound readout, mainly for a sharpened spiking readout.

### Added (iteration 008: β's gain on the task step)

- **`experiments/m2.py`:** `Agent(beta_step_cap, beta_step_shape)`: the Eq. 4b increment uses $\beta_\eta(\beta)$ in place of $\beta$, with shapes `min`, `sat` ($c\beta/(c+\beta)$), `invu` ($\beta e^{1-\beta/c}$) and `hyb` (identity to $c$, logarithmic above); the direction is unchanged (cosine 1). Exclusive with `score_unit`. Defaults off; `tests/golden.json` bit-identical (9/9).
- **`experiments/m3.py`:** `--beta-step-cap`, `--beta-step-shape`; tokens `sc<c>[sat|iu|hy]`.
- **Results:** 960 new `experiments/results/m3_*` runs at $\kappa_\beta = 0.01$: v4 and v5 at seeds 10–39, six step shapes at seeds 0–19, and the pre-registered confirmation (v5, min 4, sat 16, hyb 4) at seeds 20–39.
- **Docs:** `docs/iterations/008-beta-step-gain.md`; derivation §2.4 note; model.md and the 007 record corrected; SPEC M3 row.

### Changed (decisions, iteration 008)

- **No step shape adopted.** None of the four pre-registered tests on fresh seeds 20–39 survives Bonferroni; min 4's CartPole effect from seeds 0–19 reversed (445.6 against 499.4). `invu` rejected (MountainCar seeds that never learn, worse Acrobot).
- **v5 withdrawn; the documented model is v4 again** (the user's decision). At $\kappa_\beta = 0.01$ and n = 40, v5 is not better than v4 on any task (LunarLander nominally worse, 52.4 against 81.3, p = .13); the 1g′ recovery evidence rested on the $\kappa_\beta$ 0.0099 runs. Eq. 1g′ stays in the derivation as a tested, unadopted Proposal; its flags remain.
- **Seed noise:** the $\kappa_\beta$ 0.0099 control is replaced by more seeds at $\kappa_\beta = 0.01$ (the user's decision). v5's CartPole mean is 500.0, 399.4 and 499.4 over seeds 0–9, 10–19 and 20–39; report at least 20 seeds per arm.
- **Neuroscientist review:** a plasticity gain that saturates while response gain does not is a plausible, untested abstraction (dissociable, non-monotone neuromodulator dose-responses: Arnsten 2009, Vijayraghavan et al. 2007, Salgado et al. 2012, Seol et al. 2007); smooth saturation preferred to a hard cap; "not a gradient" corrected to "a $\beta$-dependent step size on the same gradient".

### Added (iteration 007: robust adaptive β)

- **Model v5** (`docs/model.md`): v4 plus `--beta-eps 0.005 --beta-decay 1000`, the experience-dependent set point of Eq. 1g′ (Proposal). $\beta$ now recovers after a collapse (CartPole, 20 runs: 5 of 5 drops recover, against 1 of 3 in the v4 family). Return is unchanged: pooled over $\kappa_\beta$ 0.01 and 0.0099 (n = 20), CartPole 489.8, MountainCar −130.4, Acrobot −106.7, LunarLander 92.3, against 466.0, −132.4, −106.9 and 100.6 (all p ≥ .05). Code defaults unchanged; `tests/golden.json` bit-identical (9/9).
- **`experiments/m2.py`:** `Agent(beta_eps, beta_set, beta_decay, beta_signal)`, `beta_move()`, `beta_frame()` (per-frame trace signal), constants `BETA_TF`, `BETA_TS`, `BETA_TH`; the result JSON records the new keys only when non-default.
- **`experiments/m3.py`:** `--beta-eps`, `--beta-set`, `--beta-decay`, `--beta-signal`; tokens `re<ε>[s<β₀>][d<τ_D>]` and `tr`; `nondefault()` updated.
- **Results:** 440 new `experiments/results/m3_*` runs: candidates C1–C4 and v5, the $\kappa_\beta$ 0.0099 noise control, and the iteration 006 rows extended to 10 seeds.
- **Docs:** `docs/iterations/007-robust-beta.md`; derivation §2.4 (Proposal Eq. 1g′, open issues updated), §9 item 14, Appendix A (numerical checks for 1g′ from `/derivation-check`: reduction to 1g bit for bit on 33 119 updates, fixed point and recovery rate exact, trace signal zero-mean when stationary), Appendix B item 14; three references (Abraham & Bear 1996, Eisenberger 1992, Niv et al. 2007); SPEC M3 row; CLAUDE.md v5 command.

### Changed (decisions, iteration 007)

- **v4's numbers are partly a lucky draw.** Changing $\kappa_\beta$ by 1% gives CartPole 432 and Acrobot −114. Variants are now compared against the pooled v4 family (n = 20).
- **Correction to iteration 006:** at n = 10, dividing $\beta$ out of the task step gives LunarLander 64.0 (p = .38 against v4), not 7.0; "most of LunarLander's gain is the step size" no longer holds. The 006 record is annotated.
- **A strict running maximum as the set point is rejected** (`neuroscientist`): it never decreases, so it would block disengagement after a change in task. A maximum leaking over $\tau_D = 1000$ episodes replaces it, with the same recovery.
- **Rejected:** a leaky mean of $\delta / \mathrm{RMS}(\delta)$ as the episode-free signal ($\beta$ stays at 1 on MountainCar and Acrobot); two-sided reversion toward a slow average; a $\beta$ floor of 2.
- **Not adopted:** a fixed two-sided set point ($\beta_0 = 8$; narrows LunarLander's spread, no gain in mean, hand-set prior); the per-frame outcome-reset trace (the signal works, but no single per-frame rate fits all tasks).
- **Collapses live in the weights:** frozen-checkpoint tests show re-raising $\beta$ helps only while the weights still hold a good policy. The next lever is the $\beta$ step gain at high $\beta$, not the $\beta$ rule.

### Added (iteration 006: M3 reward)

- **Model v4** (`docs/model.md`): the M3 configuration `--p-cc 0 --beta-kappa 0.01 --beta-max 16 --eta 5` (with `--td-norm 0.04`, `--la-beta 0.3`). Flow return over 10 seeds: CartPole 272.5 → 500.0, MountainCar −184.2 → −131.8, Acrobot −128.8 → −100.1, LunarLander −37.8 → 102.5 (± 94, one seed collapsed). Code defaults unchanged, so v1–v3 and `tests/golden.json` reproduce bit for bit (9/9).
- **`experiments/m2.py`:** `Agent(p_cc, beta_end, beta_ramp, beta_kappa, beta_max, score_unit)` and `ablate={"critic": "net"}`; the result JSON records them.
- **`experiments/m3.py`:** `--lam`, `--p-cc`, `--beta`, `--beta-end`, `--beta-ramp`, `--eta`, `--beta-kappa`, `--beta-max`, `--score-unit`, `--critic net`; file-name tokens for each; `summary()` keeps the variants out of the baseline groups.
- **`experiments/tasks.py`:** `CartPoleFull` (all four observations; a negative result).
- **Results:** about 515 `experiments/results/m3_*` runs, including the map-term ablation of v4 (`lam0`).
- **Docs:** `docs/iterations/006-m3-reward.md`; derivation §2.4 (credit limit, Proposal Eq. 1g), §4.6, §9 items 13–14, Appendix A (numerical checks for the credit limit, from `/derivation-check`: exact score to $4.6 \times 10^{-11}$, cosines and block decomposition) and Appendix B item 13; three references; SPEC controller entry and M3 row.

### Changed (decisions)

- **Controller built without recurrence ($p_{cc} = 0$).** Map-STDP's module-level task term cannot credit controller→controller synapses (cosine of Eq. 4b with the exact score 0.55–0.85 on trained weights, 0.83–0.99 without them). The controller is now described as a feed-forward encoder, not cortex or a workspace (`neuroscientist`).
- **Correction:** the $0.997$ cosine of Eq. 4b with the true score holds only on untrained weights.
- **Proposal: adaptive $\beta$ (Eq. 1g)**, a causal utility-trend gain, one setting for all tasks. **The time-based $\beta$ ramp is rejected as a model component** (the user's decision): it needs the run length in advance. It stays only as a labelled oracle reference flag.
- **Rejected:** a $(\mathcal{C}, \mathcal{C})$ cost penalty (not a description-length gradient), the Eq. 4f resolvent credit on reactive tasks, an entropy-gradient term, the rich and network critics, `CartPoleFull`.
- **Deferred:** a matched sharpened spiking readout (a race to $k$ spikes reaches only $\beta_{eff} \approx 1.5$).
- **Encoder:** kept fixed through M3 (`neuroscientist` review of visual-cortex models).
- **The map term still earns its place in v4.** With `--lam 0` (10 seeds): CartPole 500.0 → 396.4 (p = .05), MountainCar −131.8 → −187.1 (p = 1e-4; 8 of 10 seeds never reach the goal, so $\beta$ never rises), LunarLander 102.5 → 11.0 (p = .02); Acrobot unchanged (−100.1 against −98.3).

## 2026-10-20

### Changed (CLAUDE.md)

- **CLAUDE.md** gains three things a fresh session would otherwise learn by hitting them:
  - **A "Traps" list:** RNG-call order and the golden tests, no `zip(strict=True)`, the `m2_*` result-file namespace, and silent spiking modules.
  - **The unmerged `m4-tmaze` branch:** a T-maze M4 prototype whose iteration number 004 collides with `main`'s.
  - **The symptoms of launching outside the `map-stdp` env:** the KaTeX hook errors and CodeGraph does not connect.

### Added (iteration 005: M3 tasks, spiking heads, learner limits)

- **M3 tasks:** `experiments/tasks.py` registry (CartPole, MountainCar, Acrobot, LunarLander): module sizes, encoder, reward, episodes. `core.rbf_stimulus` (stable conjunctive RFs at arbitrary centers). CartPole encoder and reward are bit-identical to M2.
- **`experiments/m3.py`:** per-task runs on either backend, `--summary`, `--fidelity`, `--ablations`; flags `--heads`, `--engine`, `--td-norm`, `--eta-tau`, `--wmin`, `--tau-e`, `--critic`.
- **Proposal: parallel spiking heads** (`spiking.tile_heads`, `head_groups`, `pooled`; `m2.Agent(heads=H)`): H replicas on shared W, spikes pooled for $\hat{\pi}$, $\kappa$ and one race. $\hat{\pi}$ error against flow falls as $1/\sqrt{H}$ (0.35 → 0.089 at H = 16, CartPole). The return sweep has not been run.
- **Batched spiking engine** `spiking.run_heads`: raster-identical to SuperNeuroMAT (new self-check), 7–13× faster at H = 16. M3 default; M2 stays on SuperNeuroMAT. `spiking.set_inputs` writes SuperNeuroMAT's input array directly; `build` pins `sparse = False` (the jit backend is dense-only).
- **Proposal: TD normalisation** `--td-norm s`. At $s = 0.04$ it fixes the Acrobot collapse (−374 ± 156 → −128 ± 7); neutral on CartPole, MountainCar, LunarLander.
- **`Agent.diagnostics()`** logged with every eval (TD error, critic range, weights at bounds, policy entropy and sharpness). `Agent(ablate=...)` for learning-rule ablations.
- **`experiments/ceiling.py`:** policy-class ceiling by CEM. Free $R$ with Eq. 1d reaches CartPole 465, MountainCar −150, Acrobot −90 against learned 205, −183, −133 (seed 0); sharp readouts add more.
- **Results:** `experiments/results/m3_*.json` (flow, all tasks, ± `--td-norm`, CartPole and Acrobot ablations), `ceiling_*.json`, `m3_fidelity.json`.
- **Environment:** `box2d-py`, `pygame` in `environment.yml` and the `map-stdp` env.
- **Docs:** `docs/iterations/005-m3-tasks-heads.md`; `docs/model.md` rows (M3 stimulus, heads, engine, M3 reward, TD normalisation) and history; CLAUDE.md code and commands; SPEC M3 row.

### Fixed

- **Frame count in `m2.run`** was `sum(returns)`, wrong for negative-reward tasks; now counted per frame. CartPole results unchanged.

### Changed (decisions)

- **Diagnosis:** Map-STDP converges to a poor $R$ (the actions × controller matrix of the flow policy); the network can express much sharper votes, and Eq. 1d's stochasticity is a second cap. Latent memory is not the bottleneck on these fully observed tasks.
- **Rejected as fixes** (CartPole, Acrobot, 5 seeds): constant or slower task-rate annealing, a higher weight floor (it hurts: pruning to the floor is how the policy sharpens), longer eligibility, a richer critic.
- **Next:** an ideal actor-critic on $R$ with the same signals, then the spiking heads sweep.
- **Known issue:** `tests/golden.json` fails on the Mac workstation even at a clean HEAD (last-bit BLAS differences against the cluster); not regenerated. This iteration was checked against fingerprints of HEAD generated on the same machine: bit-identical.

## 2026-10-06

### Added (iteration 003: latent sinks)

- **Finding:** in the trained v3 agent, latent modules are flow sinks. They keep 0.98 of their flow, pass 0.002 to the actions and hold about half the activity.
  - **The task term causes the latent→action cut,** which also happens at $\lambda = 0$.
  - **The map term seals latents against the controller and each other,** which is where its CartPole benefit comes from (184 with the map term on actions only, against 287).
- **Proposal: latent-output barrier, Eq. 4e** (`mapstdp/core.py` `latent_output_barrier`; `experiments/m2.py --la-beta`, default 0).
  - **Derivation check:** finite differences to 9e-10, Monte Carlo 0.9999. A new self-check covers it.
  - **Cost on CartPole:** flow 274 ± 24 at $\beta$ 0.15 and 279 ± 22 at 0.3; spiking 171 ± 9 and 155 ± 9, against 287 and 184.
  - **Effect:** the latent→action share goes from 0.002 to 0.012–0.029. It regrows from sealed weights, and cross-module traffic barely rises.
  - **Confirmation:** the 20 runs through the repo CLI reproduce the prototype exactly. `pytest`: 9/9, v1–v3 bit-identical.
- **`docs/derivation.md`:**
  - **§4.5:** the diagnosis, (4e), the rejected alternatives, and why the (4a) dual diverged (its target was infeasible once modules seal).
  - **Appendix A:** the checks.
  - **Appendix B:** item 12.
- **`docs/model.md`:** a "Latent output" row and the $\beta_{LA}$ parameter.
- **`docs/SPEC.md`:** the ablations list.
- **`docs/iterations/003-latent-sinks.md`:** the record.

### Fixed (replay page)

- **Walk hops were overdrawn.** The page drew a hop for every synapse whose pre spiked at $t - 1$ and whose post spiked at $t$. That gave 2.0 hops per spike, and 57% of spikes got several parents. Stimulus input spikes got spurious parents too. The cross-module share came out at 0.31, against a true value of about 0.22.
  - **Fix:** `sp.run_frame(..., return_ext=True)` now returns which spikes were stimulus input. `m2_viz.walk_hops` gives every other spike exactly one parent, sampled in proportion to $T_{ij}$ among the spikes one step earlier. That is the exact cause posterior in branching mode.
  - **Check:** input share 0.212 against $\alpha = 0.2$, 0.788 hops per spike, every spike accounted for, and a cross-module share of 0.220 against 0.215 at the flow level.
- **The page now:**
  - marks input spikes with a ring;
  - draws every cross-module synapse with opacity proportional to $\sqrt{T}$, where before it showed only those with $T > 0.04$, which hid the weak latent→action links;
  - counts each frame's hops: total, between modules, latent→action, and input.
- **The replay uses the latent-output barrier** (`m2_viz --la-beta 0.3`; seed 4 is the best of 4, with last-100 training return 180). Its 366-frame episode has 3.7% of latent hops ending in an action module. Republished to the same artifact link.

### Changed

- **Decision reversal: the exit-floor Proposal is rejected.** Its dual enters a limit cycle, and its restored exit goes to the controller, not the actions (return 126–196).
- **Also rejected:**
  - the latent→action dual (limit cycle);
  - exemptions into the action modules (no effect, cannot regrow);
  - occupancy homeostasis (return 108);
  - a fixed set of output neurons per latent (exits to the controller);
  - an $M^{\ast}$ clip on latents.
- **Carried to M4:** credit through the latents and nonlinear carry-over. A $\dot{\theta}$-hidden CartPole test was flat for every variant (44–52).

## 2026-10-05

### Changed (expert review of the cleanup)

- **Review:** `/expert-review` of the cleanup diff by `snn-expert`, `superneuro-expert` and `neuroscientist`. All three found the cleanup numerically safe; `superneuro-expert` ran 20 spiking episodes on the old and new code and got identical frames, identical weights and the same 3.3 ms per frame. Applied, all names, docstrings and comments only; `pytest` 9/9 bit-identical and `ruff` clean afterwards:
  - **Renames in `experiments/m2.py`:** `local_plasticity` → `pre_td_updates` (all three reviewers: it is neither more local than the task term nor purely plasticity, and it mutates W and μ); `td_step` → `critic_step` (the task weight update stays in `episode`).
  - **`experiments/m2.py` docstrings:**
    - `pre_td_updates`: its update order, which the golden file locks;
    - `episode`: the task update lands one walk late;
    - `frame`: what κ and the action are in each backend;
    - `Agent`: the dual and the annealing;
    - `evaluate`: in spiking it consumes the training RNG.
  - **`mapstdp/spiking.py`:**
    - `pi_hat`: cited as the batch form of Eq. 2, replacing the bare §2.2 citation;
    - `run_frame`: what `reset()` clears;
    - `build` and `set_weights`: synapses are fixed at build time;
    - `pairings`: now says `lag`.
  - **`mapstdp/core.py`:**
    - `homeostasis_update`: marked as presynaptic conservation, an abstraction;
    - `flow`: a comment links `pi` to the iterates $x_k$.
  - **Not applied:** a separate evaluation RNG. `evaluate()` in spiking draws from the training RNG, so the logging interval changes trajectories. Fixing that would change `tests/golden.json`, so it is documented and deferred.

### Changed (docs aligned with the code; found by the codebase-cleaner)

- **`docs/derivation.md`:**
  - **Eq. 4c** now states the critic the code uses since iteration 001: $V = u \cdot \pi_{\mathcal{C}} / p_{\mathcal{C}}$ over controller neurons, with NLMS, reward −1 on termination and bootstrapping on truncation. The module-level critic is recorded as the weaker alternative (160 against 195, preliminary, iteration 001).
  - **§6 table** and **§9 item 13** are updated to match.
  - **§8 step 2:** the M2 frame is cold-started, $k = 100$ (was "about $4 \tau / \alpha$, warm-started").
  - **§8 step 5:** reordered to match the code. The pairings are the covariance count (was "causal minus acausal"), and the TD error and the annealed $\eta_e$ step come after the next frame's walk.
  - **§9 item 2:** the mean-field assumption needs the covariance count, not the balanced count. The note that it is untested in spiking is replaced by the M1 cosine of 0.992.
- **`docs/SPEC.md`:** the C2 critic line is updated to match Eq. 4c.

### Changed (code cleanup)

- **Behaviour-preserving cleanup** by the `codebase-cleaner`, with no change to numerics, RNG-call order, CLI flags, result-file names or JSON keys. `pytest -q tests` passes before and after (9/9, `golden.json` untouched). `m2.summary()` output and a short `m2.run` (flow and spiking, results written to a scratch directory) are byte-identical to the baseline.
  - **Lint:** `ruff check .` goes from 21 findings to 0. Long lines are wrapped (`m1.py` keeps the same AST), imports are sorted (`m2_viz.py`, `fdcheck.py`), `pairings` uses `lag` instead of `l`, the `core` self-check drops its semicolons and binds `beta` in `logP`, and `fdcheck.map_L` uses a `def` instead of a lambda.
  - **Pyright:** 18 errors go down to 16, after an annotation on `m2.Agent.evaluate`'s output dict. The rest are inference limits on `sum()`/`0` accumulators and optional arguments, and are left alone.
  - **Names:** `core.flow` names its iterate `pi` instead of `x`. `m2.Agent.C` becomes `Agent.ctrl`, since `C` means causal counts in `spiking`.
  - **Structure:** `m2.Agent.episode` is split along derivation §8 into `pre_td_updates` (module statistics, Eq. 4b trace, Eq. 4, Eq. 4d, Eq. 4a dual) and `critic_step` (TD error and NLMS critic). These names come from the expert review below; the cleaner first called them `local_plasticity` and `td_step`. Operation order is unchanged. The dual-target fraction 0.75 is now `m2.Q_STAR_FRAC`. Nested conditional expressions in `Agent.__init__` and `summary()` are now `if` statements.
  - **Docstrings:** added to functions that had none, with citations (`spiking.build`, `tau`, `pi_hat`, `run_frame`; `core._plogp`; `m2.Agent` and its methods, `random_return`, `run`, `summary`; `m1` helpers; `m2_viz.first_race_step`). The `m2.py` header now reads "models v1-v3".

### Added (code hygiene)

- **`.claude/agents/codebase-cleaner.md`:** adapted from another project for Map-STDP. Changes:
  - **Scope and layout:** the scope, the forbidden paths, and this repo's layout and dependency direction.
  - **Checks:** ruff, pytest and pyright in place of the original's tools.
  - **Hard limits for this code:** numerics, RNG-call order, the $W$/`W_snm` orientation, the crossbar-native rules, CLI and result-file formats, and no `strict=` on zips.
  - **Review:** it points to `/expert-review` instead of `design-guardian`, which doesn't exist here.
- **`tests/test_regression.py` and `tests/golden.json`:** a behaviour lock. Checked:
  - Fixed-seed runs are bit-identical across processes, spiking included.
  - The suite covers five closed-loop configurations, the module self-checks, pairing counts with the race readout, and `m2.summary()`.
  - Changing the weight floor by 0.01% fails all five closed-loop cases.
- **`ruff.toml`:**
  - **Rules pinned:** E, F, W, I and B, because ruff 0.16's defaults are much wider.
  - **B905 off:** `m2.evaluate` relies on zip truncation.
  - **No `ruff format`.**
  - **Baseline:** 21 findings, left for the cleaner.
- **`environment.yml`:** adds ruff and pytest; both are installed in the local env.

### Fixed

- **`m2.py --summary` crashed** on the replay recording, because its `m2_*.json` glob matched `m2_viz_episode_s3.json`. The viz outputs are renamed `viz_m2_*`.
- **`pyrightconfig.json`** no longer hard-codes the cluster env path. Pyright and the IDE now resolve imports from the active interpreter on any machine.

### Added (visualisation)

- **`experiments/m2_viz.py` and `experiments/m2_viz_template.html`:** a replay page of a trained spiking v3 agent. The left pane shows spikes and walk hops on the network, with the controller drawn as its 6×6 (θ, θ̇) grid and the race winner marked; the right pane shows CartPole. Commands:
  - `train` saves the weights (`viz_m2_W_s<seed>.npz`, renamed from `m2_viz_*` so `m2.py --summary` no longer picks them up; seeds 1–4 have last-100 returns of 193, 143, 207 and 192);
  - `record` keeps the best of 20 frozen-weight episodes (seed 3: 302 frames);
  - `page` embeds the recorded episode into the template.

### Changed (environment)

- **Local workstation env:** Miniforge in `~/miniforge3`, with the `map-stdp` env created from `environment.yml` plus katex, pyright and CodeGraph 1.6.2 (telemetry off). The math hook, the self-checks and the CodeGraph index (148 nodes) work there. The M2 runs used a scratchpad venv with the same packages before this env existed.
- **`CLAUDE.md` and `environment.yml`:** the "environment.yml is stale" note is removed; the file already lists the current stack. The setup comments now cover any machine as well as the cluster.

### Added (iteration 002: M2 done)

- **M2 results.** 110 runs, 5 seeds × 1500 episodes each, run locally on a 72-core workstation with no SLURM; results are in `experiments/results/m2_*_{v1,v2,v3}_s*.json`. Model v3, mean of the last 100 episodes, against a random policy of 22:
  - **Flow:** exempt 287 ± 21, dual 327 ± 46, none 39 ± 11, $\lambda = 0$ 187 ± 15.
  - **Spiking:** exempt 184 ± 13, $\lambda = 0$ 77 ± 4.
  - **Race vs. Eq. 1d:** KL 0.010, against a sampling floor of 0.010.

  All M2 criteria are met; SPEC §11 marks M2 done. Full record: `docs/iterations/002-m2.md` (v1 baseline, reviews, v2, annealing test, v3, outcome).
- **Reviews** by `snn-expert` (spiking diagnostics) and `neuroscientist` (plausibility), merged in the iteration record.
- **Code:**
  - `mapstdp/core.py`:
    - `homeostasis_update` (Eq. 4d);
    - `beta` in `policy`, `action_score` and `eligibility_step`, for the sharpened policy (Eq. 1e). The self-check now verifies the $\beta = 3$ score against finite differences.
  - `mapstdp/spiking.py`: `race(..., burn)`, with a self-check.
  - `experiments/m2.py`:
    - `--model {v1,v2,v3}` (default v3), `--beta`, `--eta`, `--eta-tau`;
    - the variant goes into the result filename;
    - `--summary` groups by model and variant.

  The v1 result files were renamed with a `_v1` suffix.
- **`docs/derivation.md`:**
  - **§2.2:** homeostasis as its own term (4d); weak-synapse rectification does not depend on the floor value; soft bounds (**Proposal**).
  - **§2.4:** the race equals (1d) only after relaxation, so a burn-in is adopted; sharpened policy (1e) (**Proposal**).
  - **§4.5:** M2 numbers; sealing caveat (persistence 0.996 against about 80% intrinsic input in cortex); exit-floor dual (**Proposal**, M4); the dual's $\mu$ diverges.
  - **§4.6:** $\bar{h}_j$ is presynaptic; the $\tau_e$ timescale.
  - **§8:** burn-in, (4d) and annealing in the training loop.
  - **§9, item 14:** updated.
  - **Appendix A:** iteration-002 checks.
  - **Appendix B:** item 11.
  - **References** (each checked by the `neuroscientist`): Chistiakova et al. 2014, Frank 2006, Gold & Shadlen 2007, Gurney et al. 2015, Markov et al. 2011, Royer & Paré 2003, Turrigiano et al. 1998, van Rossum et al. 2000, Zenke & Gerstner 2017.

### Changed (model v1 → v3, `docs/model.md`)

- **Decision: $d_j$ homeostasis is its own term on all plasticity** (Eq. 4d, $\epsilon_h = 5$, independent of $\lambda$), not part of the structural baseline. Inside $\lambda$ it was about 100× too weak, and it vanished at $\lambda = 0$. The task term ratcheted $d_{max}$ to 13 (flow) and 19 (spiking). Now $d_{max} \le 1.04$. Its expectation is still a column rescaling: spiking Monte Carlo cosine 0.992 (preliminary).
- **Decision: race burn-in of 10 steps.** Without it, the race read the cold-start transient (KL 0.022 against a floor of 0.010). With it, KL is at the floor.
- **Decision: anneal the task rate,** $\eta = 10$ decayed as $\eta / (1 + e/300)$ over episodes, in both backends (v3). v2 used a constant rate with spiking $\eta = 5$, and its return fell: spiking 140 → 120, flow at $\lambda = 0$ 210 → 145. The v1 ratchet had acted as hidden annealing. Constant $\eta = 10$ gives spiking 88, annealed gives 184.
- **Decision: routing exemption stays the default.** The dual is within about one sd ($t = 1.6$, $p = 0.15$), its order relative to the exemption flipped between v2 and v3, and its $\mu$ never converges.
- **Rejected for now: the sharpened policy (Eq. 1e).** At the flow level it collapsed on 3 of 5 seeds at $\beta = 4$ and on 1 of 5 at $\beta = 2$. It stays a **Proposal**, needing an entropy floor and a matched spiking readout.
- **Deferred: the exit floor on module sealing.** Sealing costs no return on CartPole (`snn-expert`), but it is implausible as anatomy (`neuroscientist`). Revisit at M4.
- **`docs/SPEC.md`:** status line, M2 marked done, the M2 routing results, and ablations (homeostasis placement, task-rate schedule, temperature, exit floor, race burn-in).

### Added (implementation, iteration 001: M1)

- **Code:**
  - `mapstdp/core.py`: numpy flow-level reference;
  - `mapstdp/spiking.py`: SuperNeuroMAT backend;
  - `experiments/m1.py`: the six M1 checks, with results in `experiments/results/m1.json`.

  Both modules have assert-based self-checks. Built by `superneuro-expert`. `pyrightconfig.json` points pyright at the `map-stdp` env.
- **`docs/model.md`:** the current model, versioned. v0 is the spec; v1 follows M1.
- **`docs/iterations/001-m1.md`:** the M1 record, with build, results, reviews (`snn-expert`, `neuroscientist`) and decisions.

### Added (iteration 002: M2, in progress)

- **Code** (built by `superneuro-expert`; self-checks pass):
  - `mapstdp/core.py`: the CartPole conjunctive encoder, $d_j$ homeostasis, and the action-gated eligibility (Eq. 4b);
  - `mapstdp/spiking.py`: the race readout;
  - `experiments/m2.py`: the closed-loop CLI and `--summary`.
- **`docs/iterations/002-m2.md`:** status, run plan, preliminary numbers, and open problems:
  - a $d_j$ ratchet from the task term;
  - the race readout reading the early transient.
- **No M2 results yet.** The SLURM jobs on `secondary` sat pending on priority for over 4 h and were cancelled on the user's instruction. M2 continues on a GPU machine that needs no SLURM.

### Changed (portability)

- **Paths:** the hooks, `.mcp.json` and the skills no longer hard-code `/scratch/vikramr2`.
  - The hooks use the active env (`CONDA_PREFIX`).
  - `math_check.js` finds katex next to the running `node`.
  - CodeGraph and the skills resolve from `PATH`.

  Launch Claude from `conda activate map-stdp` on any machine.

### Changed (iteration 001 decisions)

- **Decision: pairing count $\kappa$ is the covariance count** (derivation §2.2), replacing the balanced count. Cosine with $-W \odot \nabla D$: 0.992 covariance, 0.979 causal, 0.847 balanced, reproduced at 5k frames. References added: Sejnowski 1977 and Kempter et al. 1999.
- **Decision: presynaptic normalisation** as an output gain $1/d_j$, plus $d_j$ homeostasis $\epsilon (d_j - 1)$ in the baseline (derivation §2.2).
- **Decision: spiking uses branching mode,** with $k = 100$. Frame windows are stated in spikes per module (error ≈ $1.3 / \sqrt{S_{mod}}$). The $4 \tau / \alpha$ estimate in §2.3 was optimistic.
- **Decision: conjunctive encoding** (SPEC §4, derivation §2.4). Per-dimension fields cap the CartPole policy at about 84 steps.
- **Decision: routing exemption is the M2 default** (SPEC §6, derivation §4.5). In the flow-level closed loop, the map term without protection collapsed learning (≈53), while exempt routing at $\lambda = 0.05$ reached ≈343 against ≈195 for $\lambda = 0$. These numbers are preliminary. The dual term stays a Proposal and becomes an ablation. A constant $\lambda$ replaces the ramp, which becomes an ablation.
- **SPEC:**
  - M1 marked done;
  - status line updated;
  - ablations updated (pairing count, normalisation, encoding, schedule);
  - "Controller in $D$" provisionally answered (input layer).

### Added

- **`map-stdp` conda env** (`~/.conda/envs/map-stdp`): Python 3.11 and Node.js 26, so the tooling has Node without a system-wide install.
- **ponytail plugin** (v4.11.0, [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)), installed at project scope through `.claude/settings.json`. It is a least-code skill. Its hooks were reviewed before install: they inject instructions and write small mode files under `~/.claude` and `~/.config/ponytail`, with no network calls and no subprocesses. It needs `node` on PATH, so activate the env before launching `claude`.
- **CodeGraph MCP server** (v1.6.2, [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph)), installed in the env and registered at project scope in `.mcp.json`.
  - Telemetry is off, both by `codegraph telemetry off` and by `CODEGRAPH_TELEMETRY=0`.
  - The local index is in `.codegraph/`, which ignores itself.
  - The repo has no code yet, so the index is empty until implementation starts.
- **Project hooks** (`.claude/hooks/project_hooks.py` and `math_check.js`, registered in `.claude/settings.json`), each tested with sample events:
  - **Math check.** After any edit to `docs/*.md` or `CLAUDE.md`, KaTeX renders every math span in strict mode and banned macros (`\,` `\;` `\tag` `\{` `\}` `\|`) are flagged. Failures are fed back so they get fixed immediately.
  - **Changelog guard.** Blocks `git commit` when changes are staged without `docs/CHANGELOG.md`. Override with "skip changelog" in the message.
  - **No global installs.** Blocks pip, npm `-g` and conda installs that don't target the `map-stdp` env, a venv, or `--target`.
  - **Environment check.** At session start, warns if the `map-stdp` env or `node` is missing.
- **Project skills** (`.claude/skills/`):
  - **`/derivation-check`** (`scripts/fdcheck.py`): the finite-difference and Monte Carlo checks CLAUDE.md requires. Its selftest reproduces a map-gradient error of 7e-10 and a three-factor correlation of 0.99994.
  - **`/doc-edit`**: house rules for the docs.
  - **`/expert-review`**: runs the expert agents in parallel and merges their findings.
  - **`/changelog`**: drafts an entry in this file's style.
- **Official plugins at project scope:**
  - **math-proof**, for proofs of the derivation's theorems;
  - **pyright-lsp**, with pyright 1.1.414 installed in the env rather than globally;
  - **commit-commands**;
  - **hookify**.
- **`map-stdp` env:** gained katex, pyright, numpy and scipy.
- **SLURM template** (`slurm/job.sbatch`): the user's standard header (partition `zgdrasil`, 8 CPUs, 64 GB, 2 days). It activates `map-stdp` and runs the given command. `logs/` is gitignored.
  - CLAUDE.md gains a "Compute (SLURM)" section: long runs go to `sbatch`, never the login node.
  - The `snn-expert` and `superneuro-expert` agents point to it.
  - **Decision:** jobs go to the `secondary` partition (4 h, preemptible), on the user's instruction. Installs run on the login node.
- **Env moved to `/scratch/vikramr2/conda/envs/map-stdp`.** The `/u` home hit its 500k-file quota at 503k files (`~/.conda/pkgs` held ~224k).
  - `conda clean --all` freed ~107k files.
  - The env was rebuilt on scratch, with its package cache there too, and registered through `envs_dirs`. It now also has numba, gymnasium, matplotlib, networkx and superneuromat 3.5.0.
  - Hooks, skills and `.mcp.json` point to the new path.
  - `environment.yml` is rewritten to match; it had listed the old PyTorch stack.
  - The old home env is removed once Claude is relaunched from the new env.

## 2026-10-04

### Changed (simulator)

- **Decision reversal: SuperNeuroMAT replaces Brian2** as the spiking simulator (SPEC §3). Reasons:
  - its discrete-time matrix update makes one step one hop, and one crossbar read;
  - the per-frame closed loop is plain Python;
  - the plasticity code can be shared with the numpy reference;
  - it has CPU, JIT and GPU backends;
  - it can export to the NeuroCoreX FPGA platform (inference only).

  Brian2 stays as a fallback for continuous-time checks. SuperNeuroABM is an option for heterogeneous neurons.
- **Smoke test** (SuperNeuroMAT 3.5.0, installed only in the scratchpad):
  - `weight_mat()` is `[pre, post]` (the transpose of $W$);
  - built-in `aneg` is a per-step decay on non-coincident synapses, not acausal STDP, so Map-STDP is applied per frame outside `simulate`;
  - `add_spike` times are relative to the current step;
  - cost per frame is about 1 ms at ~100 neurons, 40–50 ms at ~500 and ~230 ms at ~1000, dominated by weight get/set in Python at larger sizes.
- **`docs/derivation.md` §2.3:** a discrete-time note. One step with delay 1 is one hop. A branching-process mode (infinite leak, stochastic input) is a Proposal for M1.
- **`CLAUDE.md`:** simulator, submodule, transpose convention, and no global installs.
- **`.claude/agents/snn-expert.md`:** tooling is no longer Brian2-specific.

### Added (simulator)

- **`.claude/agents/superneuro-expert.md`**: a subagent expert in coding SuperNeuroMAT and SuperNeuroABM. Its API facts were checked against the source, and it follows the project constraints.

### Changed (review revisions)

Applied the findings of the neuroscientist and SNN-expert reviews of `docs/derivation.md`.

- **Decision: new task term.** TD-modulated STDP with an action-gated eligibility trace (derivation Eq. 4b, 4c, §4.6) replaces $(R - \bar{R}) e_{ij}$. That term gets no action credit when the action is sampled from flow shares, and stalls under sparse reward. The new eligibility has the same crossbar form as the structural rule.
- **Proposal: dual routing term** (Eq. 4a, §4.5). The map term alone starves controller→action and controller→latent routing, because that routing is cross-module. The dual term holds routing at a target $q^{\ast}$, and the simplest variant exempts the routing pairs. It stays a Proposal until simulated.
- **Spiking timing.** Spiking frames take about $4\tau/\alpha$ (100–250 ms), not 20 ms. The RL ladder runs on a numpy flow-level reference first, and Brian2 is used for fidelity at M1 and M2.
- **Mean field.** The doc now states its three requirements: constant $d_j$, global normalisation, and causal-minus-acausal pairing counts.
- **Locality (§6).** Restated: the postsynaptic module is the local one. The doc also notes what neuromodulators can carry, and that the per-neuron baseline equals presynaptic renormalisation to first order.
- **Readout and memory.**
  - A race readout matches Eq. 1d, and the limits of the policy family are stated.
  - Linear carry-over forgets geometrically; a module-level nonlinearity is proposed.
- **Lateral inhibition (§7)** is routed through interneurons with inhibitory STDP, for Dale's law.
- **Training (§8)** gains stability measures and a $\lambda$ ramp. **Assumptions (§9)** gain items 12–15.
- **Claims softened:** controller ↔ cortex (now a functional abstraction with basal-ganglia, OFC and hippocampal analogues), the characterisation of Buesing et al., the metabolic reading of the map equation, Friston, and Lynn et al. (an fMRI measure).
- **References.** 14 added, each checked by title, venue and year during the review.
- **New numerical checks (Appendix A):**
  - action-gated eligibility: cosine 0.997 with the true $\nabla \log P$, and the Monte Carlo correlation is 0.997;
  - baseline as renormalisation: first order, relative difference $2.5\lambda$;
  - dual-term gradient: exact;
  - carry-over memory decays 0.29–0.39 per frame.

  The preliminary spiking and bandit numbers come from the review's own simulations and still need reproducing.
- **`docs/SPEC.md`:**
  - architecture analogies softened;
  - spiking timing corrected;
  - task-term decision recorded;
  - routing conflict and dual Proposal added;
  - new open questions (controller in $D$, $q^{\ast}$, nonlinear carry-over, device multiplicativity, Dale's law, efference copy);
  - new ablations;
  - M1 and M2 re-scoped.
- **`CLAUDE.md`:** task term and routing-conflict notes.

### Added

- **`.claude/agents/snn-expert.md`**: a subagent expert in SNNs, local and three-factor learning, Brian2 and closed-loop RL, focused on improving performance within the crossbar-native constraints. It can run and edit code.
- **`.claude/agents/neuroscientist.md`**: a read-only subagent that reviews biological plausibility (locality, timescales, neuromodulator specificity, Dale's law, strength of analogies) with verified citations.

### Changed

- **Reformulation: communities as states.** The primary goal is now a state-space abstraction for spiking networks in which communities are the states. Map-STDP is framed as the **inverse** of community detection: the partition is given, and the rule learns weights that make it the minimum-description-length partition of the walk. The thermodynamic surrogate hypothesis becomes a secondary side tie, and the energy proxy a secondary metric.
- **`docs/derivation.md`:**
  - New title, goal and motivation. The controller community is motivated by the cerebral cortex, with Global Workspace Theory kept as an analogy.
  - Notation adds module types $\mathcal C$ (controller), $\mathcal A$ (actions) and $\mathcal L$ (latent), frame index $f$, hops $n$, and module-pair flows $J_{ba}$, $T^K_{ba}$.
  - New §2.3, one walk per frame:
    - per-frame flow $\pi_f$, computed implicitly by $n$ hops with error $\le 2(1-\alpha)^n$ (Eq. 1a);
    - software, crossbar and spiking realisations;
    - timescale ordering (hops, then frames, then plasticity);
    - a carry-over proposal, Eq. 1b, so latent communities persist across frames.
  - New §2.4, the extracted Markov model:
    - the module chain $T^K(\mathbf o)$ (Eq. 1c);
    - why a central controller makes it stimulus-conditioned;
    - policy readout from action communities (Eq. 1d);
    - a proposed across-frame latent chain as a world model.
  - New §3.4: the inverse problem $\min_W \mathbb{E}_f[L(M; \pi_f)]$ (Eq. 3a). Theorems 1–3 hold frame by frame, and re-detecting latent modules makes training an alternating minimisation.
  - §5: candidates table gains a "state abstraction" column. New §5.9 proposes the entropy rate of the latent chain as a candidate; its modulator is not yet derived.
  - §6: per-frame walk row in the crossbar table.
  - §8: training loop restructured by frame. Only latent modules may be re-detected.
  - §9: assumptions 9–11 added (convergence within a frame, carry-over, number of latent modules).
  - Appendix A and Appendix B item 8 added.

  New results, each checked numerically on a 15-node, 5-module network:
  - the per-frame and frame-averaged map-equation gradients match finite differences ($\le 10^{-9}$);
  - the three-factor rule averaged over frames matches $-W \odot \nabla D$ (correlation 0.99996, or 0.99993 with a pooled modulator);
  - modular networks contract at nearly the $1-\alpha$ bound (0.78 vs. 0.8);
  - with carry-over held fixed, the gradient has cosine 0.98 with the full gradient;
  - the controller column of $T^K$ carries most of the stimulus dependence.
- **`docs/SPEC.md`:**
  - §1 has the new goal and hypotheses.
  - §2 has the new architecture: controller, latent and action communities, with carry-over.
  - §4 is RL-only. The ladder is CartPole → Acrobot/MountainCar → LunarLander → a partially observable task. **Vision is deferred.**
  - §5 merges stimulus and timescale. Implicit walk per frame and carry-over are **Proposals**.
  - §6: the candidates table gains a state-abstraction column and the $h_{\mathcal L}$ proposal. Decision criteria now include extracted-model quality, and energy is secondary.
  - §8: pinned vs. free latent partitions; $\lvert \mathcal L \rvert$; the world-model open question.
  - Metrics and ablations are split into §9 and §10 and extended (policy KL, implicit vs. spiking walk, carry-over, latent communities).
  - Milestones M1–M6 are re-scoped.
- **`CLAUDE.md`:** research question rewritten; conventions for per-frame flow and module types.

## 2026-10-01

### Added

- **`docs/review/search.py`** and **`docs/review/data/`**: the Phase 2 search script and its output. It queries OpenAlex, PubMed and arXiv and returns 1,870 records, which come to 1,276 after deduplication.
- **`docs/review/screening/`**: screening rules (clarifications of protocol §B.2), the 50-record calibration set, a blank author sheet, and the AI's calibration decisions, committed before the author screens.
- **`docs/review/protocol.md`**, a draft PRISMA-P protocol (frozen at v1.2 after author approval; not registered, git commit serves as the timestamp) for a systematic review of how neural signalling dissipates energy and what evidence links information-theoretic measures to metabolic or thermodynamic cost. It covers three sub-questions (mechanism, information–cost linkage, network level) and separates cost classes M/T/W/P with no pooling across classes. It also adds a gap check for description-length measures, including the map equation, and records Devil's Advocate Checkpoint 1. The protocol is awaiting author approval, and no search has been run beyond pilot counts.

### Changed

- **Review protocol v1.3:** the search strings were revised after the seed recall check missed 3 of 10 eligible seeds. The revised strings retrieve all 10. Details are in the protocol's §B.13.

## 2026-09-27

### Removed

- **The original LaTeX derivation** and its template files, superseded by `docs/derivation.md`. Its training-loop figure was dropped because it depicted the old per-sample E-step/M-step loop.
- **The PyTorch prototype in `models/`.** It had:
  - an SBM starting-topology generator;
  - a shared LIF simulation with an E-step/M-step training loop;
  - 4-bit suppression STDP (Gautam & Kohno 2023);
  - two Map-STDP variants trained on N-MNIST: fixed-community ($G_{sim}$) and lateral-inhibition ($G_{LI}$).

  The project restarts from ideation. The code remains in git history (last at `fcedb0a`).

### Added

- **`docs/derivation.md`**, a corrected markdown rewrite of the original LaTeX derivation. The rewrite:
  - fixes a sign error in the expansion of $L(M)$. The gradient's log factor becomes $\log\frac{q_\curvearrowright(p_m+q_m)}{q_m^2} \ge 0$, verified by finite differences.
  - adopts a single direction convention: $W_{ij}$ is pre $j$ → post $i$, and the walk follows spikes forward. As a result, $G_{sim}(i,j) = \mathbb I[i\notin m(j)] - \bar e_j$ is indexed by the presynaptic neuron.
  - adds the stimulus as a teleportation term, $\pi = (1-\alpha)T\pi + \alpha v(\mathbf o)$.
  - turns the identification of the firing marginal with walk flow into an explicit mean-field assumption.
  - replaces the fixed-point "convergence" proof with a contraction proof.
  - restates the lateral-inhibition variant with a soft indicator $\chi_{ij}$, and flags the problems with its anti-Hebbian rule.
- **`docs/SPEC.md`**, the project spec: goal and hypothesis, architecture, and five considerations with proposals (Brian2; the RL and vision task ladder, with classification as a contextual bandit; stimulus as teleportation; candidate description lengths vs. thermodynamic entropy; 3-factor variants). It also lists metrics, ablations and milestones M0–M6.
- **`docs/CHANGELOG.md`**, this file.

### Changed

- **`CLAUDE.md`**, rewritten for the docs-first phase. It now also lists math-rendering conventions.
- **`docs/derivation.md`**, consolidated to put interpretation first:
  - an "At a glance" section with the rule, a sign/magnitude table and a worked example;
  - a notation table saying where each quantity lives;
  - an intuition for the map equation as a two-level code, and the fact that leakage always costs bits;
  - a single assumptions list;
  - the corrections to the earlier version moved to Appendix B.

  New result: $\sum_i W_{ij} G(i,j) = 0$, so a multiplicative map term conserves each neuron's outgoing weight (checked numerically). All math now renders in KaTeX, GitHub and Markdown Preview Enhanced: no `\tag`, `\,`, `\{` or `\|`, and every span is validated with KaTeX.
- **`docs/SPEC.md`**, cross-references updated to the new section and equation numbers. $G_{sim}$ is renamed to $G$, and math spacing is fixed.
- **Decision: structure is a neuromodulated 3-factor rule.** The structural term is now cost-modulated STDP: $\Delta W_{ij} = -\lambda (g_{ij} - \bar{g}_j) \kappa_{ij}$. The neuromodulator broadcasts the *marginal* description cost $g_{ij} = \partial D / \partial J_{ij}$, a $K \times K$ module-pair table.
  - Chosen over a factorized-gradient hardware rule because of the new memristive-crossbar target.
  - The exact gradient remains the analysis reference and simulation baseline.
  - Supersedes SPEC 3-factor variants A/B/C.
- **`docs/derivation.md`**, restructured around the rule:
  - new §4: the rule, and Theorem 3, which shows the expected update is $-\lambda W_{ij} \partial D / \partial W_{ij}$; marginal vs. pointwise cost;
  - new §5: eight candidate description lengths as modulators (map equation, entropy rate, entropy production, cost-weighted map equation, Markov stability, SBM description length, predictive dissipation, cross-module information flow);
  - new §6: crossbar mapping;
  - neuromodulator/energy motivation with verified citations.

  New results, each checked numerically:
  - $M^{\ast}_m$, previously absorbed into the learning rate, is the map equation's per-module neuromodulator, firing only on module exits.
  - $G$ is exactly the lag-1 Markov-stability gradient.
  - Broadcasting pointwise codeword length is biased: correlation 0.976 with the true descent direction, against 0.9999 for the marginal cost.
  - Coarse-grained $\sigma_K$ lower-bounds $\sigma$, but $h_K$ has no general ordering with $h$.
- **`docs/SPEC.md`**, restructured:
  - §6 merges C4 and C5 into "Description length as a neuromodulator", with a candidate table and decision criteria;
  - new §7 (C6) sets memristive crossbars as the hardware target, with its constraints;
  - ablations, M1 and M5 updated.
- **`CLAUDE.md`** adds the crossbar-native constraint.
