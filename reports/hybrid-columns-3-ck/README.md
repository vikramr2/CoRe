# Study `hybrid-columns-3-ck`: the CK-faithful base against the regimes of `hybrid-columns-3`

**Pre-registered 2026-10-29 12:03 EDT, before any run of this study** (approved by the user; code at the commit recorded in `manifest.json`, the first commit after this stamp, run from worktree `../map-stdp-hc3ck`, branch `hybrid-columns-3-ck-study`). The design and tests below are fixed.

Nothing in this directory is written by hand except this README. `manifest.json`, `code.patch`, `runs/`, `summary.*` and `pairs.csv` come only from `python -m experiments.report`.

## What the study is

- **Question (user, 2026-10-29):** do the three regimes of action columns of `hybrid-columns-3` keep or improve performance against the TD-STDP backbone **as published**, that is, Chung & Kozma's softmax readout on the unnormalised action scores, at the same matched start?
- **Why it is needed:** every arm of `hybrid-columns-3` uses the normalised action scores (`--norm-logits`, iteration 016 §13). Its base is therefore a modified backbone. The cross-study reading of 016 §17 says that base is level with CK's on CartPole and Acrobot but loses about 28 points of late LunarLander return (last-100 30 vs 58 for `hybrid-columns`' CK-faithful base at unmatched start), and the Stage B pilot (016 §18, dev seeds) found `base_nl` − `base_ck` of −11 [−30, +11] AUC on CartPole and +1 [−23, +24] on Acrobot. So the LunarLander gain of the column arms in `hybrid-columns-3` (+7 to +11 last-100) is measured against a weakened base. This study supplies the missing reference on all three tasks, at the matched start and on the same seeds.
- **Companion:** `hybrid-columns-4` (CartPole at n = 40) runs at the same time.
- **Design record:** `docs/iterations/016-columns-more-tasks.md` §19.

## Design

- **One group per task, `<task>_base_ck`:** `--engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --ent-match-rec`; LunarLander and Acrobot add `--eta-cal`. These are `hybrid-columns-3`'s common flags **without `--norm-logits`**, i.e. the Stage B pilot's `base_ck` arm (016 §18). With no recurrence the two matching flags are no-ops (recorded in `low_rate`: `r_got_hz`, `ent_scale` = 1), so the arm is CK's readout at the 40 Hz start of the earlier studies, with their η rules (CK's preset η on CartPole, the calibrated local step on the other two tasks).
- **Task settings:** the presets of `experiments/tdstdp.py`, as in `hybrid-columns-3`.
- **Episodes:** as `hybrid-columns-3`: CartPole 400; LunarLander pooled over 16 copies, at least 3000 and at most 10000, stopping once solved after 3000; Acrobot 500.
- **Seeds: 5000–5019, the same as `hybrid-columns-3`, on purpose.** Each contrast below then starts from the same initial W and the same first environment seed as the arm it is compared with (both come from the run seed's stream; the trajectories diverge afterwards, as between any two arms of one study). No run of this arm has ever used these seeds (the pilot's `base_ck` used dev seeds 20–29), so nothing was selected on them. This is a stated exception to the fresh-seed rule of the earlier studies.
- **Groups and runs:** 3 groups × 20 seeds = 60 runs.
- **Code state:** `hybrid-columns-3` ran at `4bdbe45`; this study runs at the HEAD recorded in `manifest.json`. The changes between them are the race readout (`--readout race`, default off), the E-count tally and energy monitors (race only), the `rec.RKc` monitor fix (a recording change on column arms; this arm has no recurrence) and the study declarations. The golden regression test is unchanged across them, and none of the changed paths is reached by this arm's flags, so a `base_ck` run at HEAD is the run it would have been at `4bdbe45`. The cross-study contrast therefore spans two commits with one effective code path; stated here rather than assumed.

## Pre-registered tests

**Primary metric:** LunarLander last-100 return; AUC (the mean return over the fixed episodes) on CartPole and Acrobot, as in `hybrid-columns-3`.

**Contrasts:** two-sided Mann–Whitney on the primary metric; Hodges–Lehmann shift with a 95% CI. The `on` side of every pair is a group of `hybrid-columns-3` (`hybrid-columns-3/<group>`, loaded from that study's `runs/`); the `off` side is this study's `<task>_base_ck`. Holm is applied within each family, and the headline families are **per task** (three tests each), as `hybrid-columns-3`'s README states for its own contrasts.

| Family | Contrast | Asks |
| --- | --- | --- |
| **headline** (`CartPole`, `LunarLander`, `Acrobot`; 3 tests each) | `hybrid-columns-3/<task>_cols_lam0` vs `<task>_base_ck` | does implicit modularity keep or improve performance against the published backbone? |
| | `hybrid-columns-3/<task>_cols_on` vs `<task>_base_ck` | learned modularity (Map-STDP) against the published backbone |
| | `hybrid-columns-3/<task>_cols_sealed` vs `<task>_base_ck` | enforced modularity against the published backbone |
| readout (`readout`, 3 tests) | `hybrid-columns-3/<task>_base` vs `<task>_base_ck` | what does the normalised readout itself cost or gain, per task? |

For each headline contrast, as in `hybrid-columns-3`:
- **Improves:** Holm-adjusted p < 0.05 and a positive shift.
- **Keeps performance (non-inferiority):** the lower end of the 95% CI of the shift is above −Δ, with **Δ = 25** (last-100 return on LunarLander, AUC on CartPole and Acrobot).
- **Worse:** a significant negative shift.

**Pre-registered expectations** (from 016 §17's cross-study reading and the §18 pilot; expectations, not results):
- Readout family: `base` (normalised) non-inferior to `base_ck` on CartPole and Acrobot; worse on LunarLander (the late plateau of §17).
- Headline: the three regimes non-inferior to `base_ck` on CartPole and Acrobot. LunarLander is the open question and is not predicted: the regimes recovered part of the normalised base's late loss in `hybrid-columns-3`, and whether that reaches the published backbone's level is what this study asks.

**Secondary (reported, not tested for the claims):** solved count, T_s, last-100 return (all tasks), LunarLander AUC over the first 3000 episodes, collapsed runs (Acrobot last-100 = −500), the matching record per run, actor and critic rates and policy entropy per log bin (the log files), ms per frame.

**Not done:** no settings change after launch, and no seeds are added or dropped. A crashed run is relaunched from its checkpoint. If a group still has fewer than 20 runs, it is reported with its n. `hybrid-columns-3`'s `runs/` are read, never rewritten.

## Run conditions

- **Machine:** the local workstation (72 cores), with `OMP`, `MKL`, `OPENBLAS` and `NUMBA` threads set to 1, 60 workers (every run starts at once; the 20 LunarLander runs then hold 20 cores for their full length).
- **Code state:** git worktree `../map-stdp-hc3ck` (branch `hybrid-columns-3-ck-study`) at the commit recorded in `manifest.json`.
- **Projected time:** 7–8 h, set by LunarLander (6.8 h mean, 7.3 h max per run for `hybrid-columns-3`'s base); CartPole runs take about 3 min, Acrobot about 5 min.
- **One code state:** `manifest.json` records the git HEAD and the sha256 of the uncommitted code; the runner refuses to relaunch if it differs.

## Commands

```bash
python -m experiments.report run hybrid-columns-3-ck --workers 60   # launch or resume every declared run; then summarize
python -m experiments.report summarize hybrid-columns-3-ck          # summary.csv, pairs.csv and summary.md from runs/ (and hybrid-columns-3's)
```

For example:

```bash
python -m experiments.tdstdp --task LunarLander --seed 5000 --engine nlif --rate-target 40 --block 20 --fast --rate-match-rec --ent-match-rec --eta-cal --out reports/hybrid-columns-3-ck/runs/LunarLander_base_ck/s5000.json
```
