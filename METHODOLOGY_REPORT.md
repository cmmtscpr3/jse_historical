# Methodology Report: The Johor DUN First-Principles Election Prediction Engine

**Subject:** Design, estimation, and out-of-sample validation of the seat-level election
model implemented in `first_principles_prediction_engine.py` and deployed in
`first_principles_scenario_dashboard.html`.

**Purpose of this document:** to give a third party a complete, self-contained account of
(1) how the engine was built, (2) how it was backtested, and (3) an honest assessment of
whether it is a robust and defensible basis for modelling election outcomes — including
its known limitations. All figures in this report were independently recomputed from the
raw harmonised data in `DATA/`; none are quoted from the engine's own documentation.

---

## 1. Scope and objective

The engine predicts **Barisan Nasional (BN) vote share and seat outcomes** for the 56
Johor state assembly (DUN) constituencies, under user-specified assumptions about
turnout, community-level support, opposition alignment, and coalition configuration.
It is a **scenario engine**, not a poll aggregator: it converts a small number of
statewide assumptions into 56 seat-level predictions. Its stated goal is predicting
*BN's* performance; the opposition side (PH vs PN) is modelled more coarsely.

## 2. Data

Six harmonised CSV files covering three elections on identical seat boundaries:

| Election | Results file | Composition file |
|---|---|---|
| 2013 GE (DUN) | `JOHOR_2013_DUN_RESULTS_HARMONISED.csv` | `JOHOR_2013_DUN_COMPOSITION_HARMONISED.csv` |
| 2018 GE (DUN) | `JOHOR_2018_DUN_RESULTS_HARMONISED.csv` | `JOHOR_2018_DUN_COMPOSITION_HARMONISED.csv` |
| 2022 state election | `JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv` | `JOHOR_2022_DUN_COMPOSITION_HARMONISED.csv` |

Data-quality audit (performed independently for this report):

- All three years contain exactly 56 seats with no duplicates, and **seat names match
  100% across all three elections** — the panel is clean, with no boundary breaks.
- Party-level votes sum **exactly** to `TOTAL VALID VOTES` in every seat, every year.
- Registered Malay + Chinese + Indian shares sum to 94.7–99.9% (remainder: Orang Asli,
  other Bumiputera, others).
- Recorded winners match the historical record: 2018 — PH 36, BN 19, PAS 1;
  2022 — BN 40, PH 12, PN 3, MUDA 1.
- Turnout ranges: 82–90% (2013), 78–87% (2018), 43–68% (2022 snap election).

## 3. Model architecture

The model is estimated in three stages of ordinary least squares (OLS), all without
intercepts, followed by a forward-prediction equation. No machine learning is involved;
every parameter is interpretable and inspectable.

### Stage 1 — Within-group turnout rates

Seat turnout is modelled as the composition-weighted average of community turnout rates:

```
Turnout_s = TM·Malay%_s + TC·Chinese%_s + TI·Indian%_s + ε_s
```

The no-intercept specification follows from the accounting identity that overall turnout
is a weighted average of group turnouts. This recovers implied statewide rates TM, TC, TI.

### Stage 2 — Effective composition

Because communities turn out at different rates, the electorate that actually votes
differs from the registered electorate. Each seat's *effective* composition is:

```
Malay_eff_s = (Malay%_s × TM) / Turnout_s     (and similarly for Chinese, Indian)
```

In 2022 this adjustment is large: with TM = 66.2% and TC = 45.8% (a 20.3 pp gap), a seat
that is 50% Malay on the register was roughly 59% Malay at the ballot box. In 2013 and
2018 turnout was near-equal across communities and the adjustment is minor.

### Stage 3 — Community support rates

BN vote share is regressed on effective Malay and Chinese composition only:

```
BN%_s = BM·Malay_eff_s + BC·Chinese_eff_s + residual_s
```

**Indian voters are deliberately excluded** so that their contribution — alongside
candidate quality, incumbency, and local community factors — is absorbed into the
seat-level residual. Consequence: BC must be read as a *Chinese-plus-Indian combined*
rate (it is upward-biased as a pure Chinese rate), and the residual is not purely "local
personality" — it also carries each seat's Indian BN support.

### Parameter estimates by year (independently recomputed)

| Parameter | 2013 | 2018 | 2022 |
|---|---|---|---|
| BM — Malay BN support | 84.3% (se 2.1) | 65.1% (se 2.2) | 60.7% (se 2.9) |
| BC — Chinese(+Indian) BN support | 19.9% (se 3.3) | 10.0% (se 3.6) | 12.8% (se 6.3, CI 0.1–25.5) |
| TM — Malay turnout | 88.5% | 85.1% | 66.2% |
| TC — Chinese turnout | 85.6% | 85.4% | 45.8% |
| TI — Indian turnout | 93.2% | 88.5% | 36.6% |
| In-sample RMSE | 6.3 pp | 6.7 pp | 9.8 pp |

Two substantively important readings of this table:

1. **Malay BN support declined monotonically: 84% → 65% → 61%.** BN's 2022 seat
   recovery (40 of 56) was *not* a support recovery — it was produced by the turnout gap
   (which made the effective electorate more Malay) and by the opposition splitting
   between PH and PN. The model captures this mechanism explicitly, which is its core
   analytical value.
2. The higher 2022 RMSE reflects genuinely larger idiosyncratic variance (PN competing
   seriously for the first time), not model deterioration.

### Stage 4 — Forward prediction

For a future election, the engine predicts each seat as:

```
BN_pred_s = (BM + ΔMalay)·Malay_eff_next,s + (BC + ΔChinese)·Chinese_eff_next,s + residual_prev,s
```

where effective composition is derived from a target overall turnout and a Malay–Chinese
turnout gap (per-seat turnout is *derived from composition*, not assumed known), and
**the previous election's seat residuals are carried forward at full strength** as seat
fixed effects. Non-BN votes are distributed between PH, PN and others from the previous
election's observed fractions, adjustable via alignment parameters; three coalition
configurations (3-way, PH+PN pact, BN+PN pact) apply different vote-pooling rules on top.

### Implementation integrity

The interactive dashboard embeds a JavaScript re-implementation of the Python engine.
This was verified numerically: across all three coalition scenarios and varied parameter
settings, the two implementations agree to within 0.04 pp on every seat (attributable to
rounding of embedded data), with zero disagreements in predicted winners.

## 4. Backtesting methodology

### Design: a "perfect-knowledge" structural test

The backtest isolates the engine's *structure* from input-estimation error. To predict
election year T using year T−1 as baseline:

- **Given as known** (the quantities polls and voter rolls could in principle supply):
  year-T community support rates (BM, BC), year-T turnout rates (TM, TC), and year-T
  registered composition.
- **Out of sample** (the model's actual bet): the 56 seat residuals, taken from the
  year T−1 fit and carried forward unchanged; and the per-seat turnout, which the
  engine derives from composition rather than observing.
- Predictions were produced by the engine's own production code path
  (`FirstPrinciplesEngine.predict`), in the 3-way configuration, with seat winners
  scored as BN vs the strongest opposition candidate.

Each backtest is bracketed by two reference runs: a **composition-only floor**
(residuals set to zero — what demographics alone deliver) and an **in-sample ceiling**
(true year-T residuals — isolating the irreducible error from the turnout formula).

### Why this design is informative

If the engine's decomposition were wrong — if seat results were not well described by
"community support × effective composition + persistent local effect" — then carrying
forward stale residuals would not help, and could hurt. The test therefore probes the
model's two load-bearing assumptions: (a) support rates are approximately uniform
statewide within each community, and (b) seat-level deviations persist across elections.

## 5. Backtest results

### 5.1 Primary test: 2018 → 2022

| Run | RMSE | MAE | Bias | Correct calls | BN seats (actual 40) |
|---|---|---|---|---|---|
| Composition only | 8.86 pp | 7.03 pp | −1.4 pp | 48/56 | 40 |
| **2018 residuals carried forward** | **5.04 pp** | **4.03 pp** | **−1.2 pp** | **52/56** | **40** |
| In-sample ceiling (2022 residuals) | 2.77 pp | 1.98 pp | −0.9 pp | 54/56 | 40 |

The four wrong calls — Bukit Kepong and Tangkak (called BN, went opposition), Yong Peng
and Parit Yaani (called opposition, went BN) — were **all pre-flagged by the engine as
"marginal"** (within its stated uncertainty band), and they offset two-for-two, leaving
the statewide seat count exactly right. The largest vote-share error was Bukit Pasir
(+15.4 pp, seat call still correct), a seat where PN became a serious local force that
did not exist in 2018; the largest miss, Yong Peng (−12.9 pp), reflects an exceptional
2022 candidate effect (+22.5 pp local surge).

### 5.2 Replication on the opposite transition: 2013 → 2018

The identical protocol applied to the other available election pair — an election that
swung in the *opposite* direction (the 2018 anti-BN wave):

| Run | RMSE | MAE | Bias | Correct calls | BN seats (actual 19) |
|---|---|---|---|---|---|
| Composition only | 6.82 pp | 5.47 pp | −0.5 pp | 46/56 | 21 |
| **2013 residuals carried forward** | **4.71 pp** | **3.81 pp** | **−0.3 pp** | **52/56** | **19** |
| In-sample ceiling (2018 residuals) | 1.11 pp | 0.75 pp | −0.4 pp | — | — |

The design therefore has **two successful out-of-sample trials on maximally different
elections** — one where BN collapsed (2018), one where it recovered seats (2022) — and
it produced the exact statewide BN seat count in both.

### 5.3 The mechanism: residual persistence

| Transition | Correlation r | Sign flips | Std of residuals | Optimal carry slope |
|---|---|---|---|---|
| 2013 → 2018 | +0.74 | 14/56 | 6.4 → 6.8 pp | 0.79 |
| 2018 → 2022 | +0.85 | 6/56 | 6.8 → 9.9 pp | 1.24 |
| 2013 → 2022 (two cycles) | +0.59 | 18/56 | — | — |

Seat-level deviations from demographic expectation are strongly persistent across one
electoral cycle — through a change of government, a pandemic, the Sheraton Move, and the
creation of an entirely new party. Persistence decays over two cycles (r = 0.59), so
**residuals should be carried forward one election only**. The optimal carry-forward
coefficient straddles 1.0 across the two transitions (0.79 and 1.24), which makes the
engine's simple full-carry choice defensible; fitting that coefficient to a single
observed transition would risk overfitting.

### 5.4 Error decomposition

The in-sample ceiling shows the engine's structural error floor comes from deriving
per-seat turnout from composition rather than observing it: ~1.1 pp under equal-turnout
conditions (2018), ~2.8 pp under a large turnout gap (2022). Of the 5.0 pp backtest
error in 2022, roughly 2.8 pp is therefore structural and ~2.2 pp is residual drift.

## 6. Robustness and defensibility assessment

### What the evidence supports

1. **The decomposition is sound.** "Community support × effective turnout + persistent
   seat effect" is an accounting-consistent structure whose two load-bearing empirical
   assumptions (statewide-uniform support rates; one-cycle residual persistence) held on
   both available out-of-sample tests.
2. **Validated on its stated objective.** Statewide BN seat counts exact in both
   backtests; ~5 pp typical seat-level vote-share error; ~93% seat-call accuracy; and —
   importantly for decision use — its wrong calls were confined to seats it had itself
   flagged as too close to call.
3. **Transparent and auditable.** Three OLS stages and one arithmetic identity. Every
   prediction can be decomposed by hand into named inputs; disagreements resolve to
   arguments about specific, inspectable assumptions.
4. **The turnout mechanism is the right lever.** The model correctly attributes BN's
   2022 recovery to differential turnout and opposition fragmentation rather than a
   support recovery — the mechanism a scenario tool must capture to be useful.

### Known limitations (material, and should be disclosed to any user)

1. **The backtest is a best-case test.** True year-T support and turnout rates were
   supplied. In live use these come from polls; input error propagates roughly
   one-for-one — and *systematically* — into all seats of similar composition. Real
   forward error will exceed 5 pp. The validation covers the machinery, not anyone's
   polling.
2. **Support-rate levels are weakly identified; the gradient is not.** Effective Malay
   and Chinese shares are near-complementary (correlation ≈ −0.93 to −0.96), so the data
   pins down the Malay–Chinese support *gap* tightly (~48 pp in 2022) while the
   individual levels lean on the thin non-Malay/non-Chinese wedge — BC's 2022 95% CI
   spans 0.1–25.5%. Predictions within Johor's observed composition range are unaffected
   (which is why the backtests succeed), but translating an external poll number
   directly onto BM ("poll says Malay support = X, so ΔMalay = X − 60.7") is less exact
   than the dashboard's interface implies.
3. **Indian voters are not separately modelled.** Their support is absorbed partly into
   BC and partly into each seat's residual; the residual is therefore a mix of local
   effects and Indian demographics. This is a disclosed data-resolution trade-off.
4. **The opposition side is modelled coarsely.** PH/PN splits are anchored to observed
   prior-election fractions with heuristic adjustment parameters; the backtests also gave
   the opposition geometry as known. The engine is validated for predicting *BN*
   performance; PH-vs-PN outcomes rest on weaker footing.
5. **Scope: one state, two transitions, n = 56.** Both tests passed, but Johor-fitted
   parameters do not transfer elsewhere without refitting, and a boundary redelineation
   would break residual carry-forward entirely (the current panel has stable boundaries).
6. **Residuals are a one-cycle asset** (r decays from ~0.8 to 0.59 over two cycles).
   The baseline election should always be the most recent one.

### Appropriate use

Defensible as: a scenario-analysis and what-if tool for BN performance in Johor;
a device for identifying marginal battleground seats; a consistent arithmetic framework
into which analyst judgment (poll reads, local intelligence via seat overrides) is
injected. Results should be reported as ranges across plausible input scenarios.

Not defensible as: a point predictor of individual marginal seats; a poll-free forecast;
a PH-vs-PN model; or a plug-and-play model for other states or post-redelineation
boundaries without refitting and revalidation.

## 7. Reproducibility

All results in this report can be regenerated from the repository:

- `first_principles_prediction_engine.py` — model estimation, prediction engine, and
  dashboard generator (2022 baseline).
- `backtest_2018_to_2022.py` — the 2018→2022 backtest through the engine's own code
  path; regenerates `first_principles_backtest_2018_to_2022.html` (visual report:
  predicted-vs-actual scatter, per-seat errors, confusion matrix, residual-persistence
  panel, seat counts by urban–rural class and racial composition).
- `residual_persistence_analysis.py` — cross-election residual persistence analysis
  (2013/2018/2022), including the 2-variable vs 3-variable specification comparison.
- `DATA/` — the six harmonised source CSVs.

Requirements: Python with `pandas`, `numpy`, `statsmodels`. The 2013→2018 replication
and identification diagnostics in this report were computed with the same three-stage
specification applied to the 2013 and 2018 files.
