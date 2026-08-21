# Examination of the "Johor 2026 Seat-by-Seat Projection" Analyst Report

## Reconstruction of its unpublished model, validation against the actual result, and a robustness verdict

> **What this document is.** A political analyst circulated a pre-election report
> ("Johor State Election (11 July 2026): Seat-by-Seat Projection", central call
> BN ~46 / PH ~8 / PN ~1 / MUDA ~1) describing a six-step methodology but
> publishing no code. This examination (a) verifies every checkable input
> against this repository's harmonised data, (b) reconstructs the unpublished
> model as explicit code, (c) scores it against the actual result
> (**BN 48 / PH 8 / PN 0 / MUDA 0**), and (d) stress-tests its assumptions.
>
> Reproduce with: `python3 analyst_report_reconstruction.py` and
> `python3 analyst_report_sensitivity.py` (both read only `DATA/*.csv`).
> Per-seat evidence table: `analyst_report_reconstruction_seats.csv`.

---

## 1. Verdict up front

**The report's bottom-line was excellent — the best of any published forecast
for this election — but its stated methodology does not support the confidence
of its headline. It was right for partly wrong reasons.**

| | |
|---|---|
| **Data hygiene** | Impeccable. All 56 margins, all winner labels, all statewide 2022 figures verified against this repo's data to ≤0.05 pp. |
| **Reconstructability** | High. 53/56 of its seat ratings (95%) follow from six mechanical rules; only 3 seats needed seat-specific judgment. |
| **Outcome accuracy** | Central map 52/56 seats correct; topline seat error 4 (best among all forecasts it cites — Ong pro-BN: 10, Ong pro-PH: 18, naive 2022-repeat: 16, Vodus: unscorable but its BN floor missed by 28). |
| **The weak link** | Step 2, the poll-swing layer. It read "BN down ~7 pp" from a poll; BN actually finished **up ~17 pp**. The projection survived because the qualitative layer (Steps 3–5) quietly overrode the quantitative anchor. |
| **Supermajority claim** | "Almost certainly keeping two-thirds (38+)" was **not derivable from the report's own poll anchor** — every mechanical reading of that poll puts BN at 37–39 seats, a coin flip on the two-thirds line. The claim rested on the analyst's (correct) belief that the poll understated BN. |
| **Plain-win claim** | "BN clears a majority in every scenario I can construct" is genuinely robust: it survives every parameter sweep we could construct, including implausibly PH-friendly turnout differentials. |

---

## 2. What the report's "model" actually is

The report describes six steps. Reconstructed and classified:

| Step | The report's words | What it actually is |
|---|---|---|
| 1 | "Start from the 2022 baseline" | A per-seat margin table (verified perfect) |
| 2 | "Apply the swing implied by the 2026 polling" | A *narrative* anchor, not an applied swing — see §5; it points the wrong way and is never mechanically used |
| 3 | "Adjust for the PN split and the PAS 'vote BN' instruction" | The load-bearing judgment: PN-held seats fall to BN; Malay marginals get safer for BN |
| 4 | "Adjust for new vote-splitters on the PH side" | A second judgment; immaterial ex post (Bersama and MUDA all lost their deposits) |
| 5 | "Classify each seat Safe/Likely/Lean/Tossup" | Margin thresholds plus the Step-3/4 adjustments — 95% mechanically reproducible (§4) |
| 6 | "Sanity-check against the professionals" | Anchoring against forecasts that proved *worse* than the report itself |

So this is a **ratings model** (Cook-Report style), not a statistical model:
a deterministic margin-threshold classifier with a small number of qualitative
adjustments, plus seat-specific overrides. Nothing in it is probabilistic; no
uncertainty is propagated; the "plausible range" is scenario arithmetic on the
rating bands (§4.2).

## 3. Fact-check of the report's inputs

Everything checkable against this repository's data checks out:

- **All 56 seat margins** match `DATA/JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv`
  to within 0.05 pp once the convention is identified: *margin = winning
  majority ÷ total valid votes*. All 56 winner labels match.
- **Statewide 2022**: BN 43.1% / PH 26.4% / PN 24.0%, seats 40/12/3/1 — exact.
- **Turnout**: 2022 ≈ 54% ✓ (53.6% valid-basis, 54.9% ballots-basis); 2018 83% ✓ (84.5% ballots-basis).
- **Bukit Batu 137-vote majority** ✓ (exact).

Minor factual wobbles found (none load-bearing):

1. *"BN has polled a remarkably consistent ~600,000 votes across the last three
   elections"* — true for 2018 (582,531) and 2022 (599,753); **not** for 2013
   (737,876). The claim only works if "last three elections" includes the
   GE15 parliamentary vote rather than the 2013 state election.
2. *Puteri Wangsa*: the report says "MUDA's Amira Aisya defends"; nomination-day
   reporting names **Rashifah Aljunie** as MUDA's candidate there, with PH
   fielding former minister Maszlee Malik (PKR) — who won. The report also
   omits PH from its description of the fight entirely, yet PH won the seat.
3. *Jementah*: the report's note says "BN fields MIC vs DAP"; BN's candidate
   was in fact **MCA's See Ann Giap** (the 2022 candidate again), who won.
4. *Tenggaroh* is described as "mixed"; the roll says 83% Malay.

Items 2–3 matter beyond pedantry: two of the report's four "true tossups" carry
wrong candidate facts, and those are exactly the seats where candidate-level
detail is supposed to be the analyst's edge over a mechanical model.

## 4. Reconstructing the unpublished model

### 4.1 The ratings (Step 5)

Six explicit rules reproduce **53 of 56 ratings (95%)** — see
`analyst_report_reconstruction.py` PART 2:

- **R1** PN-held seat → *Lean BN gain* (the de-split logic).
- **R2** PH-held seats band by 2022 margin: <1 → *Lean BN gain*; <4 → *Tossup*;
  <9 → *Lean PH*; <20 → *Likely PH*; ≥20 → *Safe PH*.
- **R3** MUDA-held → *Tossup*.
- **R4** BN-held seats: *Safe BN* requires a **raw** margin ≥ ~14.4 pp.
- **R5** Below that, the Likely/Lean boundary uses a "de-split" effective
  margin: raw margin + (PAS-transfer × 2022 PN share) in Malay-majority
  UMNO seats — this is what turns 1-pp UMNO holds (Bukit Pasir, Parit Yaani,
  Serom) into *Likely BN*. Any transfer parameter between 0.3 and 0.7 gives the
  same 53/56: the table depends on the adjustment **existing**, not its size.
- **R6** MCA/MIC-defended seats in non-Malay-supermajority areas cap at
  *Likely BN* regardless of margin (why 26-pp Bekok is only "Likely").

The three seats the rules cannot produce are the report's genuine seat-level
judgment: **Simpang Jeram** (margin says Likely PH; the report called it Lean
BN *gain* on incumbent-death reasoning — this became its worst miss),
**Sungai Balang** (banding inconsistency: Safe at 12.6 pp while similar seats
at 11.5–11.9 are Likely), and **Tiram** (downgraded on candidate reasoning).

An asymmetry worth noting: BN-held seats need a ~14.4-pp margin to be "Safe",
PH-held seats need ~20 — defensible given the de-split dynamics, but never
stated, and it quietly bakes a pro-BN tilt into the band names.

### 4.2 The topline

The published tally (Safe BN 24 · Likely BN 14 · Lean BN 2 · Lean BN gain 5 ·
Tossup 3 · Lean PH 2 · Likely PH 2 · Safe PH 4) reproduces the toplines exactly:

- **BN-favoured = 45** ("it's at 45 before the tossups" ✓).
- **Central 46** = 45 − 1 (one unnamed PN "lean gain" handed back, giving
  PN ~1) + 2 (tossups Jementah and Tangkak booked for BN); MUDA holds its
  tossup (~1); PH keeps its 8. Total 56 ✓.
- **Ceiling 52** = 56 − the 4 Safe-PH fortresses. **Floor 42** = 45 locks − 3
  losses among MCA/MIC Chinese-mixed seats, all tossups lost.

Internal inconsistencies surfaced by making this explicit: the central row
books **PN ~1** against its own table (all three PN seats rated Lean BN
*gain*); the prose calls Johor Jaya a "true tossup" while the table rates it
Lean PH and the tally says "Tossup 3"; and the MUDA ~1 hold has no support in
the report's own poll (which showed the "others" bloc collapsing — and indeed
MUDA lost every deposit it staked).

## 5. Step 2: the quantitative layer that wasn't

The report reads the Vodus poll (fieldwork 15–29 Jun; BN 36 / PH 26 / PN 15 /
others 13 / won't-say 7 / undecided 2) against 2022 actuals and concludes
"BN down modestly (~7 points), PH roughly flat, PN collapsed (~9)".

| Reading of the same poll | BN | PH | PN |
|---|---|---|---|
| Raw shares vs 2022 actuals (the report's) | **−7.1** | −0.4 | −9.0 |
| Don't-knows reallocated | −3.5 | +2.2 | −7.5 |
| **Actual result** | **+16.9** | +6.6 | −18.6 |

Three problems, in increasing order of severity:

1. **Denominator error.** Comparing raw poll shares (which sum to 77 among
   named coalitions) against actual 2022 vote shares (which sum to ~97)
   guarantees phantom "declines". The 23-point remainder is described as
   "undecided and soft voters" when 13 points of it was **support for other
   parties** — a different thing, with different implications.
2. **Underdetermination.** Applied mechanically, the poll supports BN
   anywhere between ~37 and ~39 seats depending on unstated mapping choices
   (`analyst_report_sensitivity.py` Q2) — i.e. a *coin flip* on the 38-seat
   two-thirds line, and ~8 seats below the report's central 46. The
   "almost certainly two-thirds" headline cannot be derived from the anchor
   the report presents as its quantitative basis.
3. **Sign error ex post.** The poll itself missed BN's share by ~20 pp. The
   report's seat call survived **because** Steps 3–5 overrode Step 2 — the
   projection is presented as poll-anchored but is actually
   judgment-anchored. Had the poll been right, the report would have missed
   high by ~8 seats; it "won because the poll lost."

Uniform-swing counterfactual (reconstruction PART 4): applying the report's own
stated swing to its own baseline yields **BN 39 / PH 14 / PN 2 / MUDA 1** —
nowhere near the published BN ~46. Step 2 demonstrably contributed nothing to
the number the report is remembered for.

## 6. How it did against reality

Actual: **BN 48** (UMNO 36/37, MCA 8/15, MIC 4/4), **PH 8** (DAP 6 —
Bentayan, Penggaram, Mengkibol, Stulang, Skudai, Senai; AMANAH — Simpang
Jeram; PKR — Puteri Wangsa), **PN 0/33**, Bersama 0/15 and MUDA 0/4 with all
deposits lost. Statewide ≈ BN 60 / PH 33 / PN 5.4 on 69.6% turnout.

**Seat-level scorecard for the report:**

- Rated (non-tossup) seats: **50/53 correct**. Misses: Simpang Jeram (called
  BN gain; AMANAH held), Johor Jaya and Perling (called Lean PH; MCA took both).
- Tossups: Jementah → BN ✓ and Tangkak → BN ✓ (both booked for BN in the
  central 46); Puteri Wangsa → PKR (the central map booked a MUDA hold ✗).
- Implied central 56-seat map: **52/56 correct** — incidentally the same
  93% hit rate this repository's first-principles engine achieved in its
  2018→2022 backtest.
- All five named "most likely BN pickups": the three PN seats and Bukit Batu
  correct; Simpang Jeram wrong. The "BN downside" scenario (MCA/MIC seats
  falling) did not materialise — all six named risk seats held.

**Where the errors came from is the instructive part.** The three qualitative
mechanisms the report bet on (PN collapse flowing to BN, PH-marginal wobble,
Malay-seat de-splitting) were all directionally right — reality ran *past* the
report's central case on each, which is why the actual 48 beat its 46 from
above, and why its two Lean-PH misses (Johor Jaya, Perling) were its own
downside-for-DAP story overshooting its center. Its one clean judgment
override (Simpang Jeram) — the only place it *overruled* its own margin rules
on local intelligence — was wrong. Puteri Wangsa, the seat it declined to
call, is also the one seat no global parameterisation of 2026 can explain
(sensitivity Q3 matches 55/56 at best): a four-cornered candidate-effects
fight, exactly the class of seat where any statewide model bottoms out.

## 7. Sensitivity: which conclusions were robust?

`analyst_report_sensitivity.py` makes Steps 3–4 an explicit vote-flow engine
(PAS-transfer rate, PN retention, Chinese-marginal swing, turnout
differential) and sweeps it. Findings:

| Claim in the report | Verdict |
|---|---|
| BN wins, clears 29 "in every scenario I can construct" | **Robust.** Holds across the entire swept space, including a turnout differential twice as pro-PH as anything in Johor's history (majority never lost, even at ratio 0.5). First-past-the-post + a 3-way-split opposition does the work. |
| "Almost certainly" two-thirds (38+) | **Robust to mechanism uncertainty, NOT to the poll.** Every flow-parameter combination keeps BN ≥ 41; but every faithful mechanical reading of the Vodus poll puts BN 37–39, i.e. the claim required disbelieving the report's own quantitative anchor. |
| Central 46, range 42–52 | **Point estimate fragile, range good.** The engine spans 41–55 across plausible parameters; 46 sits in the conservative half; the actual 48 lands in the densest region. Reaching the published floor of 42 requires the Chinese-mixed seats to swing *against* MCA/MIC (the report's stated downside) or PAS transfers to fail almost entirely. |
| Turnout as "the biggest wildcard" | **Overstated.** Turnout differentials move the margin, not the outcome: two-thirds survives down to a 40%-relative pro-PH differential. And the sign of the worry was wrong ex post: turnout rose ~15 pp and *helped BN* (rural Malay turnout 66–68% vs urban ~60%). |
| PAS voters transfer "real but imperfect" | **Right, and undersold.** Best-fitting parameters imply the PN→BN consolidation ran at or beyond the report's assumption; PN retained only ~22% of its 2022 vote statewide. |
| The unpublished PN contest map matters | **No.** Whether PN's 33 seats are placed at its strongest seats, everywhere, or nowhere moves BN by ±1 seat. Not a material gap in reconstruction. |

## 8. Overall methodological assessment

**Strengths (genuine, and worth copying):**

1. Perfect baseline hygiene, with a published, verifiable seat table — rare
   and commendable; it made this reconstruction possible at all.
2. The political reads were the alpha: PN-held seats falling, PAS transfer
   favouring BN's *marginal* Malay seats (the report's "counter-intuitive
   core"), PH's urban-Chinese firewall floor of 6–8. All correct, and the
   firewall/floor logic was exactly what happened (PH won 8: the 6-seat DAP
   floor plus two candidate-effect seats).
3. Honest flagging: assumptions labelled load-bearing, competitive seats
   pre-identified (all four central-map errors were inside its own ~15-seat
   watchlist), and the limitation note ("ratings are analytical judgment, not
   measured data") is accurate.
4. Outcome: best published forecast of this election on both topline and map.

**Weaknesses (what "robust" would have required):**

1. **The quantitative dressing is ornamental.** The poll-swing layer points
   the wrong way, is internally underdetermined (§5), and is overridden
   silently. A reader cannot tell which steps produced which numbers — the
   reason this reconstruction had to be reverse-engineered.
2. **No uncertainty model.** "Lean/Likely/Safe" carry no probabilities; the
   central projection hedges against its own table (PN ~1, MUDA ~1) in ways
   the stated model cannot generate; ranges come from scenario arithmetic,
   not from propagating input uncertainty.
3. **Asymmetric banding** (BN Safe at ≥14.4 pp raw, PH Safe at ≥20) is never
   disclosed as a modelling choice.
4. **The sanity check anchored on worse forecasts.** Step 6 validated against
   Vodus (BN floor missed by 28 seats) and Ong's scenarios (seat error 10 and
   18). Agreement with bad forecasts is not evidence; the report was better
   than its own benchmarks and didn't know it.
5. **Candidate-fact errors in tossup seats** (§3) — the layer where judgment
   is supposed to add value over mechanics.
6. **Non-reproducibility as published**: no code, no formulas, no stated
   thresholds. This examination shows ~95% of it *was* formalisable — there
   was no reason not to publish the rules.

**Bottom line.** As election *analysis*, the report deserves high marks: the
three political judgments that mattered were right, made early, and stated
plainly. As a *model*, it is a margin-threshold classifier whose quantitative
layer failed silently and whose headline confidence ("almost certainly
two-thirds") was an expert prior wearing a poll costume. The honest summary
of its own performance would be: *the analyst beat their model, the model
beat its poll, and the poll missed the election.*

## 9. If this repo wants to run the same exercise forward

1. Keep the ratings architecture (it's transparent and it worked), but
   publish the rules — PART 2 of the reconstruction is a working draft.
2. Split Step 2 into explicit branches: "poll-true world" vs "transfer world"
   (sensitivity Q2 shows they imply different *headlines*, not just different
   seat counts), and state which branch the central case sits in and why.
3. Replace the topline hedges (PN ~1, MUDA ~1) with per-seat probabilities —
   even crude band-level ones (Safe .95 / Likely .8 / Lean .65 / Tossup .5)
   would have produced a distribution centred near 47 with honest tails.
4. Feed the same scenario into `first_principles_prediction_engine.py`
   (support-rate × turnout + persistent local factor) as a structurally
   different cross-check: two models agreeing for different reasons is a far
   stronger sanity check than agreement with other pundits.
5. Ingest the official 2026 per-seat results into `DATA/` when available
   (harmonised like the other years) — that unlocks the parameter
   identification the sensitivity fit currently lacks (Q3 caveats), and a
   proper 2022→2026 backtest of the first-principles engine.

## Appendix: sources for 2026 facts

Compiled 2026-08-21. 2022-and-earlier figures are from this repository's
`DATA/` only.

- Results and component tallies: The Edge Malaysia ("BN wins bigger majority
  in 2026 Johor polls; wipeout for PN, Bersama and Muda", node/810332);
  Malaysiakini 779467; FMT "OFFICIAL: BN wins 48 seats, PH 8".
- PH seat identities: FMT "DAP survives Johor test as PKR, Amanah falter"
  (12 Jul 2026).
- MCA gains incl. Jementah detail: FMT "MCA wins 8 seats in Johor,
  surpassing DAP's 6" (11 Jul 2026).
- PN wipeout and contest split (Bersatu 16 / PAS 11 / MIPP 5 / Pejuang 1 of
  33): Malay Mail "Perikatan wiped out in Johor…" (12 Jul 2026); nomination
  round-ups (The Sun, NST, 27 Jun 2026 — BN: UMNO 37/MCA 15/MIC 4; PH:
  PKR 20/AMANAH 19/DAP 17; 172 candidates).
- Bersama 15 candidates and deposit losses: Malay Mail (27 Jun 2026); FMT
  "Rafizi confirms all Bersama candidates lost their deposits" (12 Jul 2026).
- Statewide shares (~60/33/5.4), turnout 69.57%, ethnic-turnout and
  Chinese-support estimates (PH ~75% vs ~85% at GE15): RSIS "Assessment and
  Early Analysis of the 2026 Johor State Election Results"; SCMP (12 Jul 2026).
- Vodus poll breakdown (36/26/15/13/7/2, n=1,303, 15–29 Jun, OMTOS,
  DUN-level post-stratification) and its seat model (BN 20: 17 safe + 3
  marginal; PH 1; PN 3; 31 undecided): Sinar Daily article 737711; vodus.com.
