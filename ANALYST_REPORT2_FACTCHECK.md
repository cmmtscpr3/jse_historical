# Fact-Check of the Analyst's Second Report (Swing-Seat Deep-Dive, "Tindak data")

> **What this document is.** The same analyst produced a follow-up report
> examining the 15 seats "that genuinely determine the size of BN's win" —
> demographics attributed to Tindak Malaysia's 2026 dataset, 2026 candidate
> line-ups attributed to the SPR post-nomination list, and 2022 results per
> seat. This examination checks (a) every 2022 claim, demographic figure,
> electorate count, and urbanisation label against this repository's
> harmonised data (`verify_report2_swing_seats.py`), and (b) the 2026
> candidate claims against contemporary reporting, then scores the seat reads
> against the actual 11 July result. Companion to `ANALYST_REPORT_FINDINGS.md`
> (the examination of the first report).

## 1. Verdict up front

**Substantially accurate and analytically stronger than the first report —
with one clear factual error (Johor Jaya's demographics), two mildly
overstated footnotes, and the same repeated misjudgment (Simpang Jeram).**

| Layer | Verdict |
|---|---|
| 2022 results (15 seats: names, parties, shares, majorities) | **Perfect.** 44 vote shares reproduce to ≤0.05 pp; every candidate name, PN component party (Bersatu/PAS/Gerakan per seat), and majority is exact; even the subtle implication that no PH candidate stood in Bukit Kepong in 2022 is correct. |
| 2026 candidate line-ups | **Every checked claim confirmed** — including the Endau defection, both BN reshuffles, the Bukit Batu five-way, the Johor Jaya four-way, the Tangkak straight fight, and the BN candidates in Simpang Jeram, Maharani, Bekok, Yong Peng and Pekan Nanas. It also silently corrects both candidate errors in the analyst's first report. |
| Demographics (Table A) | **14 of 15 seats match** the official roll within normal 2022→2026 drift (≤ ~2 pp). **Johor Jaya's row is wrong** (details in §3). Two Orang Asli footnotes are overstated by ~1.5–2 pp. |
| Electorate counts | Consistent with a genuine 2026 roll: 12 seats grew (up to +14.1% in Puteri Wangsa, plausible for a booming suburb), 3 aged rural seats shrank slightly. Not recycled 2022 numbers. |
| Urbanisation labels | The analyst's own classification (disclosed as such) differs from the official 2021 classification in **5 of 15 seats** — and the official labels sit in this repo, in the file the analyst said they couldn't access. |
| Seat reads vs the actual result | 11 of 13 directional reads right; the two misses are Simpang Jeram (again) and the "narrow DAP edge" in Jementah. |

## 2. What was verified and how

### 2022 layer (against `DATA/JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv`)

All 15 seats pass every check: winner and runner-up names (including
roll-style short forms like "SARAS" for Saraswathy Nallathanby), component
parties, vote shares (max error 0.05 pp over 44 checked shares), majorities
(all exact, including Bukit Batu's 137), and third-party shares (including
MUDA's 13.8% in Bukit Kepong and Gerakan as PN's candidate party in Bukit
Batu, Johor Jaya, Puteri Wangsa, Yong Peng and Pekan Nanas). This layer is
flawless.

### 2026 candidate layer (against contemporary reporting)

Confirmed, with sources in §6:

- **Endau — the "standout story" is true.** Incumbent Alwiyah Talib quit
  Bersatu on 4 June 2026, rejoined UMNO, was named BN's Endau candidate, and
  won with 51.6%. One omitted nuance: she was originally UMNO (she defected
  *to* Bersatu in 2018), so this was a return, not a first crossing — and
  reporting noted grassroots friction over her candidacy, not just
  "strengthening BN's flank".
- **Both BN reshuffles are real**: Raven Kumar (three-term Tenggaroh ADUN)
  moved to Kemelah and won by 3,838; Mohd Youzaimi Yusof (BN's 2022 Endau
  candidate) took over Tenggaroh and won by 15,258. Kemelah's PN candidacy
  did switch from PAS (2022) to Bersatu's Uzzair Ismail, as stated.
- **Bukit Kepong three-way confirmed** — including the detail that PH-PKR's
  C. Subramani is the same Subramani who contested Buloh Kasap in 2022.
  Ex-MB Sahruddin Jamal defended and collapsed to third-adjacent (5,625
  votes vs BN's 16,386; BN majority 10,761).
- **Bukit Batu five-way confirmed** name-for-name (one of three five-cornered
  fights statewide). MIC's R.K. Kumaran won by **174 votes** — the report's
  vote-splitting mechanism playing out almost exactly (a 137-vote PH majority
  flipped to a 174-vote BN one).
- **Johor Jaya four-way confirmed** name-for-name (DAP's new face Lee Wern
  Yiing, MCA's returning Chan San San — winner by 7,268 — Bersama's Lau Yi
  Leong, one independent).
- **Tangkak straight fight confirmed** (Ee Chin Li vs Haw Chin Teck; BN won
  by 3,182).
- **BN candidates in Simpang Jeram (Azman Ismail), Maharani (Ashari Md
  Sarip), Bekok (Tan Chong), Yong Peng (Ling Tian Soon), Pekan Nanas (Tan
  Eng Meng)** all confirmed.
- **MUDA's four seats** (Puteri Wangsa, Bukit Batu confirmed directly;
  Simpang Jeram and Maharani forced by the MUDA-fielded-4 arithmetic) match
  the report's placements exactly.

Not individually verified (but consistent with the 172-candidate arithmetic
and MIPP's 5-seat allocation): Paloh's MIPP and independent candidates, the
Endau ASLI candidate and the "first ASLI outing in Johor" footnote, and
Puteri Wangsa's independent.

## 3. The one clear factual error: Johor Jaya's demographics

Table A gives Johor Jaya **Malay 35 / Chinese 45 / Indian 8 / other 12**,
with a highlighted footnote of **~8.5% East Malaysian Bumiputera (Sabah 4.6
+ Sarawak 3.9)**. The official roll says Malay 42.5 / Chinese 46.2 / Indian
7.6 / other 3.8, with **2.8%** East Malaysian Bumiputera.

This cannot be explained as 2022→2026 roll drift. With the claimed +6.3%
electorate growth, the claimed shares would require Johor Jaya to *lose*
~4,900 Malay electors (−12%) while growing overall, and East Malaysian
registration to *triple* in four years (+~5,700 people ≈ every single new
registration). Nor is it a row-slip from a neighbouring constituency — no
Johor DUN has a composition close to the claimed figures (nearest match is
Johor Jaya's own official row). Whether the analyst misread the Tindak sheet
or the sheet itself is wrong is indeterminate from here, but the figure is
wrong, and the report builds a distinguishing "colour" claim on it (the
East-Malaysian share is flagged twice as *notable*). The seat's rating
didn't depend on it — but it is exactly the kind of vivid, specific,
uncheckable-sounding statistic that deserves the most suspicion.

Two smaller wobbles in the same column: Endau's Orang Asli share is given as
~5% (roll: 3.2%) and Bekok's as ~3.5% (roll: 2.1%) — both overstated beyond
plausible drift, both in the direction that flatters the seat's
"distinctiveness".

Every other demographic cell in Table A tracks the official roll within
~2 pp, with a consistent slight Chinese decline and "other" uptick across
seats — the signature of a *genuine* updated roll rather than recycled or
invented numbers. The same goes for the electorate counts (moderate growth,
small declines in three ageing rural seats, +14% in the fastest-growing
suburb). In short: the data work is real; one row of it is bad.

## 4. The urbanisation column

The analyst discloses that Tindak's urban–rural field was unpopulated and
substitutes their own classification "corroborated by each seat's
registered-elector count". Checked against the official 2021 classification
(which sits in `DATA/JOHOR_2022_DUN_COMPOSITION_HARMONISED.csv`, the file
equivalent of the source the analyst lacked), **5 of 15 labels differ** —
Simpang Jeram and Maharani are officially *Urban* (labelled Semi-urban),
Endau, Kemelah and Bekok are officially *Semi-urban* (labelled Rural). The
elector-count heuristic is also contradicted by the report's own table
(rural Tenggaroh, 39,001 electors, outweighs semi-urban Yong Peng, 34,023).
Because the labels are only used as narrative colour, the impact is
cosmetic — but the systematic rural-ward skew slightly exaggerates the
"two different worlds" framing in the report's opening pattern claim.

## 5. Soundness of the analysis itself

**Scored against the actual result** (13 directional reads, excluding the
two seats left as pure tossups):

- Right: Bukit Kepong, Endau, Maharani, Bukit Batu ("likely/lean BN gain" —
  all four fell to BN), Johor Jaya ("tossup, leaning slightly BN" — BN by
  7,268), and all six Group-3 holds (Kemelah, Bekok, Yong Peng, Paloh,
  Tenggaroh, Pekan Nanas — every one held, with Tenggaroh's "Likely BN
  despite the modest 2022 margin" vindicated by a 15,258 majority).
- Wrong: **Simpang Jeram** ("lean BN gain"; AMANAH held — the same
  local-intelligence override that was the first report's worst miss,
  repeated with more conviction), and **Jementah** ("tossup, narrow DAP
  edge"; MCA won by 913 — the one place the update nudged a rating in the
  wrong direction).
- The two declared tossups: Tangkak went BN by a clearer-than-tossup 3,182;
  Puteri Wangsa went to PKR — and "PH-consolidation" was explicitly one of
  the report's three scenarios.

**Where it improves on the first report.** It corrects both of the first
report's candidate errors (Puteri Wangsa's MUDA candidate; Jementah's BN
party, MCA not MIC); it moves Johor Jaya from "Lean PH" to
"tossup-leaning-BN" (the right direction — this was the first report's
biggest rating error); it upgrades Endau and Bukit Kepong on genuine new
information (the defection; PN's lone-PAS or lone-Bersatu candidacies); and
its candidate-level texture (reshuffles, five-way splits, incumbent
retirements) is exactly the layer the first report got wrong. As
candidate-level intelligence, this is the stronger document.

**Where the caution from the first examination still applies.** The scoping
choice aged badly in one spot: the report explicitly parks Perling as "just
outside this 15" where a swing "could theoretically matter" — Perling was
the one seat that actually flipped outside its frame (8 of the 9 actual
flips were inside the 15; Perling was the ninth). Simpang Jeram shows the
recurring pattern identified in the first examination: when this analyst
overrides margin logic with local intelligence, the override — not the
model — is where the misses come from. And the ratings remain
probability-free judgments, so "lean/likely" cannot be scored more sharply
than direction.

## 6. Sources for the 2026 verification

Compiled 2026-08-21. All 2022 and demographic baselines from this
repository's `DATA/` only.

- Endau defection and result: The Star (4 Jun 2026), Sinar Daily, FMT
  (27 Jun 2026), Malaysiakini 779001.
- Bukit Kepong line-up and result: pilihanraya.my N07; Kosmo (27 Jun 2026);
  official results round-ups (Harian Metro, Malaysia Gazette).
- Kemelah / Tenggaroh reshuffle and results: pilihanraya.my N04, N33;
  BULETIN TV3 candidate and results lists.
- Bukit Batu five-way and 174-vote result: Malaysia Gazette (27 Jun 2026);
  Sinar Harian (post-result); Kosmo "MIC sapu bersih 4 kerusi DUN".
- Johor Jaya and Tangkak line-ups/results: undi.ecentral.my N10; Bernama;
  pilihanraya.my (Chan San San candidate page).
- BN candidate list (Azman Ismail, Ashari Md Sarip, Tan Chong, Ling Tian
  Soon, Tan Eng Meng): UMNO Online / Utusan / Kosmo BN candidate lists
  (24 Jun 2026); Sinar Harian 172-candidate list (27 Jun 2026).
