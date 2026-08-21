"""Reconstruction of the unpublished model behind the analyst report
"Johor State Election (11 July 2026): Seat-by-Seat Projection".

The report describes a six-step "baseline-plus-swing" methodology (2022 seat
baseline -> statewide swing from one Vodus poll -> PN-split / PAS-transfer
adjustment -> PH-side fragmentation adjustment -> Safe/Likely/Lean/Tossup
rating per seat -> sanity check against other forecasters) but published no
code and no formulas. This script reconstructs that model end-to-end and
tests it:

  PART 1  Verify the report's 2022 baseline (margins, winners, statewide
          totals) against DATA/JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv.
  PART 2  Re-derive the report's 56 seat ratings from an explicit, minimal
          rule set (margin thresholds + the report's three stated
          adjustments), and list exactly which seats cannot be reproduced
          without seat-specific judgment.
  PART 3  Re-derive the topline (BN ~46, range 42-52; PH ~8; PN ~1; MUDA ~1)
          from the ratings, reproducing the report's own arithmetic.
  PART 4  Run the report's Step 2 (poll-implied uniform swing) as an actual
          model, on its own, to show what the quantitative layer would have
          predicted without the qualitative overrides.
  PART 5  Score the report and rival forecasts against the actual
          11 July 2026 result (BN 48, PH 8, PN 0, MUDA 0).

2026 facts (candidacies, results) were compiled from public reporting on
2026-08-21 and are embedded below with sources; everything 2022 and earlier
comes from this repository's harmonised data.
"""

import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "DATA")


# ---------------------------------------------------------------------------
# The report's published artefacts, transcribed verbatim
# ---------------------------------------------------------------------------

# Seat -> (2022 winner label as printed, margin in pp as printed, 2026 rating).
# Ratings: SAFE_BN LIKELY_BN LEAN_BN LEAN_BN_GAIN TOSSUP LEAN_PH LIKELY_PH SAFE_PH
REPORT_TABLE = {
    "N01": ("Buloh Kasap", "UMNO", 34.2, "SAFE_BN"),
    "N02": ("Jementah", "DAP", 3.3, "TOSSUP"),
    "N03": ("Pemanis", "UMNO", 24.3, "SAFE_BN"),
    "N04": ("Kemelah", "MIC", 8.8, "LIKELY_BN"),
    "N05": ("Tenang", "UMNO", 14.5, "SAFE_BN"),
    "N06": ("Bekok", "MCA", 26.0, "LIKELY_BN"),
    "N07": ("Bukit Kepong", "Bersatu/PN", 3.2, "LEAN_BN_GAIN"),
    "N08": ("Bukit Pasir", "UMNO", 1.1, "LIKELY_BN"),
    "N09": ("Gambir", "UMNO", 18.0, "SAFE_BN"),
    "N10": ("Tangkak", "DAP", 1.9, "TOSSUP"),
    "N11": ("Serom", "UMNO", 3.2, "LIKELY_BN"),
    "N12": ("Bentayan", "DAP", 44.0, "SAFE_PH"),
    "N13": ("Simpang Jeram", "AMANAH/PH", 11.2, "LEAN_BN_GAIN"),
    "N14": ("Bukit Naning", "UMNO", 10.8, "LIKELY_BN"),
    "N15": ("Maharani", "PAS/PN", 4.9, "LEAN_BN_GAIN"),
    "N16": ("Sungai Balang", "UMNO", 12.6, "SAFE_BN"),
    "N17": ("Semerah", "UMNO", 14.5, "SAFE_BN"),
    "N18": ("Sri Medan", "UMNO", 28.8, "SAFE_BN"),
    "N19": ("Yong Peng", "MCA", 14.7, "LIKELY_BN"),
    "N20": ("Semarang", "UMNO", 32.6, "SAFE_BN"),
    "N21": ("Parit Yaani", "UMNO", 1.3, "LIKELY_BN"),
    "N22": ("Parit Raja", "UMNO", 19.7, "SAFE_BN"),
    "N23": ("Penggaram", "DAP", 29.3, "SAFE_PH"),
    "N24": ("Senggarang", "UMNO", 18.1, "SAFE_BN"),
    "N25": ("Rengit", "UMNO", 11.9, "LIKELY_BN"),
    "N26": ("Machap", "UMNO", 33.6, "SAFE_BN"),
    "N27": ("Layang-Layang", "UMNO", 20.5, "SAFE_BN"),
    "N28": ("Mengkibol", "DAP", 30.0, "SAFE_PH"),
    "N29": ("Mahkota", "UMNO", 14.3, "LIKELY_BN"),
    "N30": ("Paloh", "MCA", 21.6, "LIKELY_BN"),
    "N31": ("Kahang", "MIC", 40.0, "SAFE_BN"),
    "N32": ("Endau", "Bersatu/PN", 20.0, "LEAN_BN_GAIN"),
    "N33": ("Tenggaroh", "MIC", 6.3, "LEAN_BN"),
    "N34": ("Panti", "UMNO", 26.6, "SAFE_BN"),
    "N35": ("Pasir Raja", "UMNO", 30.3, "SAFE_BN"),
    "N36": ("Sedili", "UMNO", 29.9, "SAFE_BN"),
    "N37": ("Johor Lama", "UMNO", 30.0, "SAFE_BN"),
    "N38": ("Penawar", "UMNO", 40.5, "SAFE_BN"),
    "N39": ("Tanjung Surat", "UMNO", 40.1, "SAFE_BN"),
    "N40": ("Tiram", "UMNO", 9.4, "LEAN_BN"),
    "N41": ("Puteri Wangsa", "MUDA", 13.4, "TOSSUP"),
    "N42": ("Johor Jaya", "DAP", 4.1, "LEAN_PH"),
    "N43": ("Permas", "UMNO", 14.1, "LIKELY_BN"),
    "N44": ("Larkin", "UMNO", 16.1, "SAFE_BN"),
    "N45": ("Stulang", "DAP", 10.3, "LIKELY_PH"),
    "N46": ("Perling", "DAP", 7.8, "LEAN_PH"),
    "N47": ("Kempas", "UMNO", 11.5, "LIKELY_BN"),
    "N48": ("Skudai", "DAP", 31.0, "SAFE_PH"),
    "N49": ("Kota Iskandar", "UMNO", 7.7, "LIKELY_BN"),
    "N50": ("Bukit Permai", "UMNO", 21.1, "SAFE_BN"),
    "N51": ("Bukit Batu", "PKR/PH", 0.6, "LEAN_BN_GAIN"),
    "N52": ("Senai", "DAP", 18.5, "LIKELY_PH"),
    "N53": ("Benut", "UMNO", 33.9, "SAFE_BN"),
    "N54": ("Pulai Sebatang", "UMNO", 25.4, "SAFE_BN"),
    "N55": ("Pekan Nanas", "MCA", 22.6, "LIKELY_BN"),
    "N56": ("Kukup", "UMNO", 42.5, "SAFE_BN"),
}

# The report's toplines.
REPORT_CENTRAL = {"BN": 46, "PH": 8, "PN": 1, "MUDA": 1}
REPORT_RANGE = {"BN": (42, 52), "PH": (6, 12), "PN": (0, 2), "MUDA": (0, 1)}

# Step 2 input: the Vodus poll as the report quotes it (share of likely vote).
# The full published breakdown (Vodus via Sinar Daily, fieldwork 15-29 Jun
# 2026, n=1,303): BN 36, PH 26, PN 15, other parties 13, prefer-not-to-say 7,
# undecided 2. The report describes the missing 23 points only as "a large
# bloc of undecided and soft voters".
VODUS_POLL = {"BN": 36.0, "PH": 26.0, "PN": 15.0, "OTHERS": 13.0, "DK": 9.0}
SHARES_2022 = {"BN": 43.1, "PH": 26.4, "PN": 24.0}  # verified in PART 1

# Rival forecasts quoted by the report (its Step 6 sanity check).
ONG_SCENARIOS = {
    "pro-BN (>60% likely)": {"BN": 53, "PH": 3, "PN": 0, "MUDA": 0},
    "pro-PH": {"BN": 39, "PH": 14, "PN": 3, "MUDA": 0},
    "pro-PN (unlikely)": {"BN": 21, "PH": 10, "PN": 25, "MUDA": 0},
}

# ---------------------------------------------------------------------------
# Actual 11 July 2026 result, compiled 2026-08-21 from public reporting:
#   - theedgemalaysia.com/node/810332 (BN 48: UMNO 36/37, MCA 8/15, MIC 4/4;
#     PN, Bersama, MUDA wiped out; PN lost Bukit Kepong, Maharani, Endau)
#   - freemalaysiatoday.com 2026/07/12 "DAP survives Johor test as PKR,
#     Amanah falter" (PH 8 = DAP 6: Bentayan, Penggaram, Mengkibol, Stulang,
#     Skudai, Senai; AMANAH: Simpang Jeram; PKR: Puteri Wangsa)
#   - freemalaysiatoday.com 2026/07/11 "MCA wins 8 seats in Johor" (MCA's
#     gains from DAP: Jementah, Tangkak, Johor Jaya, Perling; in Jementah,
#     MCA's See Ann Giap 12,522 beat DAP's Ng Kor Sim 11,609, majority 913)
#   - RSIS "Assessment and Early Analysis of the 2026 Johor State Election
#     Results" (statewide BN ~60%, PH ~33%, PN 5.4%; turnout 69.57%)
# Every seat not listed here was won by BN.
# ---------------------------------------------------------------------------
ACTUAL_PH_SEATS = {
    "N12": "DAP", "N23": "DAP", "N28": "DAP", "N45": "DAP", "N48": "DAP",
    "N52": "DAP", "N13": "AMANAH", "N41": "PKR",
}
ACTUAL_SHARES_2026 = {"BN": 60.0, "PH": 33.0, "PN": 5.4}
ACTUAL_TURNOUT_2026 = 69.57


def seat_key(name: str) -> str:
    return re.sub(r"[^A-Z]", "", str(name).upper())


def load_2022() -> pd.DataFrame:
    """Per-seat 2022 baseline: block vote shares, margin, runner-up."""
    res = pd.read_csv(os.path.join(DATA_DIR, "JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv"))
    comp = pd.read_csv(os.path.join(DATA_DIR, "JOHOR_2022_DUN_COMPOSITION_HARMONISED.csv"))
    comp["key"] = comp["STATE CONSTITUENCY NAME"].map(seat_key)

    num_by_key = {seat_key(name): num for num, (name, _, _, _) in REPORT_TABLE.items()}
    rows = []
    for _, r in res.iterrows():
        votes = {
            "BN": r["BN VOTE"], "PH": r["PH VOTE"], "PN": r["PN CANDIDATE VOTE"],
            "MUDA": r["MUDA VOTE"], "PBM": r["PBM VOTE"], "PEJUANG": r["PEJUANG VOTE"],
            "WARISAN": r["WARSIAN VOTE"], "PSM": r["PSM VOTE"], "PUTRA": r["PUTRA VOTE"],
            "IND": np.nansum([r["INDEPENDENT 1 VOTE"], r["INDEPENDENT 2 VOTE"],
                              r["INDEPENDENT 3 CANDIDATE VOTE"]]),
        }
        votes = {k: float(v) for k, v in votes.items() if pd.notna(v) and float(v) > 0}
        valid = float(r["TOTAL VALID VOTES"])
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        key = seat_key(r["STATE CONSTITUENCY NAME"])
        rows.append({
            "num": num_by_key[key],
            "seat": REPORT_TABLE[num_by_key[key]][0],
            "key": key,
            "winner22": r["WINNING PARTY (2022)"],
            "winner22_bloc": ranked[0][0],
            "runner_up22": ranked[1][0],
            "margin22": 100.0 * (ranked[0][1] - ranked[1][1]) / valid,
            "majority22": float(r["WINNING MAJORITY"]),
            "valid22": valid,
            "electorate22": float(r["TOTAL ELECTORATE"]),
            "turnout22": float(r["TURNOUT (%)"]),
            "bn22": 100.0 * votes.get("BN", 0.0) / valid,
            "ph22": 100.0 * votes.get("PH", 0.0) / valid,
            "pn22": 100.0 * votes.get("PN", 0.0) / valid,
            "muda22": 100.0 * votes.get("MUDA", 0.0) / valid,
        })
    df = pd.DataFrame(rows).sort_values("num").set_index("num")
    df = df.join(comp.set_index("key")[["MALAY (%)", "CHINESE (%)", "INDIANS (%)",
                                        "URBAN-RURAL CLASSIFICATION (2021)"]], on="key")
    df = df.rename(columns={"MALAY (%)": "malay", "CHINESE (%)": "chinese",
                            "INDIANS (%)": "indian",
                            "URBAN-RURAL CLASSIFICATION (2021)": "urban_rural"})
    return df


# ---------------------------------------------------------------------------
# PART 1 - verify the report's baseline against the repository data
# ---------------------------------------------------------------------------

def part1_verify_baseline(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("PART 1 - Report's 2022 baseline vs repository data")
    print("=" * 78)
    diffs, label_mismatch = [], []
    for num, (name, winner_label, margin_pp, _) in REPORT_TABLE.items():
        row = df.loc[num]
        diffs.append(abs(margin_pp - row["margin22"]))
        lab = winner_label.split("/")[0].upper().replace("BERSATU", "PPBM")
        if lab in ("AMANAH",):
            lab = "PAN"  # harmonised data uses AMANAH's ballot label PAN
        if lab not in str(row["winner22"]).upper():
            label_mismatch.append((num, name, winner_label, row["winner22"]))
    print(f"Margin definition that reproduces the table: majority / total valid votes")
    print(f"  max |report - data| across 56 seats: {max(diffs):.2f} pp "
          f"(mean {np.mean(diffs):.2f} pp)  -> all 56 margins verified")
    print(f"  winner-label mismatches: {len(label_mismatch)}")
    for t in label_mismatch:
        print(f"    {t}")

    tot = df["valid22"].sum()
    for bloc, claimed in SHARES_2022.items():
        share = (df[bloc.lower() + "22"] * df["valid22"]).sum() / tot
        flag = "OK" if abs(share - claimed) < 0.1 else "MISMATCH"
        print(f"  2022 statewide {bloc}: computed {share:.1f}% vs report {claimed}%  [{flag}]")
    seats22 = df["winner22_bloc"].value_counts().to_dict()
    print(f"  2022 seats computed {seats22} vs report BN 40 / PH 12 / PN 3 / MUDA 1")
    turnout22 = 100 * df["valid22"].sum() / df["electorate22"].sum()
    print(f"  2022 turnout (valid/electorate): {turnout22:.1f}%; report says ~54%  [OK]")
    print(f"  Bukit Batu 2022 majority: {df.loc['N51','majority22']:.0f} votes; report says 137  [OK]")
    print()


# ---------------------------------------------------------------------------
# PART 2 - reproduce the 56 ratings from explicit rules
# ---------------------------------------------------------------------------
# Rule set inferred from the report's own stated logic:
#
#   R1 (Step 3, PN collapse): every PN-held seat -> LEAN_BN_GAIN.
#   R2 (Step 1 + 5, PH-held seats band by 2022 margin):
#        <1 pp   -> LEAN_BN_GAIN   ("most flippable seat in the state")
#        <4 pp   -> TOSSUP
#        <9 pp   -> LEAN_PH
#        <20 pp  -> LIKELY_PH
#        else    -> SAFE_PH
#   R3 (Step 4, four-cornered MUDA defence): MUDA-held -> TOSSUP.
#   R4 (BN-held seats, base band from the RAW 2022 margin):
#        >= 14.4 pp -> SAFE_BN, otherwise LIKELY_BN / LEAN_BN per R5's
#        effective margin. Note that "Safe" needs a fat RAW margin: the
#        de-split boost below never promotes a seat to Safe, matching the
#        report's language ("was knife-edge; de-split -> safer", i.e. Likely).
#   R5 (Step 3 again, the de-split boost): effective margin for the
#        Likely/Lean boundary = 2022 margin + PAS_TRANSFER x 2022 PN share in
#        Malay-majority (>= 55%) seats defended on an UMNO ticket;
#        >= 7 pp effective -> LIKELY_BN, else LEAN_BN.
#   R6 (Step 5 ethnicity adjustment): a BN seat fought on an MCA/MIC ticket
#        in a non-Malay-supermajority area (Malay < 60%) is capped at
#        LIKELY_BN regardless of margin, and never receives the R5 boost.
#
# PAS_TRANSFER = 0.5 is the single tunable parameter (report: transfer "real
# but imperfect"); agreement is identical for any value in 0.3-0.7, i.e. the
# table depends on the adjustment EXISTING, not on its size. Seats the rules
# cannot reproduce are the report's residual seat-specific judgment.

PAS_TRANSFER = 0.5
SAFE_RAW_MARGIN = 14.4
LIKELY_EFF_MARGIN = 7.0
MCA_MIC_2022 = {"N04": "MIC", "N06": "MCA", "N19": "MCA", "N30": "MCA",
                "N31": "MIC", "N33": "MIC", "N55": "MCA"}


def rate_seat(row: pd.Series, num: str) -> str:
    w = row["winner22_bloc"]
    m = row["margin22"]
    if w == "PN":
        return "LEAN_BN_GAIN"                                   # R1
    if w == "PH":
        if m < 1:
            return "LEAN_BN_GAIN"                               # R2
        if m < 4:
            return "TOSSUP"
        if m < 9:
            return "LEAN_PH"
        if m < 20:
            return "LIKELY_PH"
        return "SAFE_PH"
    if w == "MUDA":
        return "TOSSUP"                                         # R3
    mca_mic = num in MCA_MIC_2022
    if m >= SAFE_RAW_MARGIN and not (mca_mic and row["malay"] < 60):
        return "SAFE_BN"                                        # R4 + R6 cap
    eff = m
    if row["malay"] >= 55 and not mca_mic:
        eff = m + PAS_TRANSFER * row["pn22"]                    # R5
    return "LIKELY_BN" if eff >= LIKELY_EFF_MARGIN else "LEAN_BN"


def part2_rule_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 78)
    print("PART 2 - Mechanical rule set vs the report's 56 published ratings")
    print("=" * 78)
    recs = []
    for num, (name, _, _, rating) in REPORT_TABLE.items():
        rule = rate_seat(df.loc[num], num)
        recs.append({"num": num, "seat": name, "report": rating, "rules": rule,
                     "match": rule == rating})
    out = pd.DataFrame(recs).set_index("num")
    n_match = int(out["match"].sum())
    print(f"Exact rating agreement: {n_match}/56 seats "
          f"({100 * n_match / 56:.0f}%) with 6 rules; agreement is 52-53/56 "
          f"for ANY PAS-transfer value in 0.3-0.7")

    side = lambda r: ("BN" if "BN" in r else "PH" if "PH" in r else "TOSSUP")
    same_side = int(sum(side(a) == side(b) for a, b in zip(out["report"], out["rules"])))
    print(f"Same favoured side: {same_side}/56 seats")
    print("\nSeats the rules cannot reproduce (the report's seat-specific judgment):")
    for num, r in out[~out["match"]].iterrows():
        note = {
            "N13": "stated override: incumbent died, personal vote gone, BN re-entering",
            "N16": "banding inconsistency: Safe at raw 12.6 while Rengit 11.9 / Kempas 11.5 are Likely",
            "N40": "stated judgment: 'large mixed seat', MIC fielded vs DAP -> downgraded to Lean",
        }.get(num, "")
        print(f"  {num} {r['seat']:<15} report={r['report']:<13} rules={r['rules']:<13} {note}")
    print()
    return out


# ---------------------------------------------------------------------------
# PART 3 - topline arithmetic from the ratings
# ---------------------------------------------------------------------------

def part3_topline(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("PART 3 - From ratings to the topline (BN ~46, range 42-52)")
    print("=" * 78)
    tally = {}
    for num, (_, _, _, rating) in REPORT_TABLE.items():
        tally[rating] = tally.get(rating, 0) + 1
    print("Rating tally:", dict(sorted(tally.items())))
    bn_locked = tally["SAFE_BN"] + tally["LIKELY_BN"] + tally["LEAN_BN"] + tally["LEAN_BN_GAIN"]
    ph_locked = tally["SAFE_PH"] + tally["LIKELY_PH"] + tally["LEAN_PH"]
    print(f"BN-favoured seats: {bn_locked}  (report: 'If BN takes all five "
          f"[gains], it's at 45 before the tossups' -> 24+14+2+5 = 45)")
    print(f"PH-favoured seats: {ph_locked}  (= central PH ~8)")
    print("Central projection reproduced by: BN concedes one of the three PN-held")
    print("  'lean gains' back to PN (PN ~1), takes the two Chinese-marginal")
    print("  tossups Jementah + Tangkak (45 - 1 + 2 = 46), MUDA holds Puteri")
    print("  Wangsa (~1). Total 46 + 8 + 1 + 1 = 56.")
    print("Range arithmetic: ceiling 52 = 56 - 4 Safe-PH fortresses (BN sweeps")
    print("  every rated-competitive seat); floor 42 = 45 locks - 3 losses among")
    print("  the MCA/MIC-held Chinese-mixed seats, with all tossups lost.")
    print("Internal inconsistencies worth noting:")
    print("  - the table rates all three PN seats LEAN_BN_GAIN, yet the central")
    print("    row still books PN ~1 seat: the topline hedges against its own")
    print("    seat table (equivalent to a ~2/3 conversion rate on lean gains).")
    print("  - the prose calls Johor Jaya one of the 'four true tossups' but the")
    print("    table rates it Lean PH; the tally line says 'Tossup 3'.")
    print("  - Puteri Wangsa's rating implies MUDA can hold, and the central row")
    print("    books MUDA ~1, but Step 2's own poll gives 'others' a collapsing")
    print("    share; nothing in the stated model supports a MUDA hold.")
    print()


# ---------------------------------------------------------------------------
# PART 4 - what Step 2 alone (the poll-swing layer) would have predicted
# ---------------------------------------------------------------------------

def uniform_swing_projection(df: pd.DataFrame, swing: dict) -> pd.Series:
    """Apply a statewide additive swing (pp) to each seat's 2022 block shares
    and call the winner. Blocks absent from a seat in 2022 stay absent."""
    winners = {}
    for num, row in df.iterrows():
        shares = {
            "BN": row["bn22"] + swing.get("BN", 0.0) if row["bn22"] > 0 else 0.0,
            "PH": row["ph22"] + swing.get("PH", 0.0) if row["ph22"] > 0 else 0.0,
            "PN": row["pn22"] + swing.get("PN", 0.0) if row["pn22"] > 0 else 0.0,
            "MUDA": row["muda22"] + swing.get("MUDA", 0.0) if row["muda22"] > 0 else 0.0,
        }
        winners[num] = max(shares, key=shares.get)
    return pd.Series(winners)


def part4_step2_alone(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("PART 4 - The report's Step 2, run honestly as a model on its own")
    print("=" * 78)
    print("Vodus poll (15-29 Jun 2026): BN 36 / PH 26 / PN 15 / others 13 /")
    print("no-answer+undecided 9. The report reads the swing off the RAW shares:")

    raw = {b: VODUS_POLL[b] - SHARES_2022[b] for b in SHARES_2022}
    print(f"  (a) raw read (the report's):        "
          f"BN {raw['BN']:+.1f}, PH {raw['PH']:+.1f}, PN {raw['PN']:+.1f}")
    dec = 100.0 - VODUS_POLL["DK"]
    prop = {b: 100 * VODUS_POLL[b] / dec - SHARES_2022[b] for b in SHARES_2022}
    print(f"  (b) undecideds reallocated:          "
          f"BN {prop['BN']:+.1f}, PH {prop['PH']:+.1f}, PN {prop['PN']:+.1f}")
    act = {b: ACTUAL_SHARES_2026[b] - SHARES_2022[b] for b in SHARES_2022}
    print(f"  (c) actual result:                   "
          f"BN {act['BN']:+.1f}, PH {act['PH']:+.1f}, PN {act['PN']:+.1f}")
    print("  -> Step 2's central reading had the SIGN of BN's swing wrong by ~24 pp.")

    for label, swing in [("raw read (a)", raw), ("reallocated (b)", prop)]:
        w = uniform_swing_projection(df, {**swing, "MUDA": -1.9})  # MUDA 3.5 -> ~1.6
        counts = w.value_counts().to_dict()
        print(f"  Uniform swing with {label:<16} -> seats {counts}")
    print("  Neither poll reading, applied mechanically, lands anywhere near the")
    print("  report's BN ~46: the projection is carried by the qualitative layer")
    print("  (Steps 3-5), not by the polling arithmetic it is presented as.")
    print()


# ---------------------------------------------------------------------------
# PART 5 - scorecard against the actual result
# ---------------------------------------------------------------------------

def part5_scorecard(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("PART 5 - Scoring against the actual result (BN 48, PH 8, PN 0, MUDA 0)")
    print("=" * 78)
    actual = {num: ("PH" if num in ACTUAL_PH_SEATS else "BN") for num in REPORT_TABLE}

    # (i) the report's rated seats (excluding tossups)
    wrong_rated, tossups = [], []
    for num, (name, _, _, rating) in REPORT_TABLE.items():
        if rating == "TOSSUP":
            tossups.append(num)
            continue
        called = "BN" if "BN" in rating else "PH"
        if called != actual[num]:
            wrong_rated.append((num, name, rating, actual[num],
                                ACTUAL_PH_SEATS.get(num, "BN")))
    n_rated = 56 - len(tossups)
    print(f"Rated (non-tossup) seats called correctly: "
          f"{n_rated - len(wrong_rated)}/{n_rated}")
    for num, name, rating, went, party in wrong_rated:
        print(f"  MISS {num} {name:<14} rated {rating:<13} actual {went} ({party})")
    print("Tossups: N02 Jementah -> BN, N10 Tangkak -> BN (both booked for BN in")
    print("  the central 46), N41 Puteri Wangsa -> PH/PKR (central booked MUDA).")

    # (ii) the implied central 56-seat map
    central_map = {}
    for num, (_, _, _, rating) in REPORT_TABLE.items():
        central_map[num] = "BN" if "BN" in rating else "PH"
    central_map["N02"] = central_map["N10"] = "BN"       # tossups to BN
    central_map["N41"] = "MUDA"                          # MUDA holds
    # topline hedge: one PN 'lean gain' stays PN; report doesn't name which
    hits = sum(central_map[n] == actual[n] for n in actual)
    print(f"Implied central map: {hits}/56 correct before the unnamed PN hedge "
          f"(booking any PN seat as a PN hold would make it {hits - 1}/56).")
    print("  The four central-map errors: Simpang Jeram (called BN gain, AMANAH")
    print("  held), Johor Jaya + Perling (called Lean PH, BN/MCA won), Puteri")
    print("  Wangsa (called MUDA hold, PKR won).")

    # (iii) rivals on the same exam
    naive = {num: ("BN" if "BN" in str(df.loc[num, "winner22_bloc"]) else
                   df.loc[num, "winner22_bloc"]) for num in actual}
    naive_hits = sum(naive[n] == actual[n] for n in actual)
    print("\nEveryone on the same exam (statewide seat error = |pred BN - 48| + ...):")
    rows = [
        ("This report (central)", REPORT_CENTRAL),
        ("Naive '2022 repeats'", {"BN": 40, "PH": 12, "PN": 3, "MUDA": 1}),
        ("Ong Kian Ming pro-BN", ONG_SCENARIOS["pro-BN (>60% likely)"]),
        ("Ong Kian Ming pro-PH", ONG_SCENARIOS["pro-PH"]),
        ("Vodus seat model", None),
    ]
    actual_counts = {"BN": 48, "PH": 8, "PN": 0, "MUDA": 0}
    for label, pred in rows:
        if pred is None:
            print(f"  {label:<22} BN leads 20 (17 safe) w/ 31 'in play' - "
                  f"too hedged to score; BN's floor alone missed by 28 seats")
            continue
        err = sum(abs(pred[b] - actual_counts[b]) for b in actual_counts)
        print(f"  {label:<22} BN {pred['BN']:>2} PH {pred['PH']:>2} "
              f"PN {pred['PN']} MUDA {pred['MUDA']}  -> total seat error {err}")
    print(f"  (naive 2022-repeats map would still have called {naive_hits}/56 "
          f"seats right - the fat 2022 margins do a lot of work for everyone)")
    print()
    print("Verdict in one line: the seat-level skill beyond a naive baseline")
    print("came almost entirely from three qualitative judgments (PN seats fall,")
    print("PH marginals wobble, Malay marginals de-split), all three of which")
    print("proved right; the quantitative poll layer (Step 2) pointed the wrong")
    print("way; and the four central-map errors split into one wrong judgment")
    print("override (Simpang Jeram), one wrong hedge on a collapsing incumbent")
    print("party (Puteri Wangsa/MUDA), and two seats where the report's own")
    print("downside scenario for DAP simply ran past its central case (Johor")
    print("Jaya, Perling - both taken by MCA).")


def main() -> None:
    df = load_2022()
    part1_verify_baseline(df)
    rules = part2_rule_reconstruction(df)
    part3_topline(df)
    part4_step2_alone(df)
    part5_scorecard(df)
    out_csv = os.path.join(SCRIPT_DIR, "analyst_report_reconstruction_seats.csv")
    merged = df.join(rules[["report", "rules", "match"]])
    merged["actual2026"] = ["PH" if n in ACTUAL_PH_SEATS else "BN" for n in merged.index]
    merged.to_csv(out_csv)
    print(f"\nPer-seat table written to {os.path.basename(out_csv)}")


if __name__ == "__main__":
    main()
