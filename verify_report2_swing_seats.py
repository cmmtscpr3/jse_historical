"""Fact-check of the analyst's second report ("swing seats deep-dive, Tindak
data") against this repository's harmonised data.

The second report covers 15 seats in three groups (BN's 5 likeliest gains, 4
tossups, 6 exposed BN holds) and makes four kinds of checkable claims:

  A. 2022 results: winner/runner-up candidate names, component parties,
     vote shares, majorities, and third-party shares  -> checked against
     DATA/JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv
  B. Demographics (Table A, sourced to Tindak's 2026 dataset)
     -> compared with the repo's official 2022 roll composition; small
        drift is expected from roll growth, large gaps are flagged
  C. 2026 electorate counts -> compared with the 2022 roll (equality would
     mean the analyst mislabelled 2022 data as 2026; moderate growth is
     what a genuine 2026 roll should show)
  D. Urbanisation labels (the analyst's own classification, disclosed as
     such) -> compared with the official 2021 urban/semi-urban/rural
     classification carried in the repo's composition file

2026 candidate-lineup claims are verified separately via contemporary
reporting (see ANALYST_REPORT2_FACTCHECK.md).
"""

import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "DATA")

# ---------------------------------------------------------------------------
# Report 2's claims, transcribed
# ---------------------------------------------------------------------------
# 2022 claims: winner (name, party-label, share%), majority votes,
# runner-up (name-or-None, party-label, share%), extras: list of
# (party-label, share%) the report cites for third parties.
CLAIMS_2022 = {
    "N07": dict(w=("SAHRUDDIN JAMAL", "PN-PPBM", 44.3), maj=710,
                ru=("ISMAIL MOHAMED", "BN-UMNO", 41.1), extras=[("MUDA", 13.8)],
                note="implies no PH candidate in 2022"),
    "N13": dict(w=("SALAHUDDIN AYUB", "PH-PAN", 40.9), maj=2399,
                ru=(None, "PN-PAS", 29.7), extras=[("BN-UMNO", 28.4)]),
    "N15": dict(w=("ABDUL AZIZ TALIB", "PN-PAS", 36.0), maj=1037,
                ru=(None, "PH-PAN", 31.0), extras=[("BN-UMNO", 27.9)]),
    "N32": dict(w=("ALWIYAH TALIB", "PN-PPBM", 55.5), maj=3041,
                ru=("MOHD YOUZAIMI YUSOF", "BN-UMNO", 35.5), extras=[]),
    "N51": dict(w=("CHIONG SEN SERN", "PH-PKR", 39.2), maj=137,
                ru=("SUPAYYAH SOLAIMUTHU", "BN-MIC", 38.6),
                extras=[("PN-GERAKAN", 16.6)]),
    "N02": dict(w=("NG KOR SIM", "PH-DAP", 40.9), maj=714,
                ru=("SEE ANN GIAP", "BN-MCA", 37.6), extras=[("PN-PAS", 21.5)]),
    "N10": dict(w=("EE CHIN LI", "PH-DAP", 40.9), maj=372,
                ru=("ONG CHEE SIANG", "BN-MCA", 39.0), extras=[("PN-PPBM", 15.6)]),
    "N42": dict(w=("LIOW CAI TUNG", "PH-DAP", 41.7), maj=1922,
                ru=("CHAN SAN SAN", "BN-MCA", 37.7), extras=[("PN-GERAKAN", 17.5)]),
    "N41": dict(w=("AMIRA AISYA", "MUDA", 43.2), maj=7114,
                ru=("NG YEW AIK", "BN-MCA", 29.8), extras=[("PN-GERAKAN", 16.9)]),
    "N04": dict(w=("SARASWATHY NALLATHANBY", "BN-MIC", 41.1), maj=1611,
                ru=(None, "PH-PAN", 32.3), extras=[("PN-PAS", 25.4)]),
    "N06": dict(w=("TAN CHONG", "BN-MCA", 51.3), maj=3569,
                ru=(None, "PH-DAP", 25.3), extras=[("PN-PPBM", 21.0)]),
    "N19": dict(w=("LING TIAN SOON", "BN-MCA", 52.8), maj=2741,
                ru=(None, "PH-DAP", 38.2), extras=[("PN-GERAKAN", 9.0)]),
    "N30": dict(w=("LEE TING HAN", "BN-MCA", 55.1), maj=3176,
                ru=(None, "PH-DAP", 33.4), extras=[("PN-PAS", 10.3)]),
    "N33": dict(w=("RAVEN KUMAR", "BN-MIC", 49.1), maj=1356,
                ru=(None, "PN-PAS", 42.8), extras=[("PH-PKR", 7.1)]),
    "N55": dict(w=("TAN ENG MENG", "BN-MCA", 51.5), maj=4835,
                ru=("YEO TUNG SIONG", "PH-DAP", 28.9), extras=[("PN-GERAKAN", 12.8)]),
}

# Table A: malay/chinese/indian/other %, urbanisation label, "2026 electors"
CLAIMS_DEMO = {
    "N07": (70, 25, 3, 2, "Rural", 37683),
    "N13": (52, 45, 2, 1, "Semi-urban", 41975),
    "N15": (62, 35, 2, 1, "Semi-urban", 40040),
    "N32": (80, 14, 1, 6, "Rural", 28767),
    "N51": (37, 52, 9, 2, "Semi-urban", 49963),
    "N02": (42, 49, 8, 1, "Semi-urban", 41137),
    "N10": (41, 47, 10, 2, "Semi-urban", 36955),
    "N42": (35, 45, 8, 12, "Urban", 97685),
    "N41": (36, 51, 10, 3, "Urban", 128723),
    "N04": (55, 38, 6, 2, "Rural", 35365),
    "N06": (30, 47, 19, 4, "Rural", 27317),
    "N19": (35, 57, 7, 1, "Semi-urban", 34023),
    "N30": (47, 35, 15, 3, "Rural", 25419),
    "N33": (83, 13, 1, 3, "Rural", 39001),
    "N55": (56, 41, 1, 2, "Semi-urban", 37556),
}
# Footnote claims on the "other" column
CLAIMS_FOOTNOTES = {
    "N32": ("ORANG ASLI (%)", 5.0, "~5% Orang Asli"),
    "N06": ("ORANG ASLI (%)", 3.5, "~3.5% Orang Asli"),
    "N42": ("EM_BUMI", 8.5, "~8.5% East Malaysian Bumiputera (4.6 Sabah + 3.9 Sarawak)"),
}

SEAT_NAMES = {
    "N07": "BUKIT KEPONG", "N13": "SIMPANG JERAM", "N15": "MAHARANI",
    "N32": "ENDAU", "N51": "BUKIT BATU", "N02": "JEMENTAH", "N10": "TANGKAK",
    "N42": "JOHOR JAYA", "N41": "PUTERI WANGSA", "N04": "KEMELAH",
    "N06": "BEKOK", "N19": "YONG PENG", "N30": "PALOH", "N33": "TENGGAROH",
    "N55": "PEKAN NANAS",
}


def seat_key(name):
    return re.sub(r"[^A-Z]", "", str(name).upper())


def name_match(claimed, actual):
    """Loose match: a meaningful token shared in either direction (the data
    uses roll-style spellings and short forms like 'SARAS' for Saraswathy)."""
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        return False
    stop = {"MOHD", "MOHAMED", "MUHAMMAD", "ABDUL", "BIN", "BINTI", "HAJI", "DR"}
    ct = [t for t in re.split(r"[^A-Z]+", claimed.upper()) if len(t) >= 3 and t not in stop]
    at = [t for t in re.split(r"[^A-Z]+", str(actual).upper()) if len(t) >= 3 and t not in stop]
    return any(c == a or (len(c) >= 5 and c in "".join(at)) or (len(a) >= 5 and a in "".join(ct))
               for c in ct for a in at)


def main():
    res = pd.read_csv(os.path.join(DATA_DIR, "JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv"))
    comp = pd.read_csv(os.path.join(DATA_DIR, "JOHOR_2022_DUN_COMPOSITION_HARMONISED.csv"))
    res["key"] = res["STATE CONSTITUENCY NAME"].map(seat_key)
    comp["key"] = comp["STATE CONSTITUENCY NAME"].map(seat_key)
    res = res.set_index("key")
    comp = comp.set_index("key")

    flags = []

    # ---------------- A. 2022 results ----------------
    print("=" * 78)
    print("A. 2022 result claims vs DATA/JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv")
    print("=" * 78)
    BLOCK = {
        "BN": ("BN", "BN CANDIDATE", "BN VOTE"),
        "PH": ("PH", "PH CANDIDATE", "PH VOTE"),
        "PN": ("PN", "PN CANDIDATE", "PN CANDIDATE VOTE"),
        "MUDA": ("MUDA", "MUDA CANDIDATE", "MUDA VOTE"),
    }
    share_errs = []
    for num, cl in CLAIMS_2022.items():
        row = res.loc[seat_key(SEAT_NAMES[num])]
        valid = float(row["TOTAL VALID VOTES"])
        maj = float(row["WINNING MAJORITY"])
        problems = []

        def share_of(label):
            bloc = label.split("-")[0] if "-" in label else label
            party_col, cand_col, vote_col = BLOCK[bloc]
            v = row[vote_col]
            party = str(row[party_col]) if pd.notna(row[party_col]) else None
            return (100 * float(v) / valid if pd.notna(v) else None), party, \
                   row[cand_col] if pd.notna(row[cand_col]) else None

        # winner
        w_name, w_label, w_share = cl["w"]
        s, party, cand = share_of(w_label)
        if s is None:
            problems.append(f"winner bloc {w_label} absent in data")
        else:
            share_errs.append(abs(s - w_share))
            if abs(s - w_share) > 0.25:
                problems.append(f"winner share {w_share} vs data {s:.1f}")
            want_party = w_label.split("-")[-1]
            if want_party not in ("MUDA",) and want_party not in str(party):
                problems.append(f"winner party {w_label} vs data {party}")
            if not name_match(w_name, cand):
                problems.append(f"winner name '{w_name}' vs data '{cand}'")
        if abs(maj - cl["maj"]) > 0.5:
            problems.append(f"majority {cl['maj']} vs data {maj:.0f}")

        # runner-up
        ru_name, ru_label, ru_share = cl["ru"]
        s, party, cand = share_of(ru_label)
        if s is None:
            problems.append(f"runner-up bloc {ru_label} absent in data")
        else:
            share_errs.append(abs(s - ru_share))
            if abs(s - ru_share) > 0.25:
                problems.append(f"runner-up share {ru_share} vs data {s:.1f}")
            want_party = ru_label.split("-")[-1]
            if want_party not in str(party):
                problems.append(f"runner-up party {ru_label} vs data {party}")
            if ru_name and not name_match(ru_name, cand):
                problems.append(f"runner-up name '{ru_name}' vs data '{cand}'")

        # third parties
        for label, share in cl["extras"]:
            s, party, cand = share_of(label)
            if s is None:
                problems.append(f"cited bloc {label} absent in data")
                continue
            share_errs.append(abs(s - share))
            if abs(s - share) > 0.25:
                problems.append(f"{label} share {share} vs data {s:.1f}")
            want_party = label.split("-")[-1]
            if want_party != "MUDA" and want_party not in str(party):
                problems.append(f"cited party {label} vs data {party}")

        if cl.get("note") == "implies no PH candidate in 2022":
            if pd.notna(row["PH VOTE"]):
                problems.append("report implies no PH candidate, but data has one")

        status = "OK " if not problems else "FLAG"
        print(f"  [{status}] {num} {SEAT_NAMES[num].title():<14}"
              + ("" if not problems else " | " + "; ".join(problems)))
        flags += [(num, p) for p in problems]
    print(f"  Max share error across all {len(share_errs)} checked shares: "
          f"{max(share_errs):.2f} pp; mean {np.mean(share_errs):.2f} pp")

    # ---------------- B. demographics ----------------
    print()
    print("=" * 78)
    print("B. Table A demographics (claimed Tindak 2026) vs repo 2022 roll")
    print("=" * 78)
    print(f"  {'seat':<16}{'Malay':>14}{'Chinese':>14}{'Indian':>14}{'Other':>14}")
    for num, (m, c, i, o, urb, elec) in CLAIMS_DEMO.items():
        row = comp.loc[seat_key(SEAT_NAMES[num])]
        other_2022 = (row["ORANG ASLI (%)"] + row["BUMIPUTERA SABAH (%)"]
                      + row["BUMIPUTERA SARAWAK (%)"] + row["OTHERS (%)"])
        cells, worst = [], 0.0
        for claim, data in [(m, row["MALAY (%)"]), (c, row["CHINESE (%)"]),
                            (i, row["INDIANS (%)"]), (o, other_2022)]:
            d = claim - data
            worst = max(worst, abs(d))
            cells.append(f"{claim:>5.0f} ({d:+.1f})")
        tag = "  <-- LARGE GAP" if worst > 4 else ""
        print(f"  {num} {SEAT_NAMES[num].title():<12}" + "".join(f"{c:>14}" for c in cells) + tag)
        if worst > 4:
            flags.append((num, f"demographics differ from 2022 roll by up to {worst:.1f} pp"))
    print("  (x (+d) = claimed value, and its gap vs the official 2022 roll;")
    print("   drift of 1-3 pp is normal roll growth, larger gaps are flagged)")

    for num, (col, val, text) in CLAIMS_FOOTNOTES.items():
        row = comp.loc[seat_key(SEAT_NAMES[num])]
        actual = (row["BUMIPUTERA SABAH (%)"] + row["BUMIPUTERA SARAWAK (%)"]
                  if col == "EM_BUMI" else row[col])
        print(f"  footnote {num}: claims {text}; 2022 roll: {actual:.1f}%")
        if abs(actual - val) > 2:
            flags.append((num, f"footnote '{text}' vs 2022 roll {actual:.1f}%"))

    # ---------------- C. electorate ----------------
    print()
    print("=" * 78)
    print("C. Claimed 2026 electorate vs 2022 roll")
    print("=" * 78)
    exact = 0
    for num, (m, c, i, o, urb, elec) in CLAIMS_DEMO.items():
        e22 = float(res.loc[seat_key(SEAT_NAMES[num]), "TOTAL ELECTORATE"])
        growth = 100 * (elec - e22) / e22
        marker = ""
        if elec == int(e22):
            exact += 1
            marker = "  <-- identical to 2022 (mislabelled?)"
        elif growth < -5 or growth > 35:
            marker = "  <-- implausible change"
            flags.append((num, f"claimed 2026 electorate {elec:,} vs 2022 {e22:,.0f} ({growth:+.1f}%)"))
        elif growth < 0:
            marker = "  (small decline - plausible for an ageing seat)"
        print(f"  {num} {SEAT_NAMES[num].title():<14} 2022: {e22:>9,.0f}   "
              f"claimed 2026: {elec:>9,}   ({growth:+5.1f}%){marker}")
    if exact:
        print(f"  {exact} seat(s) identical to the 2022 roll")

    # ---------------- D. urbanisation ----------------
    print()
    print("=" * 78)
    print("D. Analyst's urbanisation labels vs official 2021 classification")
    print("=" * 78)
    diffs = 0
    for num, (m, c, i, o, urb, elec) in CLAIMS_DEMO.items():
        official = str(comp.loc[seat_key(SEAT_NAMES[num]),
                                "URBAN-RURAL CLASSIFICATION (2021)"]).title()
        mine = urb.replace("Semi-urban", "Semi Urban").split(" (")[0].title()
        same = mine == official
        diffs += (not same)
        print(f"  {num} {SEAT_NAMES[num].title():<14} analyst: {urb:<16} "
              f"official 2021: {official:<12} {'' if same else '<-- differs'}")
    print(f"  {diffs}/15 labels differ from the official classification the")
    print("  analyst said they could not access (it sits in this repo's")
    print("  composition file). Disclosed as own-classification, so a")
    print("  methodology note rather than an error - but checkable, and the")
    print("  'elector count as urbanisation signal' heuristic is contradicted")
    print("  by its own table (e.g. rural Tenggaroh has more electors than")
    print("  semi-urban Yong Peng).")

    print()
    print("=" * 78)
    print(f"TOTAL FLAGS: {len(flags)}")
    for num, p in flags:
        print(f"  {num}: {p}")


if __name__ == "__main__":
    main()
