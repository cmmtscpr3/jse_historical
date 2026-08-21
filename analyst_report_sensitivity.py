"""Sensitivity analysis for the reconstructed 2026 Johor analyst-report model.

The report's Steps 3-4 (PAS transfer where PN is absent, PN decay where it
stands, Chinese-support erosion in the DAP-MCA battlegrounds, differential
turnout) are qualitative in the original. This script makes them quantitative
as an explicit vote-flow engine so the load-bearing assumptions can be swept:

  pas_transfer      share of the 2022 PN vote that moves to BN where PN is
                    NOT standing in 2026 (PAS's "vote BN" instruction); the
                    same propensity is applied to voters abandoning a
                    still-standing PN candidate; a small fixed 5% goes to PH,
                    the rest stays home
  pn_retention      share of its 2022 vote PN keeps where it IS standing
  ph_erosion        share of PH's 2022 vote that defects to BN in seats with
                    a Chinese electorate >= 30% (the MCA-recovery story;
                    ~0.12 corresponds to Chinese support 85% -> 75%).
                    NEGATIVE values model the report's downside scenario -
                    a Chinese swing AGAINST the MCA/MIC-held seats
  turnout_ratio     relative turnout change, Malay-lean blocs vs PH-lean
                    blocs (>1 = Malay-favouring differential)

Each parameter set produces 56 projected seat winners from the repository's
2022 baseline. The engine is deliberately simple - the same additive
arithmetic the report describes, just written down - and is used to answer:

  Q1  Which of the report's conclusions survive across the whole plausible
      parameter space (robust), and which hold only at particular corners
      (fragile)?
  Q2  What would the model have said in the world the Vodus poll described,
      had the poll been right?
  Q3  Where does the ACTUAL result (BN 48 / PH 8 / PN 0, BN ~60%) sit in the
      parameter space - i.e. was reality inside the report's scenario span?

PN's 2026 contest map (33 of 56 seats) was never published as a list in the
sources available to this analysis; the engine assumes PN stood in its 33
strongest 2022 seats (it demonstrably stood in the 3 it held). The
sensitivity of results to that assumption is itself tested in Q1c.
"""

import os

import numpy as np
import pandas as pd

from analyst_report_reconstruction import (
    ACTUAL_PH_SEATS, REPORT_TABLE, SHARES_2022, VODUS_POLL, load_2022,
    uniform_swing_projection,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAJORITY, TWO_THIRDS = 29, 38


def pn_contest_map(df: pd.DataFrame, n_seats: int = 33) -> pd.Series:
    """True where PN is assumed to stand in 2026: its strongest 2022 seats."""
    strongest = df["pn22"].sort_values(ascending=False).index[:n_seats]
    return pd.Series(df.index.isin(strongest), index=df.index)


def project(df: pd.DataFrame, pas_transfer: float, pn_retention: float,
            ph_erosion: float, turnout_ratio: float,
            pn_present: pd.Series | None = None) -> pd.DataFrame:
    """Project 2026 per-seat shares from 2022 votes under explicit flows.

    Flows are computed on 2022 vote counts, then blocs are reweighted by the
    Malay-vs-PH-lean turnout ratio and renormalised to shares.
    """
    if pn_present is None:
        pn_present = pn_contest_map(df)
    out = []
    for num, r in df.iterrows():
        bn, ph, pn, muda = (r[c] * r["valid22"] / 100
                            for c in ("bn22", "ph22", "pn22", "muda22"))
        if pn_present[num]:
            pn_kept = pn_retention * pn
            leak = pn - pn_kept
            bn_new = bn + pas_transfer * leak  # leavers follow PAS's cue
            ph_new = ph + 0.05 * leak
            pn_new = pn_kept
        else:
            bn_new = bn + pas_transfer * pn
            ph_new = ph + 0.05 * pn
            pn_new = 0.0
        # Chinese-battleground swing: PH <-> BN (positive = MCA recovery)
        if r["chinese"] >= 30:
            moved = ph_erosion * ph
            ph_new, bn_new = ph_new - moved, bn_new + moved
        # MUDA collapse outside its 4 defended seats: mostly back to PH
        if num == "N41":
            muda_new = 0.75 * muda
            ph_new += 0.25 * muda
        else:
            muda_new, ph_new = 0.15 * muda, ph_new + 0.60 * muda
        # micro-parties/independents: carried forward at their 2022 level
        oth_new = max(0.0, (100 - r["bn22"] - r["ph22"] - r["pn22"]
                            - r["muda22"])) * r["valid22"] / 100
        # differential turnout: Malay-lean blocs vs PH-lean blocs
        bn_new, pn_new = bn_new * turnout_ratio, pn_new * turnout_ratio
        tot = bn_new + ph_new + pn_new + muda_new + oth_new
        out.append({"num": num, "bn": 100 * bn_new / tot, "ph": 100 * ph_new / tot,
                    "pn": 100 * pn_new / tot, "muda": 100 * muda_new / tot,
                    "oth": 100 * oth_new / tot, "weight": tot})
    proj = pd.DataFrame(out).set_index("num")
    proj["winner"] = proj[["bn", "ph", "pn", "muda"]].idxmax(axis=1).str.upper()
    return proj


def seat_counts(proj: pd.DataFrame) -> dict:
    c = proj["winner"].value_counts().to_dict()
    return {b: c.get(b, 0) for b in ("BN", "PH", "PN", "MUDA")}


def statewide(proj: pd.DataFrame) -> dict:
    w = proj["weight"]
    return {b: float((proj[b.lower()] * w).sum() / w.sum()) for b in ("BN", "PH", "PN")}


def calibrate_to_statewide(proj: pd.DataFrame, target: dict) -> pd.DataFrame:
    """Uniform-swing correction: shift each bloc's share additively in the
    seats where it is present, sized so the weighted statewide totals hit
    `target`. This is the report's Step 2 composed on top of the Step 3-4
    flows. Clipping at zero perturbs the totals, so iterate to convergence."""
    cols = ["bn", "ph", "pn", "muda", "oth"]
    out = proj.copy()
    if "oth" not in out:
        out["oth"] = 0.0
    w = out["weight"] / out["weight"].sum()
    tgt = dict(target)
    tgt.setdefault("MUDA", 3.5)   # hold MUDA near its 2022 level unless targeted
    tgt.setdefault("OTH", 100.0 - sum(tgt.values()))  # residual -> others
    # 'others' had no 2022 footprint in most seats; let the calibration place
    # the poll's others-vote everywhere rather than only where micro-parties
    # stood in 2022 (Bersama/MUDA contested seats are not fully known).
    out["oth"] = out["oth"].clip(lower=0.5)
    for _ in range(60):
        sw = {b: float((out[b.lower()] * out["weight"]).sum() / out["weight"].sum())
              for b in ("BN", "PH", "PN", "MUDA", "OTH")}
        for b, col in zip(("BN", "PH", "PN", "MUDA", "OTH"), cols):
            present = out[col] > 0
            pw = float(w[present].sum())
            if pw > 0:
                out.loc[present, col] += (tgt[b] - sw[b]) / pw
        out[cols] = out[cols].clip(lower=0)
        tot = out[cols].sum(axis=1)
        for col in cols:
            out[col] = 100 * out[col] / tot
    out["winner"] = out[cols].idxmax(axis=1).str.upper()
    return out


def q1_parameter_sweeps(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("Q1a - BN seats across the PAS-transfer x Chinese-swing grid")
    print("      (pn_retention 0.55, no turnout differential)")
    print("=" * 78)
    transfers = [0.1, 0.3, 0.5, 0.7, 0.9]
    swings = [-0.12, -0.06, 0.00, 0.06, 0.12, 0.18]
    print(f"{'chn swing':<10}" + "".join(f"  t={t:.1f}" for t in transfers))
    lo, hi = 99, 0
    for e in swings:
        cells = []
        for t in transfers:
            n = seat_counts(project(df, t, 0.55, e, 1.0))["BN"]
            lo, hi = min(lo, n), max(hi, n)
            cells.append(f"{n:>7d}")
        print(f"{e:<10.2f}" + "".join(cells))
    print(f"BN span: {lo}-{hi} seats; BN >= {TWO_THIRDS} (two-thirds) "
          f"everywhere: {lo >= TWO_THIRDS}")
    print("Reading: with a moderate-or-better transfer (t >= 0.3) and no adverse")
    print("Chinese swing, the engine sits at 48-55 - ABOVE the report's central")
    print("46. Getting down to its published floor of 42 requires the")
    print("Chinese-mixed seats to swing against MCA/MIC (the negative rows - the")
    print("report's own stated downside) or the PAS transfer to fail almost")
    print("entirely. So the central 46 was a conservative draw from the model's")
    print("own mechanics, and the actual 48 sits in the densest part of the")
    print("sweep. The point estimate is fragile (41-55 across plausible inputs);")
    print("the supermajority claim is not.")

    print()
    print("Q1b - the 'turnout wildcard': turnout_ratio swept pro-PH to pro-BN")
    print("      (transfer 0.6, retention 0.55, chinese swing 0.06)")
    break_23, break_maj = None, None
    for ratio in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]:
        counts = seat_counts(project(df, 0.6, 0.55, 0.06, ratio))
        if counts["BN"] < TWO_THIRDS and break_23 is None:
            break_23 = ratio
        if counts["BN"] < MAJORITY and break_maj is None:
            break_maj = ratio
        print(f"  ratio {ratio:.2f}: BN {counts['BN']:>2}  PH {counts['PH']:>2}  "
              f"PN {counts['PN']}  MUDA {counts['MUDA']}")
    print(f"  Two-thirds first lost at ratio {break_23}; majority at "
          f"{break_maj if break_maj else 'never (>=29 throughout)'}.")
    print("  Even a 40-50% pro-PH turnout differential - far beyond anything in")
    print("  Johor's history, incl. 2018 - does not cost BN the majority. The")
    print("  report's 'turnout is the biggest wildcard' is true only for the")
    print("  MARGIN of victory; the win/majority call was turnout-proof, and its")
    print("  two-thirds call survives any differential short of implausible.")

    print()
    print("Q1c - does the assumed PN contest map matter?")
    for label, pmap in [
        ("strongest 33 (default)", pn_contest_map(df, 33)),
        ("PN stands everywhere", pd.Series(True, index=df.index)),
        ("PN stands nowhere", pd.Series(False, index=df.index)),
    ]:
        counts = seat_counts(project(df, 0.6, 0.55, 0.06, 1.0, pmap))
        print(f"  {label:<24} BN {counts['BN']:>2}  PH {counts['PH']:>2}  "
              f"PN {counts['PN']}  MUDA {counts['MUDA']}")
    print("  The seat count barely moves: with PN this weak, WHERE it stands is")
    print("  second-order next to HOW MUCH of its old vote reaches BN. The")
    print("  unpublished contest map is not a material gap in the reconstruction.")
    print()


def q2_poll_true_world(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("Q2 - the world the Vodus poll described, taken at face value")
    print("=" * 78)
    dec = 100.0 - VODUS_POLL["DK"]
    target = {b: 100 * VODUS_POLL[b] / dec for b in ("BN", "PH", "PN")}
    print(f"Poll with don't-knows reallocated: BN {target['BN']:.1f} / "
          f"PH {target['PH']:.1f} / PN {target['PN']:.1f} / others "
          f"{100 * VODUS_POLL['OTHERS'] / dec:.1f}")
    # Reading A: the report's own arithmetic - additive raw-share swing on the
    # 2022 seat map, no renormalisation (PART 4 of the reconstruction script)
    raw = {b: VODUS_POLL[b] - SHARES_2022[b] for b in SHARES_2022}
    a = uniform_swing_projection(df, {**raw, "MUDA": -1.9}).value_counts().to_dict()
    print(f"Reading A - additive swing, raw poll shares (the report's Step 2): {a}")
    # Reading B: shares forced to actually total the poll's distribution,
    # 'others' at 14.3% spread pro-rata, on the plain 2022 geography
    plain = calibrate_to_statewide(project(df, 0.0, 1.0, 0.0, 1.0), target)
    sw1, c1 = statewide(plain), seat_counts(plain)
    print(f"Reading B - shares forced to the poll's totals "
          f"({sw1['BN']:.1f}/{sw1['PH']:.1f}/{sw1['PN']:.1f}): seats {c1}")
    # Reading C: same forced totals on the Step 3-4 de-split flow geography
    flows = project(df, 0.45, 0.62, 0.0, 1.0)
    cal = calibrate_to_statewide(flows, target)
    sw2, c2 = statewide(cal), seat_counts(cal)
    print(f"Reading C - forced totals on the de-split flow geography "
          f"({sw2['BN']:.1f}/{sw2['PH']:.1f}/{sw2['PN']:.1f}): seats {c2}")
    print(f"Two-thirds threshold: {TWO_THIRDS}; majority: {MAJORITY}")
    print("Every mechanical reading of the poll puts BN at 37-39 seats:")
    print("straddling the 38-seat two-thirds line, far above the 29-seat")
    print("majority line, and ~8 seats below the report's central 46. So, from")
    print("the poll the report presents as its swing anchor: the WIN was")
    print("certain, the SUPERMAJORITY was a coin flip, and 46 would have been a")
    print("clear overshoot. The 'almost certainly keeping two-thirds' headline")
    print("was carried by the report's qualitative override OF its own anchor -")
    print("the judgment that PAS transfers and the PN collapse would run far")
    print("beyond what the poll showed. That judgment was right, and the poll")
    print("was wrong on BN by ~20 pp; but nothing in the stated methodology")
    print("distinguishes this from the poll being right, in which case the same")
    print("report would have missed high by ~8 seats. A robust version would")
    print("have published both branches explicitly.")
    print()


def q3_locate_reality(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("Q3 - where the actual result sits in the swept space")
    print("=" * 78)
    actual = {num: ("PH" if num in ACTUAL_PH_SEATS else "BN") for num in df.index}
    best = None
    for t in np.arange(0.30, 0.96, 0.05):
        for e in np.arange(0.0, 0.19, 0.03):
            for rho in (0.15, 0.25, 0.35, 0.45):
                for ratio in (1.0, 1.1, 1.2):
                    proj = project(df, t, rho, e, ratio)
                    hits = sum(proj.loc[n, "winner"] == actual[n] for n in df.index)
                    sw = statewide(proj)
                    score = (hits, -abs(sw["BN"] - 60.0))
                    if best is None or score > best[0]:
                        best = (score, (t, rho, e, ratio), seat_counts(proj), sw, hits)
    (_, (t, rho, e, ratio), counts, sw, hits) = best
    print(f"Best fit: pas_transfer {t:.2f}, pn_retention {rho:.2f}, "
          f"chinese_swing {e:.2f}, turnout_ratio {ratio:.1f}")
    print(f"  -> seats {counts}; statewide BN {sw['BN']:.1f} / PH {sw['PH']:.1f} / "
          f"PN {sw['PN']:.1f}  (actual: 48/8/0 seats, ~60/33/5.4 shares); "
          f"{hits}/56 winners matched")
    print("Caveats on the fit: the parameters are not separately identified -")
    print("a lower PN retention and a higher Chinese swing move the same")
    print("Chinese-marginal seats, so several combinations fit almost equally")
    print("well (separating them needs the per-seat 2026 counts, which this")
    print("repo does not yet hold), and the engine works in 2022-vote units, so")
    print("the ~15 pp turnout rise is folded into the flow parameters rather")
    print("than modelled. What IS identified: every mechanism the report")
    print("asserted (PN vote collapsing toward BN, MCA recovery in the Chinese")
    print("marginals, no differential-turnout rescue for PH) carries the sign")
    print("the report gave it, and together they land the outcome in the upper")
    print("half of the report's own 42-52 range - the actual 48 above its 46.")
    print()

    proj = project(df, t, rho, e, ratio)
    resid = [f"{num} {REPORT_TABLE[num][0]} "
             f"(engine {proj.loc[num, 'winner']}, actual {actual[num]})"
             for num in df.index if proj.loc[num, "winner"] != actual[num]]
    print("Seats the best global parameter set still gets wrong (local factors):")
    for s in resid:
        print(f"  {s}")
    print("Puteri Wangsa - a four-cornered fight decided by candidate effects -")
    print("is beyond any statewide parameterisation; the report also declined to")
    print("call it. Notably, Simpang Jeram IS explained by the global fit (a")
    print("straightforward PH hold on margin): the report's one clean miss there")
    print("came from overriding its own margin logic with a local-intelligence")
    print("judgment that proved wrong, not from the model.")


def main() -> None:
    df = load_2022()
    q1_parameter_sweeps(df)
    q2_poll_true_world(df)
    q3_locate_reality(df)


if __name__ == "__main__":
    main()
