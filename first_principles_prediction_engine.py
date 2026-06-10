"""
first_principles_prediction_engine.py
======================================
First-principles prediction engine for Johor DUN election outcomes.

CORE MODEL
──────────
For each seat s, predicted BN vote share is:

    BN_next_s = BM_next × Malay_eff_next_s
              + BC_next × Chinese_eff_next_s
              + residual_22_s                          (1)

Where:
  BM_next  = BM_2022 + ΔMalay   (Malay within-group BN support)
  BC_next  = max(0, BC_2022 + ΔChinese)  (Chinese, floored at 0%)
  residual_22_s = 2022 seat fixed effect from joint OLS  (absorbs Indian
                  voters and local candidate/community factors)

EFFECTIVE COMPOSITION
─────────────────────
At expected turnout T_target with Malay-Chinese gap gap_MC:

    TC_s = T_target - (Malay_s / 100) × gap_MC
    TM_s = TC_s + gap_MC
    TI_s = TC_s           (Indian treated same as Chinese)

    Malay_eff_s   = (Malay_reg_s   / 100 × TM_s / 100) / (T_target / 100)
    Chinese_eff_s = (Chinese_reg_s / 100 × TC_s / 100) / (T_target / 100)
    Indian_eff_s  = (Indian_reg_s  / 100 × TI_s / 100) / (T_target / 100)

This is seat-specific: a 60% Malay seat has different TC than a 30% Malay seat
at the same (T_target, gap_MC), because the conservation constraint
T_target = Σ_g (g_reg% × T_g) must hold per seat.

OPPOSITION DISTRIBUTION (3-way scenario)
─────────────────────────────────────────
Non-BN votes are distributed between PH, PN, and Others using the
2022 seat-level PH/PN/other fractions as the BASELINE, then adjusted by:

  - Malay alignment (rural and urban/semi-urban sliders): shifts the fraction
    of non-BN Malay votes going to PH vs PN for each seat type
  - Chinese opposition: assumed fixed at 2022 ratio (majority PH)

COALITION SCENARIOS
───────────────────
  SCENARIO_3WAY:    BN vs PH vs PN independently
  SCENARIO_OPP:     PH + PN pact vs BN
                    joint_opp = PH_adj + PN_adj × transfer_eff
                    BN_gain   = PN_adj × (1 − transfer_eff) × defection_to_bn
  SCENARIO_BN_PN:   BN + PN pact vs PH
                    BN_gain   = PN_adj × transfer_eff
                    PH_gain   = PN_adj × (1 − transfer_eff) × defection_to_ph

UNCERTAINTY
───────────
Model RMSE = 9.2 pp. Seats within ±RMSE of the win/loss threshold are
classified as "Marginal". The grid runner varies parameters across a range
and reports the fraction of grid cells where BN wins (stability score).

BASE ESTIMATES (from 3-variable joint OLS on 2022 effective composition):
  BM_2022 = 60.7%   Malay within-group BN support
  BC_2022 = 0.0%    Chinese within-group BN support (floored; 3-var CI −17.6–15.1%)
  TM_2022 = 66.2%   Malay within-group turnout
  TC_2022 = 45.8%   Chinese within-group turnout
  TI_2022 = 36.6%   Indian within-group turnout

Usage:
    python first_principles_prediction_engine.py

Output:
    first_principles_scenario_dashboard.html
"""

import os, json, itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT     = os.path.join(SCRIPT_DIR, "first_principles_scenario_dashboard.html")

MINOR_COLS = ["PBM VOTE","PEJUANG VOTE","MUDA VOTE","WARSIAN VOTE","PSM VOTE",
              "PUTRA VOTE","INDEPENDENT 1 VOTE","INDEPENDENT 2 VOTE",
              "INDEPENDENT 3 CANDIDATE VOTE"]

SCENARIO_3WAY  = "3way"
SCENARIO_OPP   = "opp_pact"
SCENARIO_BN_PN = "bn_pn"


# ══════════════════════════════════════════════════════════════════════════════
# Input dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TurnoutConfig:
    """
    Controls the turnout environment for the next election.

    T_target:  overall expected turnout rate (%)
               2022 = 55.8%  |  2013/2018 = ~85%
               Default 80% = normal full-cycle Johor state election

    gap_MC:    Malay–Chinese turnout differential (pp)
               Malay turnout exceeds Chinese turnout by this amount.
               0 = 2013/2018 equal-participation pattern
               20 = 2022 snap-election pattern
    """
    T_target: float = 80.0
    gap_MC:   float = 0.0


@dataclass
class AlignmentConfig:
    """
    Malay opposition alignment: fraction of non-BN Malay votes going to PH
    (as opposed to PN).  The complement (1 − ph_malay_X) goes to PN.

    ph_malay_rural:  applies to Rural seats
                     Default 0.26 ≈ 2022 observed rural PH fraction
    ph_malay_urban:  applies to Urban + Semi Urban seats
                     Default 0.49 ≈ 2022 observed urban/semi-urban PH fraction

    Chinese voters: no alignment slider; 2022 PH/PN ratio held fixed per seat.
    This reflects the observation that Chinese opposition preference is
    structurally stable and predominantly PH.
    """
    ph_malay_rural: float = 0.263
    ph_malay_urban: float = 0.489


@dataclass
class CoalitionConfig:
    """
    Coalition scenario and vote-transfer parameters.

    scenario: one of SCENARIO_3WAY / SCENARIO_OPP / SCENARIO_BN_PN

    ph_pn_transfer_efficiency: (OPP scenario)
        Fraction of PN votes that follow the PH-PN pact candidate.
        Default 0.82 (literature-based prior for unified opposition).

    ph_pn_defection_to_bn: (OPP scenario)
        Of PN voters who do NOT follow the pact, fraction voting BN.
        Remainder abstain. Default 0.25.

    bn_pn_transfer_efficiency: (BN_PN scenario)
        Fraction of PN votes that transfer to BN candidate.
        Default 0.78 (post-Sheraton Move resentment discount vs OPP).

    bn_pn_defection_to_ph: (BN_PN scenario)
        Of non-transferring PN voters, fraction voting PH. Default 0.20.
    """
    scenario:                  str   = SCENARIO_3WAY
    ph_pn_transfer_efficiency: float = 0.82
    ph_pn_defection_to_bn:     float = 0.25
    bn_pn_transfer_efficiency: float = 0.78
    bn_pn_defection_to_ph:     float = 0.20


@dataclass
class DemographicSwings:
    """
    Change in BN's within-group support rates (percentage points) vs 2022.

    delta_malay:   change in BM (Malay BN support %).
                   BM_next = 60.7 + delta_malay
                   Poll translation: if poll shows Malay BN support = X%,
                   delta_malay = X − 60.7

    delta_chinese: change in BC (Chinese BN support %).
                   BC_next = max(0, 0 + delta_chinese)
                   Since BC_2022 ≈ 0%, any positive delta implies improvement.
                   Most scenarios: delta_chinese = 0.

    delta_indian:  adjustment to Indian voters' contribution via the
                   seat residual proxy. Positive = more Indian BN support.
                   Treated as a uniform shift to BN from the Indian
                   composition-weighted pool.
    """
    delta_malay:   float = 0.0
    delta_chinese: float = 0.0


@dataclass
class PredictionInputs:
    turnout:    TurnoutConfig      = field(default_factory=TurnoutConfig)
    alignment:  AlignmentConfig    = field(default_factory=AlignmentConfig)
    coalition:  CoalitionConfig    = field(default_factory=CoalitionConfig)
    swings:     DemographicSwings  = field(default_factory=DemographicSwings)
    seat_overrides: dict           = field(default_factory=dict)
    """Format: {"SEAT NAME": {"bn_pct": float, "note": str}}"""


# ══════════════════════════════════════════════════════════════════════════════
# Base data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_base_data() -> pd.DataFrame:
    """
    Returns a seat-indexed DataFrame with all 2022 data needed by the engine:
      - Registered composition (malay%, chinese%, indian%)
      - 2022 vote shares (bn22, ph22, pn22, oth22), turnout22, winner22
      - 2022 effective composition (eff_m22, eff_c22, eff_i22)
      - Joint OLS residual: residual22 (= bn22 − model fit, the seat fixed effect)
      - PH/PN/other fraction of non-BN 2022 votes (baseline opposition split)
      - UR classification and race label
    """
    res  = pd.read_csv(os.path.join(SCRIPT_DIR,"JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv"), encoding="utf-8-sig")
    comp = pd.read_csv(os.path.join(SCRIPT_DIR,"JOHOR_2022_DUN_COMPOSITION_HARMONISED.csv"),  encoding="utf-8-sig")
    for df in (res, comp): df["STATE CONSTITUENCY NAME"] = df["STATE CONSTITUENCY NAME"].str.strip()

    for c in ["BN VOTE","PH VOTE","PN CANDIDATE VOTE","TOTAL VALID VOTES"] + MINOR_COLS:
        if c in res.columns: res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0)
    for c in ["MALAY (%)","CHINESE (%)","INDIANS (%)"]:
        comp[c] = pd.to_numeric(comp[c], errors="coerce").fillna(0)

    res["other"] = sum(res[c] for c in MINOR_COLS if c in res.columns)
    tv = res["TOTAL VALID VOTES"].replace(0,1)
    res["bn22"]  = res["BN VOTE"]           / tv * 100
    res["ph22"]  = res["PH VOTE"]           / tv * 100
    res["pn22"]  = res["PN CANDIDATE VOTE"] / tv * 100
    res["oth22"] = res["other"]             / tv * 100
    tc = next(c for c in res.columns if "TURNOUT" in c.upper())
    res["turnout22"] = pd.to_numeric(res[tc], errors="coerce")
    wc = next(c for c in res.columns if "WINNING PARTY" in c.upper())
    res["winner22"]  = res[wc].str.strip()
    uc = next(c for c in comp.columns if "URBAN" in c.upper())
    comp["ur"] = comp[uc].str.strip().str.upper()

    df = res[["STATE CONSTITUENCY NAME","bn22","ph22","pn22","oth22",
              "turnout22","winner22"]].merge(
         comp[["STATE CONSTITUENCY NAME","MALAY (%)","CHINESE (%)","INDIANS (%)","ur","RACIAL COMPOSITION"]],
         on="STATE CONSTITUENCY NAME")
    df.columns = ["seat","bn22","ph22","pn22","oth22","turnout22","winner22",
                  "malay","chinese","indian","ur","race"]

    # Stage 1: within-group turnout rates
    tm_fit = sm.OLS(df["turnout22"].values, df[["malay","chinese","indian"]].values/100).fit()
    TM, TC, TI = tm_fit.params

    # Stage 2: 2022 effective composition
    T22 = df["turnout22"] / 100
    df["eff_m22"] = (df["malay"]/100   * TM/100) / T22
    df["eff_c22"] = (df["chinese"]/100 * TC/100) / T22
    df["eff_i22"] = (df["indian"]/100  * TI/100) / T22

    # Stage 3: 2-variable joint OLS (Malay_eff + Chinese_eff only).
    # Indian voters are deliberately excluded so the seat residual absorbs their
    # contribution alongside local candidate / community factors.
    # The forward prediction  BN = BM×eff_m + BC×eff_c + residual  is then
    # self-consistent: at 2022 effective compositions with no swings, it recovers
    # bn22 exactly (up to the turnout formula approximation).
    # Trade-off: BC is upward-biased (~12.8%) because omitted Indian BN support is
    # partly attributed to Chinese composition via correlated regressors.
    X2  = np.column_stack([df["eff_m22"], df["eff_c22"]])
    m3  = sm.OLS(df["bn22"].values, X2).fit()
    df["bn_fitted22"] = m3.fittedvalues
    df["residual22"]  = df["bn22"] - df["bn_fitted22"]

    # 2022 opposition PH/PN/other fractions (baseline alignment)
    opp = df["ph22"] + df["pn22"] + df["oth22"]
    df["ph_opp_frac"] = df["ph22"]  / opp.replace(0,1)
    df["pn_opp_frac"] = df["pn22"]  / opp.replace(0,1)
    df["ot_opp_frac"] = df["oth22"] / opp.replace(0,1)

    # UR group: "rural" or "urban" (Urban + Semi Urban)
    df["ur_group"] = df["ur"].map(lambda x: "rural" if x == "RURAL" else "urban")

    # Store model constants as DataFrame attributes
    df.attrs["BM"]            = float(m3.params[0])
    df.attrs["BC"]            = float(m3.params[1])   # ~12.8%; upward-biased by omitted Indian
    df.attrs["TM"]            = float(TM)
    df.attrs["TC"]            = float(TC)
    df.attrs["TI"]            = float(TI)
    df.attrs["RMSE"]          = float(np.sqrt(np.mean(m3.resid**2)))
    df.attrs["AVG_MALAY_PCT"]           = float(df["malay"].mean())
    df.attrs["ph_align_rural_default"]  = float(df[df["ur_group"]=="rural"]["ph_opp_frac"].mean())
    df.attrs["ph_align_urban_default"]  = float(df[df["ur_group"]=="urban"]["ph_opp_frac"].mean())

    return df.set_index("seat")


# ══════════════════════════════════════════════════════════════════════════════
# Prediction engine
# ══════════════════════════════════════════════════════════════════════════════

class FirstPrinciplesEngine:

    BUCKET_THRESHOLDS = [
        (0.85, "BN Safe"),
        (0.60, "BN Likely"),
        (0.40, "Marginal"),
        (0.15, "BN Unlikely"),
        (0.00, "BN Safe Loss"),
    ]

    def __init__(self, base: pd.DataFrame = None):
        self.base = base if base is not None else load_base_data()
        self.BM   = self.base.attrs["BM"]
        self.BC   = self.base.attrs["BC"]
        self.TM22 = self.base.attrs["TM"]
        self.TC22 = self.base.attrs["TC"]
        self.TI22 = self.base.attrs["TI"]
        self.RMSE = self.base.attrs["RMSE"]
        print(f"Engine loaded: BM={self.BM:.2f}%  BC={self.BC:.2f}%  RMSE={self.RMSE:.2f} pp")

    # ── Single prediction ──────────────────────────────────────────────────────

    def predict(self, inputs: PredictionInputs) -> pd.DataFrame:
        rows = [self._predict_seat(seat, self.base.loc[seat], inputs)
                for seat in self.base.index]
        return pd.DataFrame(rows).set_index("seat")

    def _predict_seat(self, seat: str, b: pd.Series, inputs: PredictionInputs) -> dict:
        # ── Seat override bypasses all model logic ────────────────────────────
        if seat in inputs.seat_overrides:
            ov   = inputs.seat_overrides[seat]
            bn   = float(ov.get("bn_pct", b["bn22"]))
            opp  = 100.0 - bn
            ph   = opp * b["ph_opp_frac"]
            pn   = opp * b["pn_opp_frac"]
            ot   = opp * b["ot_opp_frac"]
            note = ov.get("note","Seat override")
            return self._finalize(seat, b, bn, ph, pn, ot, inputs.coalition,
                                  eff_m=b["eff_m22"], eff_c=b["eff_c22"],
                                  note=note, override=True)

        # ── Step 1: Effective composition at expected turnout ─────────────────
        # Derive statewide within-group rates from (T_target, gap_MC), using the
        # average registered Malay% as the conservation anchor so that the
        # statewide average predicted turnout ≈ T_target.
        #
        # Conservation: T_statewide = avg_Malay × TM + (1 − avg_Malay) × TC
        # Combined with TM = TC + gap_MC  →
        #   TC = T_target − avg_Malay × gap_MC
        #   TM = TC + gap_MC
        #
        # Per-seat turnout then emerges from each seat's own composition:
        #   T_s = Malay_s × TM + (1 − Malay_s) × TC
        # This correctly gives higher turnout in Malay-heavy rural seats and
        # lower turnout in Chinese-heavy urban seats.

        T_target = inputs.turnout.T_target
        gap_MC   = inputs.turnout.gap_MC
        avg_m    = self.base.attrs.get("AVG_MALAY_PCT", 59.36)  # statewide avg

        TC = T_target - (avg_m / 100) * gap_MC
        TC = max(5.0,  min(98.0, TC))
        TM = min(100.0, TC + gap_MC)
        TI = TC                      # Indian = same as Chinese

        # Per-seat expected turnout (varies by composition)
        T_s = (b["malay"] / 100 * TM + (100 - b["malay"]) / 100 * TC) / 100
        T_s = max(0.05, T_s)         # guard against rounding to zero

        eff_m = (b["malay"]   / 100 * TM / 100) / T_s
        eff_c = (b["chinese"] / 100 * TC / 100) / T_s
        eff_i = (b["indian"]  / 100 * TI / 100) / T_s

        # ── Step 2: Within-group BN support ───────────────────────────────────
        sw = inputs.swings
        BM_next = self.BM + sw.delta_malay
        BC_next = max(0.0, self.BC + sw.delta_chinese)    # floor at 0; BC ~12.8% baseline

        # ── Step 3: First-principles BN vote share ────────────────────────────
        # Equation (1): BN = BM×Malay_eff + BC×Chinese_eff + residual_22
        # Indian voters are fully absorbed into residual_22.
        bn_model = (BM_next * eff_m) + (BC_next * eff_c)
        bn_pred  = np.clip(bn_model + b["residual22"], 0.0, 100.0)

        # ── Step 4: Opposition distribution ──────────────────────────────────
        opp_total = 100.0 - bn_pred

        # Chinese: hold 2022 PH/PN ratio fixed, only scale by composition change
        # Malay: adjust PH fraction using alignment slider
        ph_align = (inputs.alignment.ph_malay_rural if b["ur_group"] == "rural"
                    else inputs.alignment.ph_malay_urban)

        # Weighted opposition fractions: separately for Malay-driven and Chinese-driven parts
        # Opposition distribution: 2022 seat-level ph_opp_frac is the BASELINE.
        # The Malay alignment slider adjusts ph/pn fractions relative to that baseline.
        # At ph_align == ph_align_default (UR-group default), the distribution exactly
        # reproduces the 2022 seat-level split; the slider represents deviation from it.
        malay_opp_share   = eff_m * (1 - BM_next / 100)
        chinese_opp_share = eff_c * (1 - BC_next / 100)
        total_model_opp   = malay_opp_share + chinese_opp_share

        ph_align_default = (self.base.attrs.get("ph_align_rural_default", 0.263)
                            if b["ur_group"] == "rural"
                            else self.base.attrs.get("ph_align_urban_default", 0.489))

        if total_model_opp > 0:
            malay_share_frac = malay_opp_share / total_model_opp
            delta_ph = (ph_align - ph_align_default) * malay_share_frac
            ph_frac  = max(0.0, min(1.0, b["ph_opp_frac"] + delta_ph))
            pn_frac  = max(0.0, min(1.0, b["pn_opp_frac"] - delta_ph))
            ot_frac  = max(0.0, 1.0 - ph_frac - pn_frac)
        else:
            ph_frac = b["ph_opp_frac"]
            pn_frac = b["pn_opp_frac"]
            ot_frac = b["ot_opp_frac"]

        ph = opp_total * ph_frac
        pn = opp_total * pn_frac
        ot = opp_total * ot_frac

        return self._finalize(seat, b, bn_pred, ph, pn, ot, inputs.coalition,
                              eff_m=eff_m, eff_c=eff_c, BM_next=BM_next, BC_next=BC_next,
                              delta_eff_m=eff_m - b["eff_m22"]/100,
                              delta_eff_c=eff_c - b["eff_c22"]/100)

    def _finalize(self, seat, b, bn, ph, pn, ot, coalition: CoalitionConfig,
                  eff_m=None, eff_c=None, BM_next=None, BC_next=None,
                  delta_eff_m=0.0, delta_eff_c=0.0, note="", override=False):
        sc = coalition.scenario

        if sc == SCENARIO_3WAY:
            leading_opp = max(ph, pn, ot)
            margin = bn - leading_opp
            winner = "BN" if margin > 0 else "Opposition"

        elif sc == SCENARIO_OPP:
            te  = coalition.ph_pn_transfer_efficiency
            dtb = coalition.ph_pn_defection_to_bn
            joint = ph + pn * te
            bn   = np.clip(bn + pn * (1 - te) * dtb, 0, 100)
            leading_opp = max(joint, ot)
            margin = bn - leading_opp
            winner = "BN" if margin > 0 else "Opposition"

        elif sc == SCENARIO_BN_PN:
            te  = coalition.bn_pn_transfer_efficiency
            dtp = coalition.bn_pn_defection_to_ph
            bn  = np.clip(bn + pn * te, 0, 100)
            ph  = ph + pn * (1 - te) * dtp
            leading_opp = max(ph, ot)
            margin = bn - leading_opp
            winner = "BN" if margin > 0 else "Opposition"

        else:
            raise ValueError(f"Unknown scenario: {sc}")

        # Uncertainty: marginal if within RMSE of threshold
        is_marginal = abs(margin) < self.RMSE
        confidence  = "Marginal (within ±{:.0f} pp RMSE)".format(self.RMSE) if is_marginal else (
                      "High confidence" if abs(margin) > 2 * self.RMSE else "Moderate confidence")

        return {
            "seat":          seat,
            "bn22":          round(b["bn22"],  2),
            "bn_pred":       round(bn,         2),
            "opp_pred":      round(leading_opp,2),
            "margin_pp":     round(margin,     2),
            "predicted_winner": winner,
            "is_marginal":   is_marginal,
            "confidence":    confidence,
            "residual22":    round(b["residual22"], 2),
            "eff_m_pct":     round((eff_m or b["eff_m22"]/100)*100, 1),
            "eff_c_pct":     round((eff_c or b["eff_c22"]/100)*100, 1),
            "race":          b["race"],
            "ur":            b["ur"],
            "ur_group":      b["ur_group"],
            "scenario":      sc,
            "note":          note,
            "override":      override,
        }

    # ── Parameter grid ─────────────────────────────────────────────────────────

    def run_grid(self, base_inputs: PredictionInputs,
                 transfer_range: list = None,
                 swing_uncertainty_pp: float = 3.0) -> pd.DataFrame:
        if transfer_range is None:
            transfer_range = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        u = swing_uncertainty_pp
        dm_range = [base_inputs.swings.delta_malay + d for d in [-u, 0, u]]
        dc_range = [base_inputs.swings.delta_chinese + d for d in [-u, 0, u]]

        is_3way = base_inputs.coalition.scenario == SCENARIO_3WAY
        te_range = [base_inputs.coalition.ph_pn_transfer_efficiency] if is_3way else transfer_range

        combos = list(itertools.product(te_range, dm_range, dc_range))
        wins = {seat: 0 for seat in self.base.index}

        for (te, dm, dc) in combos:
            inp = PredictionInputs(
                turnout   = base_inputs.turnout,
                alignment = base_inputs.alignment,
                coalition = CoalitionConfig(
                    scenario=base_inputs.coalition.scenario,
                    ph_pn_transfer_efficiency=te,
                    ph_pn_defection_to_bn=base_inputs.coalition.ph_pn_defection_to_bn,
                    bn_pn_transfer_efficiency=te,
                    bn_pn_defection_to_ph=base_inputs.coalition.bn_pn_defection_to_ph,
                ),
                swings          = DemographicSwings(delta_malay=dm, delta_chinese=dc),
                seat_overrides  = base_inputs.seat_overrides,
            )
            preds = self.predict(inp)
            for seat, row in preds.iterrows():
                if row["predicted_winner"] == "BN":
                    wins[seat] += 1

        total = len(combos)
        flip  = self._flip_thresholds(base_inputs, transfer_range)

        rows = []
        for seat in self.base.index:
            score  = wins[seat] / total
            bucket = self._score_bucket(score)
            rows.append({
                "seat":             seat,
                "stability_score":  round(score, 3),
                "bn_wins":          wins[seat],
                "total_cells":      total,
                "flip_threshold_te": flip.get(seat),
                "bucket":           bucket,
                "race":             self.base.loc[seat]["race"],
                "ur":               self.base.loc[seat]["ur"],
                "residual22":       round(self.base.loc[seat]["residual22"], 2),
                "bn22":             round(self.base.loc[seat]["bn22"], 2),
            })
        return pd.DataFrame(rows).set_index("seat").sort_values("stability_score", ascending=False)

    def _flip_thresholds(self, base_inputs, te_range):
        thresholds = {}
        for seat in self.base.index:
            flip_te = None
            for te in sorted(te_range):
                inp = PredictionInputs(
                    turnout   = base_inputs.turnout,
                    alignment = base_inputs.alignment,
                    coalition = CoalitionConfig(
                        scenario=base_inputs.coalition.scenario,
                        ph_pn_transfer_efficiency=te,
                        ph_pn_defection_to_bn=base_inputs.coalition.ph_pn_defection_to_bn,
                    ),
                    swings=base_inputs.swings,
                    seat_overrides=base_inputs.seat_overrides,
                )
                pred = self._predict_seat(seat, self.base.loc[seat], inp)
                if pred["predicted_winner"] != "BN":
                    flip_te = te
                    break
            thresholds[seat] = flip_te
        return thresholds

    @classmethod
    def _score_bucket(cls, score):
        for thresh, label in cls.BUCKET_THRESHOLDS:
            if score >= thresh:
                return label
        return "BN Safe Loss"

    def summarise(self, predictions: pd.DataFrame) -> dict:
        bn  = predictions["predicted_winner"].eq("BN").sum()
        tot = len(predictions)
        maj = tot // 2 + 1
        by_bucket = {}
        if "bucket" in predictions.columns:
            by_bucket = predictions["bucket"].value_counts().to_dict()
        return {
            "bn_seats":   int(bn),
            "opp_seats":  int(tot - bn),
            "total":      tot,
            "majority":   maj,
            "bn_majority":bool(bn >= maj),
            "by_bucket":  by_bucket,
        }

    def save_results(self, df: pd.DataFrame, path: str):
        df.to_csv(path, encoding="utf-8-sig")
        print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard data extraction + HTML generation
# ══════════════════════════════════════════════════════════════════════════════

def extract_dashboard_data(engine: FirstPrinciplesEngine) -> dict:
    """
    Build the complete dataset to embed in the dashboard HTML.
    All seat-level attributes needed by the JS engine are included.
    """
    base = engine.base
    seats = []
    for seat, r in base.iterrows():
        seats.append({
            "name":        seat,
            "bn22":        round(r["bn22"],   2),
            "ph22":        round(r["ph22"],   2),
            "pn22":        round(r["pn22"],   2),
            "oth22":       round(r["oth22"],  2),
            "turnout22":   round(r["turnout22"],1),
            "malay":       round(r["malay"],   1),
            "chinese":     round(r["chinese"], 1),
            "indian":      round(r["indian"],  1),
            "ur":          r["ur"],
            "ur_group":    r["ur_group"],
            "race":        r["race"],
            "winner22":    r["winner22"],
            "eff_m22":     round(r["eff_m22"]*100, 2),
            "eff_c22":     round(r["eff_c22"]*100, 2),
            "eff_i22":     round(r["eff_i22"]*100, 2),
            "residual22":  round(r["residual22"], 2),
            "bn_fitted22": round(r["bn_fitted22"],2),
            "ph_opp_frac": round(r["ph_opp_frac"],4),
            "pn_opp_frac": round(r["pn_opp_frac"],4),
            "ot_opp_frac": round(r["ot_opp_frac"],4),
        })

    model = {
        "BM":            round(engine.BM,   4),
        "BC":            round(engine.BC,   4),
        "TM22":          round(engine.TM22, 4),
        "TC22":          round(engine.TC22, 4),
        "TI22":          round(engine.TI22, 4),
        "RMSE":          round(engine.RMSE, 3),
        "AVG_MALAY_PCT": round(float(base["malay"].mean()), 4),
        "ph_align_rural_default": round(base[base["ur_group"]=="rural"]["ph_opp_frac"].mean(),4),
        "ph_align_urban_default": round(base[base["ur_group"]=="urban"]["ph_opp_frac"].mean(),4),
    }
    return {"seats": seats, "model": model}


def build_html(data: dict) -> str:
    js_data = json.dumps(data, separators=(",", ":"))
    return HTML_TEMPLATE.replace("/*%%DATA%%*/", f"const ENGINE_DATA={js_data};")


# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Johor DUN — First Principles Prediction Engine</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#f5f5f3;color:#1a1a1a;min-height:100vh;display:flex;flex-direction:column}
.shell{display:flex;flex:1;min-height:0}
.sidebar{width:320px;flex-shrink:0;background:#fff;border-right:.5px solid #ddd;
         overflow-y:auto;padding:1.25rem 1rem 2rem}
.main{flex:1;overflow-y:auto;padding:1.25rem 1.25rem 2rem;min-width:0}

/* sidebar */
.sec-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;
           color:#888;margin:1.2rem 0 .55rem}
.sec-title:first-child{margin-top:0}
.sec-desc{font-size:11px;color:#aaa;margin-bottom:.65rem;line-height:1.45}

/* scenario radios */
.sc-opt{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;
        cursor:pointer;font-size:13px;color:#444;border:.5px solid transparent;
        margin-bottom:4px;transition:background .1s}
.sc-opt:hover{background:#f0f0f0}
.sc-opt.active{background:#EBF3FC;border-color:#B5D4F4;color:#185FA5;font-weight:500}
.sc-opt input{cursor:pointer;margin:0;accent-color:#185FA5}

/* sliders */
.sl-row{margin-bottom:.7rem}
.sl-label{display:flex;justify-content:space-between;align-items:baseline;
          font-size:12px;color:#555;margin-bottom:3px}
.sl-val{font-weight:600;color:#185FA5;font-size:13px;min-width:44px;text-align:right}
input[type=range]{width:100%;cursor:pointer;accent-color:#185FA5}
input[type=range]:disabled{opacity:.35;cursor:not-allowed}
.sl-hint{font-size:10px;color:#aaa;margin-top:2px;line-height:1.4}

/* override UI */
.ov-row{display:flex;gap:5px;margin-bottom:.5rem;align-items:center}
.ov-row select,.ov-row input[type=number]{font-size:12px;border:.5px solid #ccc;
  border-radius:6px;padding:4px 7px;background:#fff;color:#1a1a1a;font-family:inherit}
.ov-row select{flex:1;min-width:0}
.ov-row input[type=number]{width:62px}
.btn-add{font-size:12px;padding:5px 10px;border-radius:6px;cursor:pointer;
         font-family:inherit;border:.5px solid #bbb;background:#f7f7f7;color:#444}
.btn-add:hover{background:#e8f0fb;border-color:#9bbde0;color:#185FA5}
.ov-list{list-style:none;margin-top:.4rem}
.ov-item{display:flex;align-items:center;justify-content:space-between;
         font-size:12px;background:#f5f5f3;border-radius:5px;
         padding:4px 8px;margin-bottom:3px;gap:6px}
.ov-item span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ov-val{font-weight:600;color:#185FA5;white-space:nowrap}
.btn-del{background:none;border:none;cursor:pointer;color:#aaa;font-size:15px;padding:0 2px}
.btn-del:hover{color:#c0392b}
.btn-reset{font-size:12px;padding:5px 12px;border-radius:6px;cursor:pointer;
           font-family:inherit;border:.5px solid #f09595;background:#fff;
           color:#c0392b;margin-top:1rem}
.btn-reset:hover{background:#FCEBEB}

/* summary cards */
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:1rem}
.card{background:#fff;border-radius:10px;padding:.85rem 1rem;border:.5px solid #e0e0e0}
.card-lbl{font-size:11px;color:#888;margin-bottom:4px}
.card-val{font-size:28px;font-weight:500;line-height:1}
.card-sub{font-size:11px;color:#888;margin-top:3px}
.card.good{border-color:#185FA5;background:#EBF3FC}
.card.good .card-val{color:#185FA5}
.card.bad{border-color:#D85A30;background:#FEF0EB}
.card.bad .card-val{color:#D85A30}

/* majority bar */
.maj-wrap{background:#fff;border-radius:10px;padding:.85rem 1rem;margin-bottom:1rem;border:.5px solid #e0e0e0}
.maj-lbl{font-size:11px;color:#888;margin-bottom:.5rem;display:flex;justify-content:space-between}
.maj-outer{position:relative;height:22px;background:#f0f0f0;border-radius:5px;overflow:visible}
.maj-bar{height:100%;border-radius:5px;background:#185FA5;transition:width .4s;min-width:2px}
.maj-line{position:absolute;top:-4px;bottom:-4px;width:2px;background:#1a1a1a;border-radius:1px}
.maj-line-lbl{position:absolute;top:-18px;font-size:10px;color:#666;white-space:nowrap;transform:translateX(-50%)}

/* seat strip */
.strip-wrap{background:#fff;border-radius:10px;padding:.85rem 1rem 1rem;margin-bottom:1rem;border:.5px solid #e0e0e0}
.strip-ttl{font-size:11px;color:#888;margin-bottom:.5rem}
.strip{display:flex;gap:2px;overflow-x:auto;padding-bottom:6px}
.strip-tile{width:18px;height:28px;border-radius:2px;flex-shrink:0;cursor:default;transition:transform .1s}
.strip-tile:hover{transform:scaleY(1.15)}
.maj-marker{width:2px;height:34px;background:#1a1a1a;flex-shrink:0;border-radius:1px;position:relative;align-self:center}
.maj-marker::after{content:"29";position:absolute;top:-18px;left:50%;transform:translateX(-50%);font-size:9px;color:#666}

/* marginal badge */
.badge-marg{display:inline-block;background:#FEF9EC;border:.5px solid #EF9F27;color:#633806;
            font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px;white-space:nowrap}
.badge-conf{display:inline-block;background:#EBF3FC;border:.5px solid #B5D4F4;color:#0C447C;
            font-size:10px;padding:1px 6px;border-radius:4px;margin-left:6px;white-space:nowrap}

/* tabs */
.tabs{display:flex;gap:0;margin-bottom:0;border-bottom:2px solid #eee}
.tab{padding:7px 18px;font-size:13px;cursor:pointer;border:none;background:none;
     font-family:inherit;color:#777;border-bottom:2px solid transparent;
     margin-bottom:-2px;transition:color .12s,border-color .12s}
.tab.active{color:#185FA5;border-bottom-color:#185FA5;font-weight:500}
.tab-panel{display:none;padding-top:1rem}
.tab-panel.active{display:block}

/* chart scroll */
.chart-scroll{height:520px;overflow-y:auto;overflow-x:hidden}

/* table */
.tbl-ctrls{display:flex;gap:8px;margin-bottom:.6rem;flex-wrap:wrap}
.tbl-f{font-size:12px;padding:4px 10px;border:.5px solid #ccc;border-radius:6px;
       background:#fff;cursor:pointer;color:#555;font-family:inherit;transition:background .1s}
.tbl-f.active{background:#185FA5;border-color:#185FA5;color:#fff}
.tbl-wrap{overflow-x:auto;border-radius:8px;border:.5px solid #e0e0e0}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#f5f5f3;font-weight:500;text-align:left;padding:7px 10px;
   border-bottom:1px solid #ddd;cursor:pointer;user-select:none;white-space:nowrap;
   position:sticky;top:0;z-index:1}
th:hover{background:#ebebeb}
th .si{color:#aaa;margin-left:3px;font-size:10px}
td{padding:6px 10px;border-bottom:.5px solid #f0f0f0;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr.bn-safe     td:first-child{border-left:3px solid #0C447C}
tr.bn-likely   td:first-child{border-left:3px solid #378ADD}
tr.marginal    td:first-child{border-left:3px solid #BA7517}
tr.bn-unlikely td:first-child{border-left:3px solid #D85A30}
tr.safe-loss   td:first-child{border-left:3px solid #A32D2D}
tr:hover td{background:#fafafa}
.w-bn{color:#185FA5;font-weight:500}.w-opp{color:#A32D2D;font-weight:500}
.dp{color:#1D9E75}.dn{color:#c0392b}
.marg-tag{font-size:10px;background:#FEF9EC;border:.5px solid #EF9F27;color:#633806;
          padding:1px 5px;border-radius:3px;margin-left:4px}

/* residuals + opp split sections */
.resid-note{font-size:12px;color:#666;margin-bottom:.75rem;line-height:1.55}
.rmse-badge{display:inline-block;background:#f5f5f3;border:.5px solid #ccc;
            border-radius:5px;padding:2px 8px;font-size:12px;font-weight:500}

/* tooltip */
#tipbox{position:fixed;pointer-events:none;background:rgba(255,255,255,.97);
        border:.5px solid #ccc;border-radius:8px;padding:8px 12px;font-size:12px;
        display:none;z-index:9999;max-width:240px;box-shadow:0 2px 8px rgba(0,0,0,.12);line-height:1.6}

footer{font-size:10px;color:#aaa;padding:.75rem 1.25rem;border-top:.5px solid #e0e0e0;background:#fff;flex-shrink:0}
</style>
</head>
<body>
<div id="tipbox"></div>

<div class="shell">
<!-- ═══════════════════ SIDEBAR ═══════════════════ -->
<aside class="sidebar">

<div class="sec-title">Coalition configuration</div>
<label class="sc-opt active" id="lbl-3way"><input type="radio" name="sc" value="3way" checked> 3-way — BN vs PH vs PN</label>
<label class="sc-opt" id="lbl-opp"><input type="radio" name="sc" value="opp"> United opposition — PH + PN</label>
<label class="sc-opt" id="lbl-bnpn"><input type="radio" name="sc" value="bnpn"> BN + PN pact — vs PH</label>

<div class="sec-title" style="margin-top:1.1rem">Transfer parameters
  <span id="te-note" style="font-size:10px;font-weight:400;color:#bbb"> — n/a for 3-way</span>
</div>
<div class="sl-row">
  <div class="sl-label"><span>Transfer efficiency</span><span class="sl-val" id="te-out">82%</span></div>
  <input type="range" id="te" min="60" max="97" step="1" value="82" disabled>
  <p class="sl-hint">Fraction of PN votes following the coalition directive.</p>
</div>
<div class="sl-row">
  <div class="sl-label"><span id="dtb-lbl">Defection to BN</span><span class="sl-val" id="dtb-out">25%</span></div>
  <input type="range" id="dtb" min="5" max="45" step="1" value="25" disabled>
  <p class="sl-hint" id="dtb-hint">Of PN voters not following the pact, fraction voting BN rather than abstaining.</p>
</div>

<div class="sec-title" style="margin-top:1.1rem">Malay opposition alignment</div>
<p class="sec-desc">% of non-BN Malay votes going to PH (remainder to PN). Higher = PH-leaning Malay opposition.</p>
<div class="sl-row">
  <div class="sl-label"><span>Rural seats</span><span class="sl-val" id="alm-r-out">26%</span></div>
  <input type="range" id="alm-r" min="5" max="75" step="1" value="26">
  <p class="sl-hint">At 26% (2022 default) the 2022 opposition split is reproduced exactly. Higher = more Malay opposition votes shift to PH.</p>
</div>
<div class="sl-row">
  <div class="sl-label"><span>Urban / semi-urban seats</span><span class="sl-val" id="alm-u-out">49%</span></div>
  <input type="range" id="alm-u" min="15" max="85" step="1" value="49">
  <p class="sl-hint">At 49% (2022 default) the 2022 opposition split is reproduced exactly. Higher = more Malay opposition votes shift to PH.</p>
</div>

<div class="sec-title" style="margin-top:1.1rem">Turnout scenario</div>
<div class="sl-row">
  <div class="sl-label"><span>Expected overall turnout</span><span class="sl-val" id="turnout-out">56%</span></div>
  <input type="range" id="turnout" min="40" max="92" step="1" value="56">
  <p class="sl-hint">2022 snap election: 56%. 2013/2018: ~85%. Normal state election: ~80%.</p>
</div>
<div class="sl-row">
  <div class="sl-label"><span>Malay–Chinese turnout gap</span><span class="sl-val" id="gap-out">20 pp</span></div>
  <input type="range" id="gap" min="0" max="25" step="1" value="20">
  <p class="sl-hint">0 pp = equal participation (2013/2018 pattern). 20 pp = 2022 snap election pattern.</p>
</div>

<div class="sec-title" style="margin-top:1.1rem">Changes in BN support vs 2022</div>
<p class="sec-desc">ΔMalay = poll Malay BN% − 60.7. ΔChinese adjusts from the ~12.8% baseline (which absorbs Indian voter support via the 2-variable model). Indian voters are fully captured in the 2022 seat residual.</p>
<div class="sl-row">
  <div class="sl-label"><span>ΔMalay</span><span class="sl-val" id="dm-out">0.0 pp</span></div>
  <input type="range" id="dm" min="-12" max="12" step="0.5" value="0">
</div>
<div class="sl-row">
  <div class="sl-label"><span>ΔChinese</span><span class="sl-val" id="dc-out">0.0 pp</span></div>
  <input type="range" id="dc" min="-5" max="15" step="0.5" value="0">
</div>

<div class="sec-title" style="margin-top:1.1rem">Seat-specific overrides</div>
<p class="sec-desc">Override BN% for individual seats (candidate effects, local intelligence). Bypasses model entirely.</p>
<div class="ov-row">
  <select id="ov-seat"><option value="">— select seat —</option></select>
  <input type="number" id="ov-val" min="0" max="100" step="0.5" placeholder="BN%">
  <button class="btn-add" id="btn-add-ov">Add</button>
</div>
<ul class="ov-list" id="ov-list"></ul>
<button class="btn-reset" id="btn-reset">Reset all</button>

</aside>

<!-- ═══════════════════ MAIN ═══════════════════ -->
<main class="main">

<div class="cards" id="cards">
  <div class="card"><div class="card-lbl">BN seats predicted</div><div class="card-val" id="c-bn">—</div><div class="card-sub" id="c-bn-sub"></div></div>
  <div class="card"><div class="card-lbl">Opposition seats</div><div class="card-val" id="c-opp">—</div><div class="card-sub" id="c-opp-sub"></div></div>
  <div class="card" id="c-maj-card"><div class="card-lbl">Simple majority (29)</div><div class="card-val" id="c-maj">—</div><div class="card-sub" id="c-maj-sub"></div></div>
</div>

<div class="maj-wrap">
  <div class="maj-lbl"><span>Seat allocation</span><span>56 seats · majority at 29</span></div>
  <div class="maj-outer" id="maj-outer">
    <div class="maj-bar" id="maj-bar" style="width:0%"></div>
    <div class="maj-line" id="maj-line" style="left:50.9%"><div class="maj-line-lbl">29</div></div>
  </div>
</div>

<div class="strip-wrap">
  <div class="strip-ttl">Seats sorted by predicted BN margin · hover for details · <span class="badge-marg">± amber = marginal (within ±9 pp RMSE)</span></div>
  <div class="strip" id="strip"></div>
</div>

<div class="tabs">
  <button class="tab active" data-tab="predict">Predictions</button>
  <button class="tab" data-tab="resid">2022 residuals</button>
  <button class="tab" data-tab="oppsplit">2022 opposition split</button>
</div>

<!-- Predictions tab -->
<div class="tab-panel active" id="tab-predict">
  <div class="tbl-ctrls">
    <button class="tbl-f active" data-f="all">All</button>
    <button class="tbl-f" data-f="bn">BN wins</button>
    <button class="tbl-f" data-f="opp">Opp wins</button>
    <button class="tbl-f" data-f="marg">Marginal</button>
  </div>
  <div class="tbl-wrap">
    <table><thead><tr>
      <th data-col="name">Seat <span class="si">↕</span></th>
      <th data-col="race">Composition <span class="si">↕</span></th>
      <th data-col="ur">UR <span class="si">↕</span></th>
      <th data-col="bn22">2022 BN% <span class="si">↕</span></th>
      <th data-col="bn_pred">Pred BN% <span class="si">↕</span></th>
      <th data-col="margin_pp">Margin <span class="si">↕</span></th>
      <th data-col="residual22">2022 residual <span class="si">↕</span></th>
      <th data-col="winner">Outcome <span class="si">↕</span></th>
    </tr></thead><tbody id="tbl-body"></tbody></table>
  </div>
</div>

<!-- Residuals tab -->
<div class="tab-panel" id="tab-resid">
  <p class="resid-note">
    Each seat's <strong>2022 residual</strong> = actual BN% − first-principles model fit (BM×Malay_eff + BC×Chinese_eff).
    The residual captures Indian voter contributions, candidate effects, and local community dynamics not explained by racial composition alone.
    Model RMSE = <span class="rmse-badge" id="rmse-badge">—</span>.
    <strong>Seats with large positive residuals</strong> (BN outperformed) may have unusually strong BN candidates or community ties.
    <strong>Large negative residuals</strong> (BN underperformed) may reflect strong opposition candidates, local grievances, or other factors.
    Analysts should scrutinise these seats and consider applying seat overrides where there is specific intelligence.
  </p>
  <div style="height:600px;overflow-y:auto;overflow-x:hidden">
    <div style="position:relative;height:1100px">
      <canvas id="residChart" role="img" aria-label="2022 seat residuals bar chart">2022 residuals for all 56 Johor DUN seats.</canvas>
    </div>
  </div>
  </div>
</div>

<!-- Opposition split tab -->
<div class="tab-panel" id="tab-oppsplit">
  <p class="resid-note">
    In 2022, non-BN votes split between PH and PN very differently depending on seat type.
    This is the <strong>baseline distribution</strong> that the Malay alignment sliders shift from.
    Rural seats are PN-dominated (26% PH); urban seats are more evenly split (57% PH).
    The Chinese voter pool overwhelmingly favours PH — this is held fixed without a slider.
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem">
    <div style="background:#f9f9f7;border-radius:8px;padding:.85rem .9rem .7rem">
      <div style="font-size:12px;font-weight:500;color:#555;margin-bottom:.4rem">PH fraction of non-BN votes by UR class (2022)</div>
      <div style="position:relative;height:180px"><canvas id="urSplitChart" role="img" aria-label="PH fraction by urban-rural">PH fraction of non-BN votes by UR class.</canvas></div>
    </div>
    <div style="background:#f9f9f7;border-radius:8px;padding:.85rem .9rem .7rem">
      <div style="font-size:12px;font-weight:500;color:#555;margin-bottom:.4rem">PH fraction of non-BN votes by composition group (2022)</div>
      <div style="position:relative;height:180px"><canvas id="raceSplitChart" role="img" aria-label="PH fraction by racial composition">PH fraction of non-BN votes by racial composition group.</canvas></div>
    </div>
  </div>
</div>

</main>
</div>

<footer>
  First-principles model: BN_pred = BM×Malay_eff + BC×Chinese_eff + seat_residual_2022.
  BM=60.7%, BC=0% (3-variable joint OLS on effective voter composition, 2022 Johor DUN).
  RMSE ≈ 9.2 pp. Marginal seats = within ±RMSE of win/loss threshold. Indian voters absorbed into seat residual.
  Opposition distribution based on 2022 PH/PN/other fractions, adjusted by Malay alignment sliders.
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
/*%%DATA%%*/

const SEATS   = ENGINE_DATA.seats;
const MODEL   = ENGINE_DATA.model;
const TOTAL   = 56, MAJ = 29;
const RMSE    = MODEL.RMSE;

const BUCKET_CLR = {
  "BN Safe":"#0C447C","BN Likely":"#378ADD","Marginal":"#BA7517",
  "BN Unlikely":"#D85A30","BN Safe Loss":"#A32D2D"
};
const ROW_CLS = {
  "BN Safe":"bn-safe","BN Likely":"bn-likely","Marginal":"marginal",
  "BN Unlikely":"bn-unlikely","BN Safe Loss":"safe-loss"
};
const RACE_CLR = {
  'Chinese-majority (>50%)':'#c0392b','Mixed (no race >50%)':'#d4a017',
  'Malay (50-59%)':'#b8e6b8','Malay (60-69%)':'#7dcc7d',
  'Malay (70-79%)':'#4ab54a','Malay (80-89%)':'#2e8b2e','Malay (90+%)':'#1a5c1a'
};
const AX = '#888';

document.getElementById('rmse-badge').textContent = RMSE.toFixed(1) + ' pp';

let overrides = {}, tableFilter = 'all', sortState = {col:'margin_pp',asc:false};
let allPreds = [];
let chart = null;

/* ── Core engine (JS mirror of Python) ── */
function runEngine() {
  const sc  = document.querySelector('input[name="sc"]:checked').value;
  const te  = parseInt(document.getElementById('te').value)  / 100;
  const dtb = parseInt(document.getElementById('dtb').value) / 100;
  const almR= parseInt(document.getElementById('alm-r').value) / 100;
  const almU= parseInt(document.getElementById('alm-u').value) / 100;
  const T   = parseInt(document.getElementById('turnout').value);
  const gap = parseInt(document.getElementById('gap').value);
  const dm  = parseFloat(document.getElementById('dm').value);
  const dc  = parseFloat(document.getElementById('dc').value);
  const di  = 0;  // Indian absorbed into residual — not a model input

  const BM_next = MODEL.BM + dm;
  const BC_next = Math.max(0, MODEL.BC + dc);

  return SEATS.map(s => {
    /* seat override */
    if (overrides[s.name] !== undefined) {
      const bn_ov = overrides[s.name];
      const opp   = 100 - bn_ov;
      let ph = opp * s.ph_opp_frac, pn = opp * s.pn_opp_frac, ot = opp * s.ot_opp_frac;
      return finalise(s, bn_ov, ph, pn, ot, sc, te, dtb, 0, true);
    }

    /* effective composition — statewide TM/TC derived from T_target and gap,
       anchored on avg Malay% so statewide avg turnout ≈ T_target.
       Per-seat turnout then varies naturally by composition. */
    const avg_m = MODEL.AVG_MALAY_PCT / 100;
    const TC    = Math.max(5, Math.min(98, T - avg_m * gap));
    const TM    = Math.min(100, TC + gap);
    const T_s   = Math.max(0.05, (s.malay/100 * TM + (100-s.malay)/100 * TC) / 100);
    const eff_m = (s.malay/100  * TM/100) / T_s;
    const eff_c = (s.chinese/100 * TC/100) / T_s;
    /* Indian voters absorbed into seat residual — not modelled explicitly. */
    const bn_model = BM_next * eff_m + BC_next * eff_c;
    const bn_pred  = Math.max(0, Math.min(100, bn_model + s.residual22));

    /* opposition distribution */
    const opp_total  = 100 - bn_pred;
    const ph_align   = s.ur_group === 'rural' ? almR : almU;
    const malay_opp  = eff_m * Math.max(0, 1 - BM_next/100);
    const chin_opp   = eff_c * Math.max(0, 1 - BC_next/100);
    const total_opp  = malay_opp + chin_opp;

    let ph, pn, ot;
    if (total_opp > 0) {
      const ph_align_def = s.ur_group==='rural' ? MODEL.ph_align_rural_default : MODEL.ph_align_urban_default;
      const delta_ph     = (ph_align - ph_align_def) * (malay_opp / total_opp);
      const ph_frac      = Math.max(0, Math.min(1, s.ph_opp_frac + delta_ph));
      const pn_frac      = Math.max(0, Math.min(1, s.pn_opp_frac - delta_ph));
      ph = opp_total * ph_frac;
      pn = opp_total * pn_frac;
      ot = opp_total * Math.max(0, 1 - ph_frac - pn_frac);
    } else {
      ph = opp_total * s.ph_opp_frac;
      pn = opp_total * s.pn_opp_frac;
      ot = opp_total * s.ot_opp_frac;
    }
    return finalise(s, bn_pred, ph, pn, ot, sc, te, dtb, eff_m, false);
  });
}

function finalise(s, bn, ph, pn, ot, sc, te, dtb, eff_m, override) {
  let leading_opp, margin, winner;
  if (sc === '3way') {
    leading_opp = Math.max(ph, pn, ot);
  } else if (sc === 'opp') {
    const joint = ph + pn * te;
    bn = Math.min(100, bn + pn*(1-te)*dtb);
    leading_opp = Math.max(joint, ot);
  } else {
    bn = Math.min(100, bn + pn*te);
    ph = ph + pn*(1-te)*dtb;
    leading_opp = Math.max(ph, ot);
  }
  margin = bn - leading_opp;
  winner = margin > 0 ? 'BN' : 'Opposition';
  const is_marg = Math.abs(margin) < RMSE;
  const bucket  = bucketFromScore(winner === 'BN' ? 0.9 + margin/200 : 0.1 - margin/200, margin);
  return {seat:s.name, race:s.race, ur:s.ur, ur_group:s.ur_group,
          bn22:s.bn22, bn_pred:Math.round(bn*10)/10,
          opp_pred:Math.round(leading_opp*10)/10,
          margin_pp:Math.round(margin*10)/10,
          winner, is_marg, bucket,
          residual22:s.residual22, override};
}

function bucketFromScore(_, margin) {
  const m = margin;
  if (m >  2*RMSE) return 'BN Safe';
  if (m >  0)      return m < RMSE ? 'Marginal' : 'BN Likely';
  if (m > -RMSE)   return 'Marginal';
  if (m > -2*RMSE) return 'BN Unlikely';
  return 'BN Safe Loss';
}

/* ── Update all UI ── */
function update() {
  allPreds = runEngine();
  const bn  = allPreds.filter(p=>p.winner==='BN').length;
  const opp = TOTAL - bn;
  document.getElementById('c-bn').textContent  = bn;
  document.getElementById('c-opp').textContent = opp;
  document.getElementById('c-bn-sub').textContent  = 'of '+TOTAL+' seats';
  document.getElementById('c-opp-sub').textContent = 'of '+TOTAL+' seats';
  const majCard = document.getElementById('c-maj-card');
  const majVal  = document.getElementById('c-maj');
  const majSub  = document.getElementById('c-maj-sub');
  majCard.className = 'card '+(bn>=MAJ?'good':'bad');
  majVal.textContent  = bn>=MAJ?'YES':'NO';
  majSub.textContent  = bn>=MAJ?`Majority by ${bn-MAJ+1} seat(s)`:`Short by ${MAJ-bn} seat(s)`;

  document.getElementById('maj-bar').style.width = (bn/TOTAL*100).toFixed(1)+'%';
  document.getElementById('maj-line').style.left = ((MAJ-0.5)/TOTAL*100).toFixed(2)+'%';

  updateStrip(); updateChart(); updateTable();
}

/* ── Strip ── */
const tip = document.getElementById('tipbox');
function showTip(e,p) {
  const margTxt = p.is_marg?'<br><span style="color:#BA7517;font-weight:500">Marginal (±'+RMSE.toFixed(0)+' pp RMSE)</span>':'';
  tip.innerHTML = `<strong>${p.seat}</strong><br>${p.race}<br>${p.ur}<br>
    Margin: <strong>${p.margin_pp>=0?'+':''}${p.margin_pp.toFixed(1)} pp</strong>  ·  ${p.winner==='BN'?'<span style="color:#185FA5">BN</span>':'<span style="color:#A32D2D">Opp</span>'}${margTxt}`;
  tip.style.display='block'; positionTip(e);
}
function hideTip(){tip.style.display='none';}
function positionTip(e){
  const x=e.clientX+14,y=e.clientY-10;
  tip.style.left=(x+250>window.innerWidth?x-265:x)+'px';
  tip.style.top=(y+80>window.innerHeight?y-80:y)+'px';
}
document.addEventListener('mousemove',e=>{if(tip.style.display==='block')positionTip(e);});

function updateStrip() {
  const sorted=[...allPreds].sort((a,b)=>b.margin_pp-a.margin_pp);
  const strip=document.getElementById('strip');
  strip.innerHTML='';
  sorted.forEach((p,idx)=>{
    if(idx===MAJ-1){const mk=document.createElement('div');mk.className='maj-marker';strip.appendChild(mk);}
    const tile=document.createElement('div');
    tile.className='strip-tile';
    tile.style.background = BUCKET_CLR[p.bucket]||'#aaa';
    tile.style.opacity = p.winner==='BN'?'1':'0.7';
    tile.style.outline = p.is_marg?'2px solid #BA7517':'none';
    tile.addEventListener('mouseenter',e=>showTip(e,p));
    tile.addEventListener('mouseleave',hideTip);
    strip.appendChild(tile);
  });
}

/* ── Margin chart ── */
function updateChart() {
  const sorted=[...allPreds].sort((a,b)=>b.margin_pp-a.margin_pp);
  if(chart)chart.destroy();
  chart=new Chart(document.getElementById('marginChart')||{},{
    type:'bar',
    data:{labels:sorted.map(p=>p.seat),
      datasets:[{data:sorted.map(p=>p.margin_pp),
        backgroundColor:sorted.map(p=>p.is_marg?'#BA7517':p.winner==='BN'?'#185FA5':'#c0392b'),
        borderWidth:0,barThickness:14}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      animation:{duration:250},
      plugins:{legend:{display:false},
        tooltip:{callbacks:{title:i=>[i[0].label],
          label:i=>{const p=sorted[i.dataIndex];
            return[`Margin: ${i.raw>=0?'+':''}${i.raw.toFixed(1)} pp`,
                   `BN pred: ${p.bn_pred.toFixed(1)}%   Opp: ${p.opp_pred.toFixed(1)}%`,
                   `2022 residual: ${p.residual22>=0?'+':''}${p.residual22.toFixed(1)} pp`,
                   p.is_marg?'⚠ Marginal (within ±'+RMSE.toFixed(0)+' pp RMSE)':p.bucket];
          }}}},
      scales:{
        x:{title:{display:true,text:'Predicted BN margin (pp)',color:AX,font:{size:11}},
           ticks:{color:AX,callback:v=>(v>=0?'+':'')+v+'pp'},
           grid:{color:ctx=>ctx.tick.value===0?'rgba(0,0,0,.4)':'rgba(0,0,0,.06)',
                 lineWidth:ctx=>ctx.tick.value===0?1.5:1}},
        y:{ticks:{color:'#555',font:{size:10},autoSkip:false,maxRotation:0},grid:{display:false}}
      }}
  });
}

/* ── Table ── */
function updateTable() {
  let data=[...allPreds];
  if(tableFilter==='bn')   data=data.filter(p=>p.winner==='BN');
  if(tableFilter==='opp')  data=data.filter(p=>p.winner==='Opposition');
  if(tableFilter==='marg') data=data.filter(p=>p.is_marg);
  const col=sortState.col;
  data.sort((a,b)=>{
    const va=a[col],vb=b[col];
    return typeof va==='string'?sortState.asc?va.localeCompare(vb):vb.localeCompare(va):sortState.asc?va-vb:vb-va;
  });
  const fmtM=v=>v>=0?`<span class="dp">+${v.toFixed(1)} pp</span>`:`<span class="dn">${v.toFixed(1)} pp</span>`;
  const fmtR=v=>v>=0?`<span class="dp">+${v.toFixed(1)}</span>`:`<span class="dn">${v.toFixed(1)}</span>`;
  document.getElementById('tbl-body').innerHTML=data.map(p=>`<tr class="${ROW_CLS[p.bucket]}">
    <td>${p.seat}${p.is_marg?'<span class="marg-tag">marginal</span>':''}</td>
    <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis">${p.race}</td>
    <td>${p.ur}</td>
    <td>${p.bn22.toFixed(1)}%</td>
    <td><strong>${p.bn_pred.toFixed(1)}%</strong></td>
    <td>${fmtM(p.margin_pp)}</td>
    <td>${fmtR(p.residual22)}</td>
    <td class="${p.winner==='BN'?'w-bn':'w-opp'}">${p.winner}</td>
  </tr>`).join('');
}

/* ── Scenario toggles ── */
document.querySelectorAll('input[name="sc"]').forEach(r=>{
  r.addEventListener('change',()=>{
    document.querySelectorAll('.sc-opt').forEach(l=>l.classList.remove('active'));
    r.closest('.sc-opt').classList.add('active');
    const is3 = r.value==='3way';
    document.getElementById('te').disabled  = is3;
    document.getElementById('dtb').disabled = is3;
    document.getElementById('te-note').style.display = is3?'':'none';
    document.getElementById('dtb-lbl').textContent   = r.value==='bnpn'?'Defection to PH':'Defection to BN';
    document.getElementById('dtb-hint').innerHTML    = r.value==='bnpn'
      ? 'Of non-transferring PN voters, fraction voting PH rather than abstaining.'
      : 'Of PN voters not following the pact, fraction voting BN rather than abstaining.';
    update();
  });
});

/* ── All sliders ── */
[['te','%',1],['dtb','%',1],['alm-r','%',1],['alm-u','%',1],
 ['turnout','%',1],['gap',' pp',1],['dm',' pp',.1],['dc',' pp',.1]
].forEach(([id,sfx,step])=>{
  const el=document.getElementById(id);
  const out=document.getElementById(id+'-out');
  el.addEventListener('input',()=>{
    const v=parseFloat(el.value);
    out.textContent=(step<1?(v>=0?'+':'')+v.toFixed(1):v)+sfx;
    update();
  });
});

/* ── Tabs ── */
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
    if(t.dataset.tab==='resid') renderResidChart();
    if(t.dataset.tab==='oppsplit') renderOppSplitCharts();
  });
});

/* ── Table filters + sort ── */
document.querySelectorAll('.tbl-f').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.tbl-f').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); tableFilter=b.dataset.f; updateTable();
  });
});
document.querySelectorAll('th[data-col]').forEach(th=>{
  th.addEventListener('click',()=>{
    const c=th.dataset.col;
    if(sortState.col===c)sortState.asc=!sortState.asc;
    else{sortState.col=c;sortState.asc=['name','race','ur'].includes(c);}
    document.querySelectorAll('th .si').forEach(i=>i.textContent='↕');
    th.querySelector('.si').textContent=sortState.asc?'↑':'↓';
    updateTable();
  });
});

/* ── Overrides ── */
const ovSelect=document.getElementById('ov-seat');
SEATS.forEach(s=>{const o=document.createElement('option');o.value=s.name;o.textContent=s.name;ovSelect.appendChild(o);});
document.getElementById('btn-add-ov').addEventListener('click',()=>{
  const seat=ovSelect.value, val=parseFloat(document.getElementById('ov-val').value);
  if(!seat||isNaN(val)||val<0||val>100)return;
  overrides[seat]=val; renderOverrides(); ovSelect.value=''; document.getElementById('ov-val').value=''; update();
});
function renderOverrides(){
  const list=document.getElementById('ov-list');
  list.innerHTML=Object.entries(overrides).map(([s,v])=>`
    <li class="ov-item"><span title="${s}">${s}</span><span class="ov-val">BN ${v.toFixed(1)}%</span>
    <button class="btn-del" data-seat="${s}">×</button></li>`).join('');
  list.querySelectorAll('.btn-del').forEach(b=>b.addEventListener('click',()=>{delete overrides[b.dataset.seat];renderOverrides();update();}));
}

/* ── Reset ── */
document.getElementById('btn-reset').addEventListener('click',()=>{
  document.querySelector('input[value="3way"]').checked=true;
  document.querySelector('input[value="3way"]').dispatchEvent(new Event('change'));
  [['te','82'],['dtb','25'],['alm-r','26'],['alm-u','49'],['turnout','56'],['gap','20'],
   ['dm','0'],['dc','0']].forEach(([id,v])=>{document.getElementById(id).value=v;document.getElementById(id).dispatchEvent(new Event('input'));});
  overrides={};renderOverrides();update();
});

/* ── Residuals chart (lazy) ── */
let residChart=null;
function renderResidChart(){
  if(residChart)return;
  const sorted=[...SEATS].sort((a,b)=>Math.abs(b.residual22)-Math.abs(a.residual22));
  const colors=sorted.map(s=>s.residual22>=0?'rgba(24,95,165,.75)':'rgba(192,57,43,.75)');
  residChart=new Chart(document.getElementById('residChart'),{
    type:'bar',
    data:{labels:sorted.map(s=>s.name),datasets:[{data:sorted.map(s=>s.residual22),
      backgroundColor:colors,borderWidth:0,barThickness:12}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{title:i=>[i[0].label],
        label:i=>{const s=sorted[i.dataIndex];
          return[`Residual: ${i.raw>=0?'+':''}${i.raw.toFixed(1)} pp`,
                 `Actual: ${s.bn22.toFixed(1)}%   Fitted: ${s.bn_fitted22.toFixed(1)}%`,`${s.race}  ·  ${s.ur}`];}}}},
      scales:{x:{ticks:{color:AX,callback:v=>(v>=0?'+':'')+v+'pp'},
        grid:{color:ctx=>ctx.tick.value===0?'rgba(0,0,0,.4)':'rgba(0,0,0,.06)',lineWidth:ctx=>ctx.tick.value===0?1.5:1}},
        y:{ticks:{color:'#555',font:{size:10},autoSkip:false,maxRotation:0},grid:{display:false}}}}
  });
}

/* ── Opposition split charts (lazy) ── */
let oppChartsDone=false;
function renderOppSplitCharts(){
  if(oppChartsDone)return; oppChartsDone=true;
  const urMap={'URBAN':0,'SEMI URBAN':0,'RURAL':0}, urN={};
  SEATS.forEach(s=>{urMap[s.ur]=(urMap[s.ur]||0)+s.ph_opp_frac; urN[s.ur]=(urN[s.ur]||0)+1;});
  const urLabels=['URBAN','SEMI URBAN','RURAL'];
  const urVals=urLabels.map(k=>urN[k]?Math.round(urMap[k]/urN[k]*100):0);
  const urColors=['#378ADD','#7dcc7d','#4ab54a'];
  new Chart(document.getElementById('urSplitChart'),{type:'bar',
    data:{labels:urLabels,datasets:[{data:urVals,backgroundColor:urColors,borderWidth:0,barThickness:36}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:i=>`PH fraction: ${i.raw}%`}}},
      scales:{x:{ticks:{color:AX},grid:{display:false}},
        y:{min:0,max:100,ticks:{color:AX,callback:v=>v+'%'},grid:{color:'rgba(0,0,0,.06)'},
           title:{display:true,text:'PH % of non-BN votes',color:AX,font:{size:10}}}}}});
  const catOrder=['Chinese-majority (>50%)','Mixed (no race >50%)','Malay (50-59%)','Malay (60-69%)','Malay (70-79%)','Malay (80-89%)','Malay (90+%)'];
  const raceMap={},raceN={};
  SEATS.forEach(s=>{raceMap[s.race]=(raceMap[s.race]||0)+s.ph_opp_frac; raceN[s.race]=(raceN[s.race]||0)+1;});
  const raceVals=catOrder.map(c=>raceN[c]?Math.round(raceMap[c]/raceN[c]*100):0);
  const raceCols=catOrder.map(c=>RACE_CLR[c]||'#888');
  new Chart(document.getElementById('raceSplitChart'),{type:'bar',
    data:{labels:catOrder.map(c=>c.replace('(>50%)','').replace('(no race >50%)','').trim()),
      datasets:[{data:raceVals,backgroundColor:raceCols,borderWidth:0,barThickness:16}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:i=>`PH fraction: ${i.raw}%`}}},
      scales:{x:{ticks:{color:AX,font:{size:10},maxRotation:35},grid:{display:false}},
        y:{min:0,max:100,ticks:{color:AX,callback:v=>v+'%'},grid:{color:'rgba(0,0,0,.06)'},
           title:{display:true,text:'PH % of non-BN votes',color:AX,font:{size:10}}}}}});
}

/* ── Init ── */
document.getElementById('te-note').style.display='';
document.getElementById('maj-line').style.left=((MAJ-0.5)/TOTAL*100).toFixed(2)+'%';
/* Add margin chart canvas lazily to predictions tab */
const chartDiv=document.createElement('div');
chartDiv.className='chart-scroll';
chartDiv.innerHTML='<div style="position:relative;height:1200px"><canvas id="marginChart" role="img" aria-label="Predicted BN margin chart for all seats">Predicted BN margin per seat.</canvas></div>';
document.getElementById('tab-predict').appendChild(chartDiv);
update();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading 2022 base data and fitting joint OLS...")
    base   = load_base_data()
    engine = FirstPrinciplesEngine(base)

    print("\nExtracting dashboard data...")
    dash_data = extract_dashboard_data(engine)

    print("Generating HTML...")
    html = build_html(dash_data)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {OUTPUT}")

    # Demo prediction
    print("\n── Demo: status quo at 80% equal turnout ──")
    inputs = PredictionInputs()
    preds  = engine.predict(inputs)
    s = engine.summarise(preds)
    print(f"BN: {s['bn_seats']} seats | Opp: {s['opp_seats']} | Majority: {'YES' if s['bn_majority'] else 'NO'}")
    print(f"Marginal seats: {preds['is_marginal'].sum()}")

    print("\n── Demo: united opposition at 82% TE ──")
    inputs2 = PredictionInputs(coalition=CoalitionConfig(scenario=SCENARIO_OPP))
    preds2  = engine.predict(inputs2)
    s2 = engine.summarise(preds2)
    print(f"BN: {s2['bn_seats']} seats | Opp: {s2['opp_seats']} | Majority: {'YES' if s2['bn_majority'] else 'NO'}")


if __name__ == "__main__":
    main()
