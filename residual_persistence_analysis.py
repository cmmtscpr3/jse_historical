#!/usr/bin/env python3
"""
Cross-election residual persistence analysis for the Johor DUN first-principles model.

Question being tested (user's claim):
  "When seat-specific residuals are carried forward into future predictions,
   what matters is the CROSS-ELECTION CORRELATION of residuals (rho), not RMSE.
   Therefore RMSE is not the right measure to pick the 'best' model."

Approach:
  1. For each year (2013, 2018, 2022) apply the SAME canonical model spec:
       Stage 1: turnout decomposition (OLS no-intercept) -> within-group turnout T_g
       Stage 2: effective (turnout-weighted) composition
       Stage 3: BN% ~ effective composition (OLS no-intercept)
     under two specs: 2-var (Malay, Chinese) [the deployed engine] and 3-var (+Indian).
  2. Extract per-seat residuals for every year/spec.
  3. Estimate rho = corr(resid_t, resid_{t+1}) for 2013->2018 and 2018->2022,
     overall and by seat type.
  4. Compute optimal shrinkage k* and the resulting forecast error,
     and BACKTEST carry-forward (predict 2022 residual from 2018 residual)
     under k=0 (no carry), k=1 (full carry), k=rho (optimal shrinkage).
  5. Candidate-turnover effect: does residual persist more when the SAME BN
     candidate contests the seat again?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

HIST = "/sessions/trusting-clever-gauss/mnt/JSE historical"

YEARS = {
    2013: ("JOHOR_2013_DUN_RESULTS_HARMONISED.csv",      "JOHOR_2013_DUN_COMPOSITION_HARMONISED.csv"),
    2018: ("JOHOR_2018_DUN_RESULTS_HARMONISED.csv",      "JOHOR_2018_DUN_COMPOSITION_HARMONISED.csv"),
    2022: ("JOHOR_2022_ELECTION_RESULTS_HARMONISED.csv", "JOHOR_2022_DUN_COMPOSITION_HARMONISED.csv"),
}

NAME = "STATE CONSTITUENCY NAME"


def load_year(res_file, comp_file):
    res = pd.read_csv(f"{HIST}/{res_file}", encoding="utf-8-sig")
    comp = pd.read_csv(f"{HIST}/{comp_file}", encoding="utf-8-sig")
    res[NAME] = res[NAME].astype(str).str.strip().str.upper()
    comp[NAME] = comp[NAME].astype(str).str.strip().str.upper()

    # BN vote share of valid votes
    res["bn_pct"] = res["BN VOTE"].astype(float) / res["TOTAL VALID VOTES"].replace(0, 1).astype(float) * 100.0
    res["turnout"] = pd.to_numeric(res["TURNOUT (%)"], errors="coerce")

    # Indian column name varies (INDIAN (%) vs INDIANS (%))
    ind_col = "INDIAN (%)" if "INDIAN (%)" in comp.columns else "INDIANS (%)"
    comp = comp.rename(columns={ind_col: "indian_pct",
                                "MALAY (%)": "malay_pct",
                                "CHINESE (%)": "chinese_pct"})
    for c in ["malay_pct", "chinese_pct", "indian_pct"]:
        comp[c] = pd.to_numeric(comp[c], errors="coerce")

    # urban-rural classification column name varies by year
    ur_col = [c for c in comp.columns if c.startswith("URBAN")]
    ur_col = ur_col[0] if ur_col else None

    keep_res = [NAME, "bn_pct", "turnout", "WINNING PARTY"] if "WINNING PARTY" in res.columns \
               else [NAME, "bn_pct", "turnout", "WINNING PARTY (2022)"]
    # BN candidate for turnover analysis
    if "BN CANDIDATE" in res.columns:
        keep_res.append("BN CANDIDATE")
    keep_comp = [NAME, "malay_pct", "chinese_pct", "indian_pct", "RACIAL COMPOSITION"]
    if ur_col:
        keep_comp.append(ur_col)

    df = res[keep_res].merge(comp[keep_comp], on=NAME, how="inner")
    if ur_col:
        df = df.rename(columns={ur_col: "urban_rural"})
    df = df.dropna(subset=["bn_pct", "turnout", "malay_pct", "chinese_pct", "indian_pct"])
    return df


def ols_no_intercept(y, X_df):
    X = X_df.values
    model = sm.OLS(y.values, X).fit()
    fitted = model.predict(X)
    resid = y.values - fitted
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "params": model.params,
        "r2": float(model.rsquared),
        "rmse": rmse,
        "fitted": fitted,
        "resid": resid,
    }


def fit_year(df, spec="2var"):
    """Run the 3-stage canonical model and return per-seat residuals."""
    # Stage 1: turnout decomposition (registered composition, no intercept)
    X1 = df[["malay_pct", "chinese_pct", "indian_pct"]] / 100.0
    s1 = ols_no_intercept(df["turnout"], X1)
    TM, TC, TI = s1["params"]

    # Stage 2: effective (turnout-weighted) composition
    eff_m = (df["malay_pct"] / 100.0 * TM)
    eff_c = (df["chinese_pct"] / 100.0 * TC)
    eff_i = (df["indian_pct"] / 100.0 * TI)
    denom = eff_m + eff_c + eff_i
    df = df.copy()
    df["eff_m"] = eff_m / denom * 100.0
    df["eff_c"] = eff_c / denom * 100.0
    df["eff_i"] = eff_i / denom * 100.0

    # Stage 3: BN% on effective composition (no intercept)
    if spec == "2var":
        X3 = df[["eff_m", "eff_c"]] / 100.0
    else:
        X3 = df[["eff_m", "eff_c", "eff_i"]] / 100.0
    s3 = ols_no_intercept(df["bn_pct"], X3)
    df["resid"] = s3["resid"]
    df["fitted"] = s3["fitted"]
    return df, {"T": (TM, TC, TI), "stage1_r2": s1["r2"], "coefs": s3["params"],
                "r2": s3["r2"], "rmse": s3["rmse"]}


def pearson(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan, mask.sum()
    return float(np.corrcoef(a[mask], b[mask])[0, 1]), int(mask.sum())


# ------------------------------------------------------------------ run
print("=" * 78)
print("CROSS-ELECTION RESIDUAL PERSISTENCE ANALYSIS")
print("=" * 78)

data = {}
fits = {}
for spec in ["2var", "3var"]:
    data[spec] = {}
    fits[spec] = {}
    for yr, (rf, cf) in YEARS.items():
        df = load_year(rf, cf)
        dff, info = fit_year(df, spec=spec)
        data[spec][yr] = dff
        fits[spec][yr] = info

# In-sample fit table
print("\nIN-SAMPLE FIT BY YEAR AND SPEC")
print(f"{'spec':>6} {'year':>6} {'n':>4} {'stg1R2':>7} {'BM':>7} {'BC':>7} {'BI':>7} {'R2':>6} {'RMSE':>6}")
for spec in ["2var", "3var"]:
    for yr in YEARS:
        info = fits[spec][yr]
        c = info["coefs"]  # already in percentage points (X scaled to 0-1, y in %)
        bm = c[0]
        bc = c[1]
        bi = c[2] if len(c) > 2 else np.nan
        n = len(data[spec][yr])
        print(f"{spec:>6} {yr:>6} {n:>4} {info['stage1_r2']:>7.3f} {bm:>7.2f} {bc:>7.2f} "
              f"{bi:>7.2f} {info['r2']:>6.3f} {info['rmse']:>6.2f}")


def build_resid_panel(spec):
    """Wide panel of residuals across years on common seats."""
    frames = []
    for yr in YEARS:
        d = data[spec][yr][[NAME, "resid", "RACIAL COMPOSITION"]].copy()
        d = d.rename(columns={"resid": f"r{yr}", "RACIAL COMPOSITION": f"race{yr}"})
        frames.append(d.set_index(NAME))
    panel = frames[0].join(frames[1], how="inner").join(frames[2], how="inner")
    return panel


print("\n" + "=" * 78)
print("RESIDUAL PERSISTENCE (rho) — correlation of seat residuals across elections")
print("=" * 78)
for spec in ["2var", "3var"]:
    panel = build_resid_panel(spec)
    n = len(panel)
    rho_1318, _ = pearson(panel["r2013"].values, panel["r2018"].values)
    rho_1822, _ = pearson(panel["r2018"].values, panel["r2022"].values)
    rho_1322, _ = pearson(panel["r2013"].values, panel["r2022"].values)
    # pooled consecutive (stack 13->18 and 18->22)
    a = np.concatenate([panel["r2013"].values, panel["r2018"].values])
    b = np.concatenate([panel["r2018"].values, panel["r2022"].values])
    rho_pool, _ = pearson(a, b)
    print(f"\n[{spec}]  common seats n={n}")
    print(f"   rho(2013,2018) = {rho_1318:+.3f}")
    print(f"   rho(2018,2022) = {rho_1822:+.3f}")
    print(f"   rho(2013,2022) = {rho_1322:+.3f}   (two cycles apart)")
    print(f"   rho pooled consecutive = {rho_pool:+.3f}")
    print(f"   std(resid): 2013={panel['r2013'].std():.2f}  "
          f"2018={panel['r2018'].std():.2f}  2022={panel['r2022'].std():.2f}")

print("\n" + "=" * 78)
print("PERSISTENCE BY SEAT TYPE (using deployed 2-var spec, 2018->2022)")
print("=" * 78)
panel = build_resid_panel("2var")
panel["bucket"] = panel["race2022"].apply(
    lambda s: "Chinese/Mixed" if ("Chinese" in str(s) or "Mixed" in str(s)) else "Malay-majority")
for b, g in panel.groupby("bucket"):
    rho, nn = pearson(g["r2018"].values, g["r2022"].values)
    print(f"   {b:<16} n={nn:>3}  rho(2018,2022) = {rho:+.3f}")

print("\n" + "=" * 78)
print("FORECAST BACKTEST: predict 2022 residual from 2018 residual")
print("=" * 78)
for spec in ["2var", "3var"]:
    panel = build_resid_panel(spec)
    r18 = panel["r2018"].values
    r22 = panel["r2022"].values
    rho, n = pearson(r18, r22)
    s18, s22 = r18.std(), r22.std()
    # optimal shrinkage minimizing E[(r22 - k r18)^2]: k* = rho * s22/s18
    k_star = rho * s22 / s18
    def fr(k):  # forecast RMSE of (r22 - k*r18)
        e = r22 - k * r18
        return float(np.sqrt(np.mean(e ** 2)))
    print(f"\n[{spec}]  n={n}  rho={rho:+.3f}  s18={s18:.2f}  s22={s22:.2f}  k*={k_star:.3f}")
    print(f"   in-sample RMSE (2022 fit, what model-selection 'optimises'): {fits[spec][2022]['rmse']:.2f}")
    print(f"   forecast RMSE  k=0 (no carry-forward) : {fr(0):.2f}")
    print(f"   forecast RMSE  k=1 (full carry-forward): {fr(1):.2f}")
    print(f"   forecast RMSE  k=rho* (optimal shrink) : {fr(k_star):.2f}")
    print(f"   theory  RMSE_resid*sqrt(1-rho^2)       : {s22*np.sqrt(1-rho**2):.2f}")

print("\n" + "=" * 78)
print("CANDIDATE-TURNOVER EFFECT (2-var spec, 2018->2022)")
print("=" * 78)
try:
    c18 = data["2var"][2018][[NAME, "BN CANDIDATE"]].copy()
    c22 = data["2var"][2022][[NAME, "BN CANDIDATE"]].copy()
    c18["BN CANDIDATE"] = c18["BN CANDIDATE"].astype(str).str.strip().str.upper()
    c22["BN CANDIDATE"] = c22["BN CANDIDATE"].astype(str).str.strip().str.upper()
    cand = c18.merge(c22, on=NAME, suffixes=("_18", "_22"))
    cand["same"] = cand["BN CANDIDATE_18"] == cand["BN CANDIDATE_22"]
    panel = build_resid_panel("2var").reset_index()
    m = panel.merge(cand[[NAME, "same"]], on=NAME, how="inner")
    for flag, g in m.groupby("same"):
        rho, nn = pearson(g["r2018"].values, g["r2022"].values)
        lbl = "SAME BN candidate" if flag else "DIFFERENT BN candidate"
        print(f"   {lbl:<24} n={nn:>3}  rho(2018,2022) = {rho:+.3f}")
except Exception as e:
    print("   (candidate-turnover skipped:", e, ")")

print("\n" + "=" * 78)
print("MODEL-SELECTION PARADOX: in-sample RMSE vs forecast RMSE (2018->2022)")
print("=" * 78)
print(f"{'spec':>6} {'in-sample RMSE':>15} {'rho':>7} {'sqrt(1-rho^2)':>14} {'forecast RMSE':>14}")
rows = []
for spec in ["2var", "3var"]:
    panel = build_resid_panel(spec)
    r18, r22 = panel["r2018"].values, panel["r2022"].values
    rho, _ = pearson(r18, r22)
    s22 = r22.std()
    k_star = rho * s22 / r18.std()
    fc = float(np.sqrt(np.mean((r22 - k_star * r18) ** 2)))
    ins = fits[spec][2022]["rmse"]
    rows.append((spec, ins, rho, np.sqrt(1 - rho ** 2), fc))
    print(f"{spec:>6} {ins:>15.2f} {rho:>7.3f} {np.sqrt(1-rho**2):>14.3f} {fc:>14.2f}")
d_ins = rows[0][1] - rows[1][1]
d_fc = rows[0][4] - rows[1][4]
print(f"\n   Going 2var -> 3var: in-sample RMSE improves by {d_ins:.2f} pp,")
print(f"   but FORECAST RMSE improves by only {d_fc:.2f} pp")
print(f"   -> {(1 - d_fc/d_ins)*100:.0f}% of the in-sample 'improvement' is non-persistent noise")
print(f"      (rho FALLS from {rows[0][2]:.3f} to {rows[1][2]:.3f} — the extra variable absorbs")
print(f"       persistent seat signal into a coefficient, so it cannot be carried forward).")

print("\nDONE.")
