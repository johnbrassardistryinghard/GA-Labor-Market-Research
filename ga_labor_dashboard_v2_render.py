###############################################################################
#                                                                             #
#   GEORGIA LABOR MARKET RESEARCH DASHBOARD — Plotly Dash v2.0               #
#                                                                             #
#   Author:   John Brassard                                                   #
#   Created:  2026-04-18  |  Revised: 2026-04-26                              #
#   Purpose:  Interactive executive dashboard presenting all analyses from    #
#             the Georgia Labor Market Research Pipeline.                     #
#                                                                             #
#   pip install:                                                              #
#     pip install dash dash-bootstrap-components plotly pandas numpy          #
#     pip install fredapi statsmodels scipy openpyxl                          #
#                                                                             #
###############################################################################

# %% [0] Imports
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_VERSION = "Dashboard-v2.0"


# %% [0B] Constants, Palette & API Keys
# =============================================================================
# SECTION 0 — CONSTANTS & CONFIGURATION
# =============================================================================

FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError(
        "FRED_API_KEY environment variable not set. "
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
        "then set: export FRED_API_KEY=your_key_here  (Windows: set FRED_API_KEY=your_key_here)"
    )
fred = Fred(api_key=FRED_KEY)

# Resolve BLS file relative to this script so it works on any machine
BLS_FILE = Path(__file__).parent / "data" / "ststdsadata.xlsx"

START_DATE = "2000-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

STATES = ["GA", "AL", "FL", "TN", "NC", "SC"]
STATE_NAMES = {
    "GA": "Georgia", "AL": "Alabama", "FL": "Florida",
    "TN": "Tennessee", "NC": "North Carolina", "SC": "South Carolina",
}
STATE_FULL_NAMES = list(STATE_NAMES.values())

# --- Brand palette (executive theme) ---
# Primary: navy (institutional, neutral, signals research credibility)
# Accent:  orange — reserved for Georgia data only, so it pops on every chart
# Support: muted greys for peer states, US always black-dashed
BRAND = {
    "primary":   "#1F3A5F",   # deep navy — navbar, headers
    "primary_d": "#162A45",   # darker navy — gradients
    "accent":    "#D55E00",   # Georgia orange (Okabe-Ito)
    "good":      "#009E73",   # teal-green
    "bad":       "#C44536",   # muted brick red (less alarming than pure red)
    "warn":      "#E69F00",   # amber
    "neutral":   "#5A6C7D",   # slate
    "bg_soft":   "#F7F8FA",   # page background tint
    "card_bg":   "#FFFFFF",
    "border":    "#E5E8EC",
    "ink":       "#1A1A1A",
    "ink_soft":  "#5A6C7D",
}

PALETTE = {
    "Georgia": BRAND["accent"],
    "Alabama": "#0072B2", "Florida": "#009E73",
    "Tennessee": "#E69F00", "North Carolina": "#56B4E9",
    "South Carolina": "#CC79A7", "US": "#000000",
}

INDUSTRY_PALETTE = {
    "Manufacturing": "#1b9e77", "Trade/Transport/Util": "#d95f02",
    "Prof/Business Svcs": "#7570b3", "Education/Health": "#e7298a",
    "Leisure/Hospitality": "#66a61e", "Government": "#e6ab02",
    "Construction": "#a6761d", "Financial Activities": "#666666",
}

CYCLE_COLORS = {
    "Pre-GFC": "#0072B2", "GFC": "#D55E00", "Recovery": "#009E73",
    "COVID": "#E69F00", "Post-COVID": "#CC79A7",
}

CYCLE_ORDER = ["Pre-GFC", "GFC", "Recovery", "COVID", "Post-COVID"]


# %% [0C] Helper Functions
def assign_cycle(date):
    d = pd.Timestamp(date)
    if d <= pd.Timestamp("2007-09-30"):
        return "Pre-GFC"
    elif d <= pd.Timestamp("2009-06-30"):
        return "GFC"
    elif d <= pd.Timestamp("2019-12-31"):
        return "Recovery"
    elif d <= pd.Timestamp("2021-06-30"):
        return "COVID"
    else:
        return "Post-COVID"


def last_val(df):
    sub = df.dropna(subset=["value"]).sort_values("date")
    if len(sub) == 0:
        return np.nan
    return sub.iloc[-1]["value"]


def common_layout():
    """
    Plotly layout defaults — tuned for executive presentation.
    NOTE: Axis styling (grid, line colors) is applied separately via
    apply_axis_style() so that per-chart yaxis/xaxis customizations
    (e.g., range, tickvals) don't collide with **kwargs unpacking.
    """
    return dict(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                  size=12, color=BRAND["ink"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0)",
                    font=dict(size=11)),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=80, b=70),
        paper_bgcolor="white",
        plot_bgcolor="#FBFCFD",
    )


def apply_axis_style(fig):
    """
    Apply executive axis styling (grid, line colors) to a figure.
    Uses update_xaxes/update_yaxes which MERGE with existing settings,
    so this is safe to call after charts that customize their axes.
    """
    fig.update_xaxes(gridcolor="#EEF1F4", linecolor="#D5DAE0", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F4", linecolor="#D5DAE0", zeroline=False)
    return fig


def add_recessions(fig, rec_blocks):
    for _, row in rec_blocks.iterrows():
        fig.add_vrect(
            x0=row["xmin"], x1=row["xmax"],
            fillcolor="lightgrey", opacity=0.25, layer="below", line_width=0,
        )
    return fig


def make_title(headline, takeaway=None):
    """
    Two-line title: bold headline (the claim) + lighter subtitle (the takeaway).
    Executive charts should make a *point*, not just describe data.
    """
    if takeaway:
        return dict(
            text=(f"<b style='font-size:15px;color:{BRAND['ink']}'>{headline}</b>"
                  f"<br><span style='font-size:11.5px;color:{BRAND['ink_soft']};"
                  f"font-weight:400'>{takeaway}</span>"),
            x=0.0, xanchor="left", y=0.97, yanchor="top",
        )
    return dict(
        text=f"<b style='font-size:15px;color:{BRAND['ink']}'>{headline}</b>",
        x=0.0, xanchor="left", y=0.97, yanchor="top",
    )


# %% [1] Data Pipeline — BLS LAUS
# =============================================================================
# SECTION 1 — DATA PIPELINE
# =============================================================================
print("Loading data pipeline...")
print("  [1/9] Reading BLS LAUS data...")

df_bls_raw = pd.read_excel(
    BLS_FILE, sheet_name=0, skiprows=7, header=None,
    names=["fips", "state_area", "year", "month",
           "pop", "lf_total", "lfpr", "emp_total",
           "ep_ratio", "unemp_total", "ur"],
    dtype={"fips": str, "state_area": str, "year": str, "month": str},
)

df_bls_laus = df_bls_raw.dropna(subset=["state_area", "year", "month", "ur"]).copy()
df_bls_laus["year"] = df_bls_laus["year"].astype(float).astype(int)
df_bls_laus["month"] = df_bls_laus["month"].astype(float).astype(int)
df_bls_laus["date"] = pd.to_datetime(df_bls_laus[["year", "month"]].assign(day=1))

for col in ["pop", "lf_total", "lfpr", "emp_total", "ep_ratio", "unemp_total", "ur"]:
    df_bls_laus[col] = pd.to_numeric(df_bls_laus[col], errors="coerce")

abbrev_map = {v: k for k, v in STATE_NAMES.items()}
df_bls_se = df_bls_laus[df_bls_laus["state_area"].isin(STATE_FULL_NAMES)].copy()
df_bls_se["state_abbrev"] = df_bls_se["state_area"].map(abbrev_map)

df_bls_all_states = df_bls_laus[
    ~df_bls_laus["state_area"].isin(["Los Angeles County", "New York city"])
].copy()


# %% [1B] Data Pipeline — FRED State Fundamentals
print("  [2/9] Pulling FRED data...")


def pull_fred(series_id, label, start=START_DATE, end=END_DATE):
    time.sleep(0.12)
    try:
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = s.reset_index()
        df.columns = ["date", "value"]
        df["series_id"] = series_id
        df["label"] = label
        return df
    except Exception as e:
        print(f"  FAILED: {series_id} -- {e}")
        return pd.DataFrame({"date": [pd.NaT], "value": [np.nan],
                             "series_id": [series_id], "label": [label]})


def pull_fred_flex(series_id, label, start=START_DATE, end=END_DATE):
    return pull_fred(series_id, label, start, end)


ur_series = [
    ("GAUR", "Georgia"), ("ALUR", "Alabama"), ("FLUR", "Florida"),
    ("TNUR", "Tennessee"), ("NCUR", "North Carolina"),
    ("SCUR", "South Carolina"), ("UNRATE", "US"),
]
df_fred_ur = pd.concat([pull_fred(s, l) for s, l in ur_series], ignore_index=True)
df_fred_ur["metric"] = "ur"

lfpr_series = [
    ("LBSSA13", "Georgia"), ("LBSSA01", "Alabama"), ("LBSSA12", "Florida"),
    ("LBSSA47", "Tennessee"), ("LBSSA37", "North Carolina"),
    ("LBSSA45", "South Carolina"), ("CIVPART", "US"),
]
df_fred_lfpr = pd.concat([pull_fred(s, l) for s, l in lfpr_series], ignore_index=True)
df_fred_lfpr["metric"] = "lfpr"

payroll_series = [
    ("GANA", "Georgia"), ("ALNA", "Alabama"), ("FLNA", "Florida"),
    ("TNNA", "Tennessee"), ("NCNA", "North Carolina"),
    ("SCNA", "South Carolina"), ("PAYEMS", "US"),
]
df_fred_payrolls = pd.concat([pull_fred(s, l) for s, l in payroll_series], ignore_index=True)
df_fred_payrolls["metric"] = "payrolls"


# %% [1C] Data Pipeline — CPS Demographics
print("  [3/9] Pulling CPS demographics...")
cps_lfpr_series = [
    ("CIVPART", "All 16+"),
    ("LNS11300060", "Prime-Age 25-54"),
    ("LRAC25MAUSM156S", "Prime-Age Men 25-54"),
    ("LRAC25FEUSM156S", "Prime-Age Women 25-54"),
    ("LNS11300001", "Men 16+"),
    ("LNS11300002", "Women 16+"),
    ("LNS11300036", "Age 16-24"),
    ("LNS11324230", "Age 55+"),
]
df_cps_lfpr = pd.concat([pull_fred(s, l) for s, l in cps_lfpr_series], ignore_index=True)
df_cps_lfpr["metric"] = "lfpr_cps"

cps_ur_series = [
    ("UNRATE", "All"),
    ("LNS14000060", "Prime-Age 25-54"),
    ("LNS14000001", "Men"),
    ("LNS14000002", "Women"),
    ("LNS14000006", "Black"),
    ("LNS14000009", "Hispanic"),
    ("LNS14000003", "White"),
    ("LNS14027659", "Less than HS"),
    ("LNS14027660", "HS Grad"),
    ("LNS14027689", "Some College"),
    ("LNS14027662", "Bachelors+"),
]
df_cps_ur = pd.concat([pull_fred(s, l) for s, l in cps_ur_series], ignore_index=True)
df_cps_ur["metric"] = "ur_cps"


# %% [1D] Data Pipeline — GA Economic Development
print("  [4/9] Pulling GA economic development...")
df_ga_gdp = pull_fred_flex("GARQGSP", "GA Real GDP")
df_ga_pcpi = pull_fred_flex("GAPCPI", "GA Per Capita Personal Income")
df_us_pcpi = pull_fred_flex("A792RC0A052NBEA", "US Per Capita Personal Income")


# %% [1E] Data Pipeline — JOLTS
print("  [5/9] Pulling JOLTS...")
jolts_series = [
    ("JTSJOR", "Job Openings Rate"), ("JTSHIR", "Hire Rate"),
    ("JTSQUR", "Quit Rate"), ("JTSLDR", "Layoffs Rate"),
]
df_jolts = pd.concat([pull_fred(s, l) for s, l in jolts_series], ignore_index=True)
df_jolts["metric"] = "jolts"


# %% [1F] Data Pipeline — Wages & CPI
print("  [6/9] Pulling wages & CPI...")
df_ahe = pull_fred("CES0500000003", "Avg Hourly Earnings (Private)")
df_eci = pull_fred_flex("ECIALLCIV", "Employment Cost Index")
df_cpi = pull_fred("CPIAUCSL", "CPI-U (All Urban)")


# %% [1G] Data Pipeline — GA Industry Payrolls
print("  [7/9] Pulling GA industry payrolls...")
ga_industry_series = [
    ("GAMFG", "Manufacturing"), ("GATRAD", "Trade/Transport/Util"),
    ("GAPBSV", "Prof/Business Svcs"), ("GAEDUH", "Education/Health"),
    ("GALEIH", "Leisure/Hospitality"), ("GAGOVT", "Government"),
    ("GACONS", "Construction"), ("GAFIRE", "Financial Activities"),
]
df_ga_industry = pd.concat(
    [pull_fred(s, l) for s, l in ga_industry_series], ignore_index=True
)
df_ga_industry["metric"] = "ga_industry"


# %% [1H] Data Pipeline — NBER Recession Bands
print("  [8/9] Building recession bands...")
df_recession = pull_fred("USREC", "NBER Recession").dropna(subset=["date"])
df_recession_sorted = df_recession.sort_values("date").reset_index(drop=True)
df_recession_sorted["value"] = df_recession_sorted["value"].astype(int)
df_recession_sorted["block"] = (
    df_recession_sorted["value"].diff().fillna(0).ne(0).cumsum()
)
rec_blocks = (
    df_recession_sorted[df_recession_sorted["value"] == 1]
    .groupby("block")
    .agg(xmin=("date", "min"), xmax=("date", "max"))
    .reset_index(drop=True)
)


# %% [2] Derived Variables
# =============================================================================
# SECTION 3 — DERIVED VARIABLES
# =============================================================================
print("  [9/9] Building derived variables...")

# 3A: Merge BLS + FRED state panel
df_state_ur_bls = df_bls_se[["date", "state_area", "ur"]].copy()
df_state_ur_bls.rename(columns={"state_area": "label", "ur": "value"}, inplace=True)
df_state_ur_bls["metric"] = "ur"
df_state_ur_bls["source"] = "bls_xlsx"

df_state_lfpr_bls = df_bls_se[["date", "state_area", "lfpr"]].copy()
df_state_lfpr_bls.rename(columns={"state_area": "label", "lfpr": "value"}, inplace=True)
df_state_lfpr_bls["metric"] = "lfpr"
df_state_lfpr_bls["source"] = "bls_xlsx"

df_us_ur_fred = df_fred_ur[df_fred_ur["label"] == "US"][["date", "label", "value"]].copy()
df_us_ur_fred["metric"] = "ur"
df_us_ur_fred["source"] = "fred"

df_us_lfpr_fred = df_fred_lfpr[df_fred_lfpr["label"] == "US"][["date", "label", "value"]].copy()
df_us_lfpr_fred["metric"] = "lfpr"
df_us_lfpr_fred["source"] = "fred"

df_panel_ur = pd.concat([df_state_ur_bls, df_us_ur_fred], ignore_index=True)
df_panel_ur = df_panel_ur[df_panel_ur["date"] >= START_DATE].copy()

df_panel_lfpr = pd.concat([df_state_lfpr_bls, df_us_lfpr_fred], ignore_index=True)
df_panel_lfpr = df_panel_lfpr[df_panel_lfpr["date"] >= START_DATE].copy()

# 3B: Payroll growth rates
df_payroll_growth = (
    df_fred_payrolls.dropna(subset=["date", "value"])
    .sort_values(["label", "date"]).copy()
)
df_payroll_growth["yoy_pct"] = df_payroll_growth.groupby("label")["value"].pct_change(12) * 100

# 3C: Indexed payrolls (Feb 2020 = 100)
idx_base_2020 = (
    df_fred_payrolls.dropna(subset=["value"])
    .loc[df_fred_payrolls["date"] == "2020-02-01", ["label", "value"]]
    .rename(columns={"value": "base_2020"})
)
df_payroll_indexed = (
    df_fred_payrolls.dropna(subset=["date", "value"])
    .merge(idx_base_2020, on="label", how="left")
)
df_payroll_indexed["idx_2020"] = np.where(
    df_payroll_indexed["base_2020"].notna(),
    df_payroll_indexed["value"] / df_payroll_indexed["base_2020"] * 100,
    np.nan,
)

# 3D: GA vs US spread
df_ga_ur = (
    df_panel_ur[df_panel_ur["label"] == "Georgia"][["date", "value"]]
    .rename(columns={"value": "ga_ur"})
)
df_us_ur_only = (
    df_panel_ur[df_panel_ur["label"] == "US"][["date", "value"]]
    .rename(columns={"value": "us_ur"})
)
df_ga_lfpr_only = (
    df_panel_lfpr[df_panel_lfpr["label"] == "Georgia"][["date", "value"]]
    .rename(columns={"value": "ga_lfpr"})
)
df_us_lfpr_only = (
    df_panel_lfpr[df_panel_lfpr["label"] == "US"][["date", "value"]]
    .rename(columns={"value": "us_lfpr"})
)

df_spread = (
    df_ga_ur
    .merge(df_us_ur_only, on="date", how="inner")
    .merge(df_ga_lfpr_only, on="date", how="inner")
    .merge(df_us_lfpr_only, on="date", how="inner")
)
df_spread["ur_spread"] = df_spread["ga_ur"] - df_spread["us_ur"]
df_spread["lfpr_spread"] = df_spread["ga_lfpr"] - df_spread["us_lfpr"]
df_spread["cycle"] = df_spread["date"].apply(assign_cycle)

# 3E: Labor market tightness
df_jor = (
    df_jolts[df_jolts["label"] == "Job Openings Rate"][["date", "value"]]
    .rename(columns={"value": "jor"})
)
df_unrate = (
    df_fred_ur[df_fred_ur["label"] == "US"][["date", "value"]]
    .rename(columns={"value": "unrate"})
)
df_tightness = df_jor.merge(df_unrate, on="date", how="inner")
df_tightness["tightness"] = df_tightness["jor"] / df_tightness["unrate"]

# 3F: Real AHE
df_ahe_clean = df_ahe[["date", "value"]].rename(columns={"value": "ahe"}).dropna()
df_cpi_clean = df_cpi[["date", "value"]].rename(columns={"value": "cpi"}).dropna()
df_real_ahe = df_ahe_clean.merge(df_cpi_clean, on="date", how="inner").sort_values("date")
latest_cpi = df_real_ahe["cpi"].iloc[-1]
df_real_ahe["real_ahe"] = df_real_ahe["ahe"] / df_real_ahe["cpi"] * latest_cpi
df_real_ahe["ahe_yoy"] = df_real_ahe["ahe"].pct_change(12) * 100
df_real_ahe["real_yoy"] = df_real_ahe["real_ahe"].pct_change(12) * 100
df_real_ahe["cpi_yoy"] = df_real_ahe["cpi"].pct_change(12) * 100

# 3G: ECI YoY growth
df_eci_growth = df_eci.dropna(subset=["value"]).sort_values("date").copy()
df_eci_growth["eci_yoy"] = df_eci_growth["value"].pct_change(4) * 100

# 3I: GA industry shares & growth
df_ga_ind_growth = (
    df_ga_industry.dropna(subset=["date", "value"])
    .sort_values(["label", "date"]).copy()
)
df_ga_ind_growth["yoy_chg"] = df_ga_ind_growth.groupby("label")["value"].diff(12)
df_ga_ind_growth["yoy_pct"] = df_ga_ind_growth.groupby("label")["value"].pct_change(12) * 100

latest_ind_date = df_ga_ind_growth.dropna(subset=["value"])["date"].max()
df_ga_ind_shares = df_ga_ind_growth[df_ga_ind_growth["date"] == latest_ind_date].copy()
total_ind = df_ga_ind_shares["value"].sum()
df_ga_ind_shares["share"] = df_ga_ind_shares["value"] / total_ind * 100
df_ga_ind_shares = df_ga_ind_shares.sort_values("share", ascending=False)

# 3J: GA GDP growth
df_ga_gdp_growth = df_ga_gdp.dropna(subset=["value"]).sort_values("date").copy()
df_ga_gdp_growth["yoy_pct"] = df_ga_gdp_growth["value"].pct_change(1) * 100


# %% [3] Summary Tables
# =============================================================================
# SECTION 4 — SUMMARY TABLES
# =============================================================================

latest_ur = (
    df_panel_ur.dropna(subset=["value"]).sort_values("date")
    .groupby("label").tail(1)[["label", "value"]]
    .rename(columns={"value": "ur"})
)
latest_lfpr = (
    df_panel_lfpr.dropna(subset=["value"]).sort_values("date")
    .groupby("label").tail(1)[["label", "value"]]
    .rename(columns={"value": "lfpr"})
)
latest_payroll = (
    df_payroll_growth.dropna(subset=["yoy_pct"]).sort_values("date")
    .groupby("label").tail(1)[["label", "yoy_pct", "value"]]
    .rename(columns={"yoy_pct": "payroll_yoy_pct", "value": "payroll_level"})
)

table1 = (
    latest_ur
    .merge(latest_lfpr, on="label", how="left")
    .merge(latest_payroll, on="label", how="left")
)

table4 = (
    df_ga_ind_shares[["label", "value", "share", "yoy_chg", "yoy_pct"]]
    .sort_values("value", ascending=False).reset_index(drop=True)
)

rank_base = latest_ur[latest_ur["label"] != "US"].copy()
rank_base["ur_rank"] = rank_base["ur"].rank()
rank_lfpr = latest_lfpr[latest_lfpr["label"] != "US"].copy()
rank_lfpr["lfpr_rank"] = rank_lfpr["lfpr"].rank(ascending=False)
rank_payroll = latest_payroll[latest_payroll["label"] != "US"].copy()
rank_payroll["payroll_rank"] = rank_payroll["payroll_yoy_pct"].rank(ascending=False)

table5 = (
    rank_base[["label", "ur", "ur_rank"]]
    .merge(rank_lfpr[["label", "lfpr", "lfpr_rank"]], on="label", how="left")
    .merge(rank_payroll[["label", "payroll_yoy_pct", "payroll_rank"]], on="label", how="left")
)
table5["overall_rank"] = (
    table5["ur_rank"] + table5["lfpr_rank"] + table5["payroll_rank"]
).rank()
table5 = table5.sort_values("overall_rank").reset_index(drop=True)


# %% [4] Statistical Tests & Econometric Models
# =============================================================================
# SECTION 5 — STATISTICAL TESTS & MODELS
# =============================================================================

welch_results = []
for cyc in CYCLE_ORDER:
    sub = df_spread[df_spread["cycle"] == cyc].dropna(subset=["ga_ur", "us_ur"])
    if len(sub) > 2:
        tt = stats.ttest_ind(sub["ga_ur"], sub["us_ur"], equal_var=False)
        welch_results.append({
            "Cycle": cyc,
            "GA Mean UR": round(sub["ga_ur"].mean(), 2),
            "US Mean UR": round(sub["us_ur"].mean(), 2),
            "Difference": round(sub["ga_ur"].mean() - sub["us_ur"].mean(), 2),
            "p-value": round(tt.pvalue, 4),
            "Sig": "***" if tt.pvalue < 0.01 else ("**" if tt.pvalue < 0.05 else ("*" if tt.pvalue < 0.1 else "")),
        })
df_welch = pd.DataFrame(welch_results)

# Model 1: OLS — GA UR Determinants
print("Running econometric models...")
df_m1 = (
    df_spread[["date", "ga_ur", "us_ur"]]
    .merge(
        df_jolts[df_jolts["label"] == "Job Openings Rate"][["date", "value"]]
        .rename(columns={"value": "jor"}),
        on="date", how="left",
    )
    .merge(df_real_ahe[["date", "ahe_yoy"]], on="date", how="left")
    .dropna(subset=["ga_ur", "us_ur", "jor", "ahe_yoy"])
)
df_m1["trend"] = (df_m1["date"] - df_m1["date"].min()).dt.days / 365.25

m1 = None
m1_table = pd.DataFrame()
if len(df_m1) > 30:
    m1 = smf.ols("ga_ur ~ us_ur + jor + ahe_yoy + trend", data=df_m1).fit(cov_type="HC3")
    m1_rows = []
    for var in m1.params.index:
        m1_rows.append({
            "Variable": var,
            "Coefficient": round(m1.params[var], 4),
            "Robust SE": round(m1.bse[var], 4),
            "t-stat": round(m1.tvalues[var], 3),
            "p-value": round(m1.pvalues[var], 4),
            "Sig": "***" if m1.pvalues[var] < 0.01 else ("**" if m1.pvalues[var] < 0.05 else ("*" if m1.pvalues[var] < 0.1 else "")),
        })
    m1_table = pd.DataFrame(m1_rows)

# Model 2: STL Decomposition
stl_df = pd.DataFrame()
ga_lfpr_ts_data = (
    df_panel_lfpr[
        (df_panel_lfpr["label"] == "Georgia") & df_panel_lfpr["value"].notna()
    ].sort_values("date").copy()
)
if len(ga_lfpr_ts_data) > 36:
    lfpr_series_ts = pd.Series(
        ga_lfpr_ts_data["value"].values,
        index=pd.DatetimeIndex(ga_lfpr_ts_data["date"].values),
    )
    inferred_freq = pd.infer_freq(lfpr_series_ts.index)
    if inferred_freq is None:
        lfpr_series_ts = lfpr_series_ts.resample("MS").mean()
    lfpr_series_ts = lfpr_series_ts.interpolate(method="linear")
    stl_result = STL(lfpr_series_ts, period=12, robust=True).fit()
    stl_df = pd.DataFrame({
        "date": lfpr_series_ts.index,
        "trend": stl_result.trend,
        "seasonal": stl_result.seasonal,
        "remainder": stl_result.resid,
    })

# Model 3: AR(1) Forecast
ar1_model = None
fc_mean = fc_ci = fc_dates = None
pay_series = None
ga_pay_yoy = (
    df_payroll_growth[
        (df_payroll_growth["label"] == "Georgia") & df_payroll_growth["yoy_pct"].notna()
    ].sort_values("date").copy()
)
if len(ga_pay_yoy) > 24:
    pay_series = pd.Series(
        ga_pay_yoy["yoy_pct"].values,
        index=pd.DatetimeIndex(ga_pay_yoy["date"].values),
    )
    inferred_freq = pd.infer_freq(pay_series.index)
    if inferred_freq is None:
        pay_series = pay_series.resample("MS").mean()
    pay_series = pay_series.interpolate(method="linear")
    ar1_model = ARIMA(pay_series, order=(1, 0, 0)).fit()
    fc = ar1_model.get_forecast(steps=12)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=0.05)
    fc_dates = pd.date_range(
        start=pay_series.index[-1] + pd.DateOffset(months=1),
        periods=12, freq="MS",
    )

# Model 4: Shift-Share
print("Running shift-share analysis...")
national_ind_series = [
    ("MANEMP", "Manufacturing"), ("USTPU", "Trade/Transport/Util"),
    ("USPBS", "Prof/Business Svcs"), ("USEHS", "Education/Health"),
    ("USLAH", "Leisure/Hospitality"), ("USGOVT", "Government"),
    ("USCONS", "Construction"), ("USFIRE", "Financial Activities"),
]
df_nat_industry = pd.concat(
    [pull_fred(s, l) for s, l in national_ind_series], ignore_index=True
).dropna(subset=["date", "value"])

ss_end = latest_ind_date
ss_start = ss_end - pd.DateOffset(years=5)


# %% [4B] Shift-Share & Demographic Helpers
def get_closest(df, target_date):
    df = df.dropna(subset=["value"]).copy()
    df["diff"] = (df["date"] - target_date).abs()
    return df.sort_values("diff").head(1).drop(columns=["diff"])


ga_start = (
    df_ga_ind_growth.dropna(subset=["value"])
    .groupby("label").apply(lambda g: get_closest(g, ss_start))
    .reset_index(level=0)[["label", "value"]]
    .rename(columns={"value": "ga_emp_start"})
)
ga_end = (
    df_ga_ind_growth.dropna(subset=["value"])
    .groupby("label").apply(lambda g: get_closest(g, ss_end))
    .reset_index(level=0)[["label", "value"]]
    .rename(columns={"value": "ga_emp_end"})
)
nat_start = (
    df_nat_industry.groupby("label")
    .apply(lambda g: get_closest(g, ss_start))
    .reset_index(level=0)[["label", "value"]]
    .rename(columns={"value": "nat_emp_start"})
)
nat_end = (
    df_nat_industry.groupby("label")
    .apply(lambda g: get_closest(g, ss_end))
    .reset_index(level=0)[["label", "value"]]
    .rename(columns={"value": "nat_emp_end"})
)

nat_total_start = nat_start["nat_emp_start"].sum()
nat_total_end = nat_end["nat_emp_end"].sum()
nat_total_growth = (nat_total_end / nat_total_start) - 1

df_shift_share = (
    ga_start.merge(ga_end, on="label")
    .merge(nat_start, on="label").merge(nat_end, on="label")
)
df_shift_share["ga_change"] = df_shift_share["ga_emp_end"] - df_shift_share["ga_emp_start"]
df_shift_share["nat_ind_growth"] = (
    df_shift_share["nat_emp_end"] / df_shift_share["nat_emp_start"]
) - 1
df_shift_share["national_effect"] = df_shift_share["ga_emp_start"] * nat_total_growth
df_shift_share["industry_mix"] = (
    df_shift_share["ga_emp_start"]
    * (df_shift_share["nat_ind_growth"] - nat_total_growth)
)
df_shift_share["competitive"] = (
    df_shift_share["ga_change"]
    - df_shift_share["national_effect"]
    - df_shift_share["industry_mix"]
)

ss_table = df_shift_share[
    ["label", "ga_change", "national_effect", "industry_mix", "competitive"]
].round(1)
ss_table.columns = ["Industry", "Total Change", "National Effect", "Industry Mix", "Competitive Effect"]

df_cps_ur_plot = df_cps_ur[
    df_cps_ur["value"].notna() & (df_cps_ur["date"] >= START_DATE)
].copy()


# %% [4C] Demographic Group Classifier
def assign_group(label):
    if label in ["Black", "Hispanic", "White"]:
        return "Race/Ethnicity"
    elif label in ["Less than HS", "HS Grad", "Some College", "Bachelors+"]:
        return "Education"
    else:
        return "Sex/Overall"


df_cps_ur_plot["group_type"] = df_cps_ur_plot["label"].apply(assign_group)

df_state_ranks = df_bls_all_states[df_bls_all_states["date"] >= START_DATE].copy()
df_state_ranks["ur_pctile"] = (
    df_state_ranks.groupby("date")["ur"]
    .rank(method="average", pct=True) * 100
)
df_ga_rank = df_state_ranks[df_state_ranks["state_area"] == "Georgia"].sort_values("date")

df_ga_hist = df_bls_laus[df_bls_laus["state_area"] == "Georgia"].sort_values("date").copy()
df_ga_hist["ur_12m_avg"] = df_ga_hist["ur"].rolling(12).mean()

gfc_start = pd.Timestamp("2007-12-01")
covid_start = pd.Timestamp("2020-02-01")
compare_months = 60


# %% [4D] Recession Window Helper
def get_recession_window(df, label_val, rec_start, months):
    sub = df[(df["label"] == label_val) & (df["date"] >= rec_start)
             & (df["date"] <= rec_start + pd.DateOffset(months=months))].sort_values("date").copy()
    sub["t"] = ((sub["date"] - rec_start).dt.days / 30.44).round().astype(int)
    return sub


ga_ur_gfc = get_recession_window(df_panel_ur, "Georgia", gfc_start, compare_months)
ga_ur_covid = get_recession_window(df_panel_ur, "Georgia", covid_start, compare_months)
ga_pay_gfc = get_recession_window(df_fred_payrolls, "Georgia", gfc_start, compare_months)
ga_pay_covid = get_recession_window(df_fred_payrolls, "Georgia", covid_start, compare_months)
if len(ga_pay_gfc) > 0:
    ga_pay_gfc["idx"] = ga_pay_gfc["value"] / ga_pay_gfc["value"].iloc[0] * 100
if len(ga_pay_covid) > 0:
    ga_pay_covid["idx"] = ga_pay_covid["value"] / ga_pay_covid["value"].iloc[0] * 100

df_beveridge = (
    df_jolts[df_jolts["label"] == "Job Openings Rate"]
    .dropna(subset=["value"])[["date", "value"]]
    .rename(columns={"value": "jor"})
    .merge(
        df_fred_ur[df_fred_ur["label"] == "US"][["date", "value"]]
        .rename(columns={"value": "unrate"}),
        on="date", how="inner",
    )
)
df_beveridge["cycle"] = df_beveridge["date"].apply(assign_cycle)

cycle_spread_means = (
    df_spread.dropna(subset=["cycle", "ur_spread"])
    .groupby("cycle")
    .agg(mean_spread=("ur_spread", "mean"),
         start_date=("date", "min"), end_date=("date", "max"))
    .reset_index()
)
cycle_spread_means["mid_date"] = (
    cycle_spread_means["start_date"]
    + (cycle_spread_means["end_date"] - cycle_spread_means["start_date"]) / 2
)

print("Data pipeline complete.\n")


# %% [5] KPI Values
# =============================================================================
# CURRENT KPI VALUES
# =============================================================================

curr_ga_ur = df_panel_ur[df_panel_ur["label"] == "Georgia"].dropna(subset=["value"]).sort_values("date").iloc[-1]["value"]
curr_us_ur = df_panel_ur[df_panel_ur["label"] == "US"].dropna(subset=["value"]).sort_values("date").iloc[-1]["value"]
curr_ga_lfpr = df_panel_lfpr[df_panel_lfpr["label"] == "Georgia"].dropna(subset=["value"]).sort_values("date").iloc[-1]["value"]
curr_us_lfpr = df_panel_lfpr[df_panel_lfpr["label"] == "US"].dropna(subset=["value"]).sort_values("date").iloc[-1]["value"]

se_peer_ur_avg = latest_ur[~latest_ur["label"].isin(["US", "Georgia"])]["ur"].mean()
se_peer_lfpr_avg = latest_lfpr[~latest_lfpr["label"].isin(["US", "Georgia"])]["lfpr"].mean()

ga_payroll_yoy_val = latest_payroll[latest_payroll["label"] == "Georgia"]["payroll_yoy_pct"].values
ga_payroll_yoy = ga_payroll_yoy_val[0] if len(ga_payroll_yoy_val) > 0 else np.nan
us_payroll_yoy_val = latest_payroll[latest_payroll["label"] == "US"]["payroll_yoy_pct"].values
us_payroll_yoy = us_payroll_yoy_val[0] if len(us_payroll_yoy_val) > 0 else np.nan

# Latest GA UR percentile rank — used in exec summary
curr_ga_pctile = df_ga_rank.iloc[-1]["ur_pctile"] if len(df_ga_rank) > 0 else np.nan

# GA UR data point as of date (for "as of" caption)
ga_ur_asof = df_panel_ur[df_panel_ur["label"] == "Georgia"].dropna(subset=["value"]).sort_values("date").iloc[-1]["date"]


# %% [6] Chart Builder Functions
# =============================================================================
# CHART BUILDER FUNCTIONS
# =============================================================================

def build_state_ts(df, headline, takeaway, ylabel, date_min=None, date_max=None):
    """State time series chart (UR, LFPR, etc.)."""
    fig = go.Figure()
    filtered = df.copy()
    if date_min is not None:
        filtered = filtered[filtered["date"] >= date_min]
    if date_max is not None:
        filtered = filtered[filtered["date"] <= date_max]

    for lab in ["Alabama", "Florida", "Tennessee", "North Carolina", "South Carolina"]:
        sub = filtered[filtered["label"] == lab].sort_values("date")
        if len(sub) > 0:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["value"], mode="lines",
                name=lab, line=dict(width=1.2, color=PALETTE.get(lab, "grey")),
                opacity=0.75,
                hovertemplate=f"{lab}: %{{y:.1f}}%<extra></extra>",
            ))
    sub_us = filtered[filtered["label"] == "US"].sort_values("date")
    if len(sub_us) > 0:
        fig.add_trace(go.Scatter(
            x=sub_us["date"], y=sub_us["value"], mode="lines",
            name="US", line=dict(width=2, color="#000000", dash="dash"),
            hovertemplate="US: %{y:.1f}%<extra></extra>",
        ))
    sub_ga = filtered[filtered["label"] == "Georgia"].sort_values("date")
    if len(sub_ga) > 0:
        fig.add_trace(go.Scatter(
            x=sub_ga["date"], y=sub_ga["value"], mode="lines",
            name="Georgia", line=dict(width=3, color=BRAND["accent"]),
            hovertemplate="Georgia: %{y:.1f}%<extra></extra>",
        ))

    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(headline, takeaway),
        yaxis_title=ylabel,
    )
    apply_axis_style(fig)
    return fig


# %% [P1a] Build — UR Time Series
def build_p1a(date_min=None, date_max=None):
    df = df_panel_ur[df_panel_ur["date"] >= START_DATE]
    return build_state_ts(
        df,
        "Georgia has closed its unemployment gap with the nation",
        "From a +1.5pp deficit during the GFC, Georgia now matches or beats US UR.",
        "Unemployment Rate (%)", date_min, date_max,
    )


# %% [P1b] Build — LFPR Time Series
def build_p1b(date_min=None, date_max=None):
    df = df_panel_lfpr[df_panel_lfpr["date"] >= START_DATE]
    return build_state_ts(
        df,
        "Georgia LFPR persistently tracks below the national rate",
        "A structural participation gap, not cyclical — visible across every business cycle.",
        "Labor Force Participation Rate (%)", date_min, date_max,
    )


# %% [P1c] Build — Indexed Payrolls
def build_p1c(date_min=None, date_max=None):
    fig = go.Figure()
    pi = df_payroll_indexed[
        (df_payroll_indexed["date"] >= START_DATE) & df_payroll_indexed["idx_2020"].notna()
    ]
    if date_min is not None:
        pi = pi[pi["date"] >= date_min]
    if date_max is not None:
        pi = pi[pi["date"] <= date_max]

    for lab in ["Alabama", "Florida", "Tennessee", "North Carolina", "South Carolina"]:
        sub = pi[pi["label"] == lab].sort_values("date")
        if len(sub) > 0:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["idx_2020"], mode="lines",
                name=lab, line=dict(width=1.2, color=PALETTE.get(lab, "grey")),
                opacity=0.75,
                hovertemplate=f"{lab}: %{{y:.1f}}<extra></extra>",
            ))
    sub_us = pi[pi["label"] == "US"].sort_values("date")
    if len(sub_us) > 0:
        fig.add_trace(go.Scatter(
            x=sub_us["date"], y=sub_us["idx_2020"], mode="lines",
            name="US", line=dict(width=2, color="#000000", dash="dash"),
            hovertemplate="US: %{y:.1f}<extra></extra>",
        ))
    sub_ga = pi[pi["label"] == "Georgia"].sort_values("date")
    if len(sub_ga) > 0:
        fig.add_trace(go.Scatter(
            x=sub_ga["date"], y=sub_ga["idx_2020"], mode="lines",
            name="Georgia", line=dict(width=3, color=BRAND["accent"]),
            hovertemplate="Georgia: %{y:.1f}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="#808080", line_width=0.5)
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Georgia recovered to pre-pandemic payrolls faster than US average",
            "Indexed to Feb 2020 = 100. Higher line = stronger recovery.",
        ),
        yaxis_title="Index (Feb 2020 = 100)",
    )
    apply_axis_style(fig)
    return fig


# %% [P1d] Build — YoY Payroll Growth
def build_p1d(date_min=None, date_max=None):
    fig = go.Figure()
    pg = df_payroll_growth[
        (df_payroll_growth["date"] >= START_DATE) & df_payroll_growth["yoy_pct"].notna()
    ]
    if date_min is not None:
        pg = pg[pg["date"] >= date_min]
    if date_max is not None:
        pg = pg[pg["date"] <= date_max]

    for lab in ["Alabama", "Florida", "Tennessee", "North Carolina", "South Carolina"]:
        sub = pg[pg["label"] == lab].sort_values("date")
        if len(sub) > 0:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["yoy_pct"], mode="lines",
                name=lab, line=dict(width=1.2, color=PALETTE.get(lab, "grey")),
                opacity=0.75,
                hovertemplate=f"{lab}: %{{y:.1f}}%<extra></extra>",
            ))
    sub_us = pg[pg["label"] == "US"].sort_values("date")
    if len(sub_us) > 0:
        fig.add_trace(go.Scatter(
            x=sub_us["date"], y=sub_us["yoy_pct"], mode="lines",
            name="US", line=dict(width=2, color="#000000", dash="dash"),
            hovertemplate="US: %{y:.1f}%<extra></extra>",
        ))
    sub_ga = pg[pg["label"] == "Georgia"].sort_values("date")
    if len(sub_ga) > 0:
        fig.add_trace(go.Scatter(
            x=sub_ga["date"], y=sub_ga["yoy_pct"], mode="lines",
            name="Georgia", line=dict(width=3, color=BRAND["accent"]),
            hovertemplate="Georgia: %{y:.1f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#808080", line_width=0.5)
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Year-over-year payroll growth tracks above the US trend",
            "Georgia consistently in the top half of SE peers since 2021.",
        ),
        yaxis_title="YoY Growth (%)",
    )
    apply_axis_style(fig)
    return fig


# %% [P9] Build — GA vs US Spread
def build_p9(date_min=None, date_max=None):
    fig = go.Figure()
    sp = df_spread[df_spread["date"] >= START_DATE].sort_values("date")
    if date_min is not None:
        sp = sp[sp["date"] >= date_min]
    if date_max is not None:
        sp = sp[sp["date"] <= date_max]

    fig.add_trace(go.Scatter(
        x=sp["date"], y=sp["ur_spread"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(196,69,54,0.35)",
        line=dict(width=0), name="GA worse than US",
        hovertemplate="%{y:+.2f}pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sp["date"], y=sp["ur_spread"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(0,158,115,0.35)",
        line=dict(width=0), name="GA better than US",
        hovertemplate="%{y:+.2f}pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sp["date"], y=sp["ur_spread"],
        mode="lines", line=dict(width=0.8, color=BRAND["neutral"]),
        name="Spread", hovertemplate="Spread: %{y:+.2f}pp<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.5)

    for _, row in cycle_spread_means.iterrows():
        fig.add_shape(
            type="line", x0=row["start_date"], x1=row["end_date"],
            y0=row["mean_spread"], y1=row["mean_spread"],
            line=dict(color="black", width=1.5),
        )
        fig.add_annotation(
            x=row["mid_date"], y=row["mean_spread"],
            text=f"<b>{row['cycle']}</b><br>{row['mean_spread']:+.2f}pp",
            showarrow=False, font=dict(size=9, color="black"),
            yshift=18,
        )

    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Georgia's UR has flipped from above-national to at-or-below-national",
            "Cycle-mean spread averaged +0.5pp pre-2010; now consistently negative.",
        ),
        yaxis_title="Spread (pp): GA UR minus US UR",
    )
    apply_axis_style(fig)
    return fig


# %% [P2] Build — Prime-Age LFPR
def build_p2():
    fig = go.Figure()
    prime_labels_map = {
        "Prime-Age Men 25-54": "#0072B2",
        "Prime-Age Women 25-54": BRAND["accent"],
        "Prime-Age 25-54": "#000000",
    }
    df_prime = df_cps_lfpr[
        df_cps_lfpr["label"].isin(prime_labels_map.keys())
        & df_cps_lfpr["value"].notna()
        & (df_cps_lfpr["date"] >= START_DATE)
    ]
    pre_covid = df_prime[df_prime["date"] == "2020-01-01"]
    for _, row in pre_covid.iterrows():
        fig.add_hline(y=row["value"], line_dash="dot", line_color="#696969", line_width=0.5,
                      annotation_text=f"Pre-COVID: {row['value']:.1f}%",
                      annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#808080")

    for lab, color in prime_labels_map.items():
        sub = df_prime[df_prime["label"] == lab].sort_values("date")
        if len(sub) > 0:
            lw = 2.5 if lab == "Prime-Age 25-54" else 1.5
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["value"], mode="lines",
                name=lab, line=dict(width=lw, color=color),
                hovertemplate=f"{lab}: %{{y:.1f}}%<extra></extra>",
            ))

    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Prime-age participation has not fully recovered from COVID",
            "The most economically active cohort still sits below 2019 levels.",
        ),
        yaxis_title="Labor Force Participation Rate (%)",
        yaxis=dict(range=[65, 95]),
    )
    apply_axis_style(fig)
    return fig


# %% [P3] Build — Demographic Bars
def build_p3():
    bar_lfpr = (
        df_cps_lfpr.dropna(subset=["value"]).sort_values("date")
        .groupby("label").tail(1)[["label", "value"]]
    )
    bar_ur = (
        df_cps_ur[df_cps_ur["label"] != "All"]
        .dropna(subset=["value"]).sort_values("date")
        .groupby("label").tail(1)[["label", "value"]]
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>Participation Rate (%)</b>", "<b>Unemployment Rate (%)</b>"],
        horizontal_spacing=0.18,
    )
    bl = bar_lfpr.sort_values("value")
    avg_lfpr = bl["value"].mean()
    colors_lfpr = [BRAND["good"] if v >= avg_lfpr else BRAND["bad"] for v in bl["value"]]
    fig.add_trace(go.Bar(
        y=bl["label"], x=bl["value"], orientation="h",
        marker_color=colors_lfpr, marker_line_width=0,
        text=[f"{v:.1f}%" for v in bl["value"]],
        textposition="outside", name="LFPR", showlegend=False,
    ), row=1, col=1)
    fig.add_vline(x=avg_lfpr, line_dash="dash", line_color="#999999", row=1, col=1)

    bu = bar_ur.sort_values("value")
    avg_ur = bu["value"].mean()
    colors_ur = [BRAND["good"] if v <= avg_ur else BRAND["bad"] for v in bu["value"]]
    fig.add_trace(go.Bar(
        y=bu["label"], x=bu["value"], orientation="h",
        marker_color=colors_ur, marker_line_width=0,
        text=[f"{v:.1f}%" for v in bu["value"]],
        textposition="outside", name="UR", showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=avg_ur, line_dash="dash", line_color="#999999", row=1, col=2)

    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Education and gender drive the largest gaps in the labor market",
            "Bachelor's+ workers face roughly half the unemployment rate of those without a HS diploma.",
        ),
        height=560,
    )
    apply_axis_style(fig)
    return fig


# %% [P4] Build — UR by Demographic (Faceted)
def build_p4(group_type="Sex/Overall"):
    filtered = df_cps_ur_plot[df_cps_ur_plot["group_type"] == group_type]
    fig = go.Figure()
    colors_list = px.colors.qualitative.Dark2
    for i, lab in enumerate(filtered["label"].unique()):
        sub = filtered[filtered["label"] == lab].sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["value"], mode="lines",
            name=lab, line=dict(width=1.8, color=colors_list[i % len(colors_list)]),
            hovertemplate=f"{lab}: %{{y:.1f}}%<extra></extra>",
        ))
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Unemployment risk varies sharply by demographic group",
            "College graduates weather every recession with materially lower unemployment.",
        ),
        yaxis_title="Unemployment Rate (%)",
    )
    apply_axis_style(fig)
    return fig


# %% [P5a] Build — GA Real GDP Growth
def build_p5a():
    gdp_plot = df_ga_gdp_growth.dropna(subset=["yoy_pct"])
    colors_gdp = [BRAND["good"] if v >= 0 else BRAND["bad"] for v in gdp_plot["yoy_pct"]]
    fig = go.Figure(go.Bar(
        x=gdp_plot["date"], y=gdp_plot["yoy_pct"],
        marker_color=colors_gdp, marker_line_width=0,
        hovertemplate="Date: %{x|%Y-%m}<br>YoY: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=0.5)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Georgia outpaced national GDP growth through the post-COVID boom",
            "Real GDP YoY % — green denotes growth, red denotes contraction.",
        ),
        yaxis_title="YoY Growth (%)", showlegend=False,
    )
    apply_axis_style(fig)
    return fig


# %% [P5b] Build — Per Capita Income
def build_p5b():
    fig = go.Figure()
    for geo, color, label_name in [
        ("GA Per Capita Personal Income", BRAND["accent"], "Georgia"),
        ("US Per Capita Personal Income", "#000000", "US"),
    ]:
        src = df_ga_pcpi if "GA" in geo else df_us_pcpi
        sub = src.dropna(subset=["value"]).sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["value"], mode="lines",
            name=label_name,
            line=dict(width=2.2 if label_name == "Georgia" else 1.5,
                      color=color, dash="dash" if label_name == "US" else "solid"),
            hovertemplate=f"{label_name}: $%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Georgia per-capita income has grown but still trails the US",
            "The gap has narrowed slightly but persisted across every cycle since 2000.",
        ),
        yaxis_title="Per Capita Personal Income ($)",
        yaxis_tickformat="$,.0f",
    )
    apply_axis_style(fig)
    return fig


# %% [P5c] Build — Industry Employment Bars
def build_p5c():
    ind = df_ga_ind_shares.sort_values("value")
    colors_ind = [INDUSTRY_PALETTE.get(lab, "#999999") for lab in ind["label"]]
    fig = go.Figure(go.Bar(
        y=ind["label"], x=ind["value"], orientation="h",
        marker_color=colors_ind, marker_line_width=0,
        text=[f"{v:.0f}K  ({s:.1f}%)" for v, s in zip(ind["value"], ind["share"])],
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f}K<extra></extra>",
    ))
    fig.update_layout(
        **common_layout(),
        title=make_title(
            f"Trade, Education/Health, and Government anchor Georgia's employment base",
            f"Snapshot of total nonfarm payrolls by sector as of {latest_ind_date.date()}.",
        ),
        xaxis_title="Employment (thousands)", showlegend=False,
    )
    apply_axis_style(fig)
    return fig


# %% [P6] Build — Industry Stacked Area / Toggle
def build_p6(chart_type="area"):
    ind_data = df_ga_ind_growth[
        df_ga_ind_growth["value"].notna() & (df_ga_ind_growth["date"] >= START_DATE)
    ]
    ind_wide = ind_data.pivot_table(index="date", columns="label", values="value").sort_index()
    labels_ordered = ind_wide.columns.tolist()

    fig = go.Figure()
    if chart_type == "area":
        for lab in labels_ordered:
            fig.add_trace(go.Scatter(
                x=ind_wide.index, y=ind_wide[lab], mode="lines",
                name=lab, stackgroup="one",
                line=dict(width=0.5, color=INDUSTRY_PALETTE.get(lab, "#999999")),
                fillcolor=INDUSTRY_PALETTE.get(lab, "#999999"),
                hovertemplate=f"{lab}: %{{y:.0f}}K<extra></extra>",
            ))
    elif chart_type == "line":
        for lab in labels_ordered:
            fig.add_trace(go.Scatter(
                x=ind_wide.index, y=ind_wide[lab], mode="lines",
                name=lab, line=dict(width=1.8, color=INDUSTRY_PALETTE.get(lab, "#999999")),
                hovertemplate=f"{lab}: %{{y:.0f}}K<extra></extra>",
            ))
    elif chart_type == "bar":
        latest = ind_wide.iloc[-1]
        fig.add_trace(go.Bar(
            x=latest.values, y=latest.index, orientation="h",
            marker_color=[INDUSTRY_PALETTE.get(l, "#999999") for l in latest.index],
            marker_line_width=0,
            text=[f"{v:.0f}K" for v in latest.values],
            textposition="outside",
        ))

    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Professional Services and Education/Health drive secular employment growth",
            "Manufacturing share has held steady; service-sector growth dominates the trend.",
        ),
        yaxis_title="Employment (thousands)" if chart_type != "bar" else "",
    )
    apply_axis_style(fig)
    return fig


# %% [P7] Build — Beveridge Curve
def build_p7():
    fig = go.Figure()
    bev_sorted = df_beveridge.sort_values("date")
    fig.add_trace(go.Scatter(
        x=bev_sorted["unrate"], y=bev_sorted["jor"],
        mode="lines", line=dict(width=0.5, color="#cccccc"),
        showlegend=False, hoverinfo="skip",
    ))
    for cyc, color in CYCLE_COLORS.items():
        sub = df_beveridge[df_beveridge["cycle"] == cyc]
        fig.add_trace(go.Scatter(
            x=sub["unrate"], y=sub["jor"], mode="markers",
            name=cyc, marker=dict(size=7, color=color, opacity=0.75,
                                  line=dict(width=0.3, color="white")),
            hovertemplate=f"{cyc}<br>UR: %{{x:.1f}}%<br>JOR: %{{y:.1f}}%<br>%{{text}}<extra></extra>",
            text=sub["date"].dt.strftime("%b %Y"),
        ))
    bev_latest = df_beveridge.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[bev_latest["unrate"]], y=[bev_latest["jor"]],
        mode="markers", name="Current",
        marker=dict(size=14, color=BRAND["bad"], symbol="diamond",
                    line=dict(width=1.5, color="white")),
        hovertemplate=f"Current ({bev_latest['date'].strftime('%b %Y')})<br>UR: %{{x:.1f}}%<br>JOR: %{{y:.1f}}%<extra></extra>",
    ))

    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Beveridge Curve has shifted outward — labor scarcity is structural",
            "Same UR levels now coexist with materially higher job-opening rates than pre-2020.",
        ),
        xaxis_title="Unemployment Rate (%)",
        yaxis_title="Job Openings Rate (%)",
    )
    # Override hovermode for scatter (overrides common_layout's "x unified" default)
    fig.update_layout(hovermode="closest")
    apply_axis_style(fig)
    return fig


# %% [P7b] Build — Tightness Ratio Time Series
def build_tightness_ts():
    fig = go.Figure()
    t = df_tightness.sort_values("date")
    fig.add_trace(go.Scatter(
        x=t["date"], y=t["tightness"], mode="lines",
        line=dict(width=2.2, color=BRAND["accent"]),
        hovertemplate="Tightness: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1, line_dash="dash", line_color="#808080", line_width=0.5,
                  annotation_text="Balanced (ratio = 1)", annotation_position="top left")
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Labor market tightness peaked in 2022 and is normalizing",
            "Job openings per unemployed person — a forward indicator of wage pressure.",
        ),
        yaxis_title="Tightness Ratio (JOR ÷ UR)",
    )
    apply_axis_style(fig)
    return fig


# %% [P8a] Build — Nominal vs Real AHE
def build_p8a():
    df_wp = df_real_ahe[
        df_real_ahe["ahe_yoy"].notna() & (df_real_ahe["date"] >= START_DATE)
    ].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_wp["date"], y=df_wp["ahe_yoy"], mode="lines",
        name="Nominal wages YoY%", line=dict(width=1.7, color="#0072B2"),
        hovertemplate="Nominal: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_wp["date"], y=df_wp["real_yoy"], mode="lines",
        name="Real wages YoY%", line=dict(width=1.7, color=BRAND["accent"]),
        hovertemplate="Real: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_wp["date"], df_wp["date"][::-1]]),
        y=pd.concat([df_wp["ahe_yoy"], df_wp["real_yoy"][::-1]]),
        fill="toself", fillcolor="rgba(230,159,0,0.15)",
        line=dict(width=0), name="Inflation gap", showlegend=True,
        hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#808080", line_width=0.5)
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Real wages turned positive in 2023 after two years of inflation erosion",
            "Workers are gaining purchasing power again as inflation cools below wage growth.",
        ),
        yaxis_title="Year-over-Year %",
    )
    apply_axis_style(fig)
    return fig


# %% [P8b] Build — ECI YoY Growth
def build_p8b():
    eci_plot = df_eci_growth[
        df_eci_growth["eci_yoy"].notna() & (df_eci_growth["date"] >= START_DATE)
    ].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eci_plot["date"], y=eci_plot["eci_yoy"], mode="lines",
        line=dict(width=1.8, color=BRAND["good"]),
        hovertemplate="ECI YoY: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#808080", line_width=0.5)
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Employment Cost Index growth is decelerating from 2022 highs",
            "Total compensation (wages + benefits) — the Fed's preferred wage gauge.",
        ),
        yaxis_title="Year-over-Year %",
    )
    apply_axis_style(fig)
    return fig


# %% [P10] Build — GA UR Percentile Rank
def build_p10():
    fig = go.Figure()
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(196,69,54,0.05)", line_width=0)
    fig.add_hrect(y0=0, y1=25, fillcolor="rgba(0,158,115,0.08)", line_width=0)

    fig.add_trace(go.Scatter(
        x=df_ga_rank["date"], y=df_ga_rank["ur_pctile"], mode="lines",
        line=dict(width=1.5, color=BRAND["accent"]),
        hovertemplate="Percentile: %{y:.0f}<extra></extra>",
        showlegend=False,
    ))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        numeric_dates = (df_ga_rank["date"] - df_ga_rank["date"].min()).dt.days.values.astype(float)
        smooth = lowess(df_ga_rank["ur_pctile"].values, numeric_dates, frac=0.15)
        smooth_dates = df_ga_rank["date"].min() + pd.to_timedelta(smooth[:, 0], unit="D")
        fig.add_trace(go.Scatter(
            x=smooth_dates, y=smooth[:, 1], mode="lines",
            line=dict(width=1.2, color="black", dash="dash"),
            name="LOWESS Trend", showlegend=True,
            hoverinfo="skip",
        ))
    except Exception:
        pass

    fig.add_hline(y=50, line_dash="dash", line_color="#808080", line_width=0.5)
    fig.add_hline(y=25, line_dash="dot", line_color=BRAND["good"], line_width=0.4)

    ga_latest = df_ga_rank.iloc[-1]
    fig.add_annotation(
        x=ga_latest["date"], y=ga_latest["ur_pctile"],
        text=f"Current: {ga_latest['ur_pctile']:.0f}th pctile",
        showarrow=True, arrowhead=2,
        font=dict(size=11, color=BRAND["accent"], family="Inter, sans-serif"),
        bgcolor="white", bordercolor=BRAND["accent"], borderpad=4,
    )

    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "Georgia climbed from below-median to top-quartile on UR among states",
            "Lower percentile = lower unemployment relative to other US states.",
        ),
        yaxis_title="Percentile Rank (lower = better)",
        yaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100],
                   ticktext=["1st (lowest UR)", "25th", "50th", "75th", "100th (highest UR)"]),
    )
    apply_axis_style(fig)
    return fig


# %% [P12] Build — 50-Year Historical GA UR
def build_p12():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ga_hist["date"], y=df_ga_hist["ur"], mode="lines",
        name="Monthly", line=dict(width=0.6, color=BRAND["accent"]),
        opacity=0.4, hovertemplate="Monthly UR: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_ga_hist["date"], y=df_ga_hist["ur_12m_avg"], mode="lines",
        name="12-mo Rolling Avg", line=dict(width=2.2, color=BRAND["accent"]),
        hovertemplate="12-mo Avg: %{y:.1f}%<extra></extra>",
    ))
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "50 years of Georgia unemployment: from double-digits to historic lows",
            "The current cycle low rivals the late-1990s peak expansion.",
        ),
        yaxis_title="Unemployment Rate (%)",
    )
    apply_axis_style(fig)
    return fig


# %% [P11] Build — GFC vs COVID Comparison
def build_p11():
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["<b>Georgia Unemployment Rate</b>",
                        "<b>Georgia Nonfarm Payrolls (Indexed to 100 at Recession Start)</b>"],
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(
        x=ga_ur_gfc["t"], y=ga_ur_gfc["value"], mode="lines",
        name="GFC (Dec 2007)", line=dict(width=2.2, color="#0072B2"),
        hovertemplate="GFC t=%{x}: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ga_ur_covid["t"], y=ga_ur_covid["value"], mode="lines",
        name="COVID (Feb 2020)", line=dict(width=2.2, color=BRAND["accent"]),
        hovertemplate="COVID t=%{x}: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    if len(ga_pay_gfc) > 0:
        fig.add_trace(go.Scatter(
            x=ga_pay_gfc["t"], y=ga_pay_gfc["idx"], mode="lines",
            name="GFC Payrolls", line=dict(width=2.2, color="#0072B2"),
            showlegend=False,
            hovertemplate="GFC t=%{x}: %{y:.1f}<extra></extra>",
        ), row=2, col=1)
    if len(ga_pay_covid) > 0:
        fig.add_trace(go.Scatter(
            x=ga_pay_covid["t"], y=ga_pay_covid["idx"], mode="lines",
            name="COVID Payrolls", line=dict(width=2.2, color=BRAND["accent"]),
            showlegend=False,
            hovertemplate="COVID t=%{x}: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=100, line_dash="dash", line_color="#808080", line_width=0.5, row=2, col=1)

    fig.update_layout(
        **common_layout(),
        title=make_title(
            "COVID shock was sharper but recovery was roughly 3x faster than the GFC",
            "Same state, two very different recessions — months on x-axis aligned at recession start.",
        ),
        height=620,
    )
    fig.update_xaxes(title_text="Months Since Recession Start", row=2, col=1)
    fig.update_yaxes(title_text="Percent", row=1, col=1)
    fig.update_yaxes(title_text="Index", row=2, col=1)
    apply_axis_style(fig)
    return fig


# %% [M2] Build — STL Decomposition
def build_m2_stl():
    if len(stl_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="STL data unavailable", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=18))
        return fig

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["<b>Trend Component</b>", "<b>Seasonal Component</b>",
                        "<b>Remainder (Irregular) Component</b>"],
        vertical_spacing=0.10,
    )
    fig.add_trace(go.Scatter(
        x=stl_df["date"], y=stl_df["trend"], mode="lines",
        line=dict(width=1.8, color=BRAND["accent"]), name="Trend",
        hovertemplate="Trend: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    add_recessions(fig, rec_blocks)

    fig.add_trace(go.Scatter(
        x=stl_df["date"], y=stl_df["seasonal"], mode="lines",
        line=dict(width=1, color="#0072B2"), name="Seasonal",
        hovertemplate="Seasonal: %{y:.3f}pp<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#808080", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=stl_df["date"], y=stl_df["remainder"], mode="lines",
        line=dict(width=1, color=BRAND["good"]), name="Remainder",
        hovertemplate="Remainder: %{y:.3f}pp<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#808080", row=3, col=1)

    fig.update_layout(
        **common_layout(),
        title=make_title(
            "STL decomposition isolates the structural trend from cyclical and seasonal noise",
            "Georgia LFPR series — separating the signal from the noise.",
        ),
        height=680,
    )
    fig.update_yaxes(title_text="LFPR (%)", row=1, col=1)
    fig.update_yaxes(title_text="pp", row=2, col=1)
    fig.update_yaxes(title_text="pp", row=3, col=1)
    apply_axis_style(fig)
    return fig


# %% [M3] Build — AR(1) Forecast
def build_m3_ar1():
    if ar1_model is None or pay_series is None:
        fig = go.Figure()
        fig.add_annotation(text="AR(1) model unavailable", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=18))
        return fig

    fig = go.Figure()
    mask_2015 = pay_series.index >= "2015-01-01"
    fig.add_trace(go.Scatter(
        x=pay_series.index[mask_2015], y=pay_series.values[mask_2015],
        mode="lines", name="Actual", line=dict(width=1.8, color="black"),
        hovertemplate="Actual: %{y:.1f}%<extra></extra>",
    ))
    fitted_vals = ar1_model.fittedvalues
    fig.add_trace(go.Scatter(
        x=fitted_vals.index[mask_2015], y=fitted_vals.values[mask_2015],
        mode="lines", name="Fitted", line=dict(width=1, color="#0072B2", dash="dash"),
        hovertemplate="Fitted: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_mean.values, mode="lines",
        name="Forecast", line=dict(width=2.4, color=BRAND["accent"]),
        hovertemplate="Forecast: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(fc_dates) + list(fc_dates[::-1]),
        y=list(fc_ci.iloc[:, 1].values) + list(fc_ci.iloc[:, 0].values[::-1]),
        fill="toself", fillcolor="rgba(213,94,0,0.15)",
        line=dict(width=0), name="95% CI", showlegend=True,
        hoverinfo="skip",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#808080", line_width=0.5)
    add_recessions(fig, rec_blocks)
    fig.update_layout(
        **common_layout(),
        title=make_title(
            "AR(1) model projects continued moderate payroll growth over next 12 months",
            "Forecast (orange) with 95% confidence interval — assumes mean-reverting dynamics.",
        ),
        yaxis_title="YoY Growth (%)",
    )
    apply_axis_style(fig)
    return fig


# %% [7] Dash App — Initialise
# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        # Inter — modern, executive-feeling sans-serif from Google Fonts
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="GA Labor Market Dashboard | John Brassard",
)

# Expose Flask server for gunicorn (required for Render deployment)
server = app.server

# Inject base CSS for typography & polish
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background-color: #F7F8FA;
                color: #1A1A1A;
                -webkit-font-smoothing: antialiased;
            }
            .navbar { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
            .card {
                border: 1px solid #E5E8EC !important;
                border-radius: 6px !important;
                background-color: #FFFFFF !important;
            }
            .nav-tabs {
                border-bottom: 2px solid #E5E8EC;
                margin-bottom: 0;
            }
            .nav-tabs .nav-link {
                color: #5A6C7D;
                font-weight: 500;
                border: none;
                padding: 0.85rem 1.25rem;
                font-size: 0.92rem;
                letter-spacing: 0.01em;
            }
            .nav-tabs .nav-link:hover {
                color: #1F3A5F;
                background-color: rgba(31,58,95,0.04);
                border: none;
            }
            .nav-tabs .nav-link.active {
                color: #D55E00 !important;
                font-weight: 600 !important;
                border-bottom: 3px solid #D55E00 !important;
                background-color: transparent !important;
            }
            h1, h2, h3, h4, h5, h6 { font-weight: 600; letter-spacing: -0.01em; }
            .kpi-value {
                font-size: 2.1rem;
                font-weight: 700;
                line-height: 1.1;
                letter-spacing: -0.02em;
            }
            .kpi-label {
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #5A6C7D;
            }
            .kpi-delta {
                font-size: 0.8rem;
                font-weight: 500;
            }
            .takeaway-card {
                border-left: 4px solid #D55E00 !important;
                background-color: #FFFFFF !important;
                padding: 1.25rem 1.5rem;
            }
            .section-header {
                font-size: 1.15rem;
                font-weight: 600;
                color: #1F3A5F;
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
                padding-bottom: 0.4rem;
                border-bottom: 1px solid #E5E8EC;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# %% [7B] KPI Card Component
def kpi_card(label, value, delta_text, delta_context, good_when="lower"):
    """
    Refined KPI card.
    good_when: 'lower' (UR is good when down) or 'higher' (LFPR/payrolls good when up)
    """
    # Parse the numeric delta to determine color
    try:
        delta_val = float(delta_text.replace("+", "").replace("pp", "").replace("%", "").strip())
        if good_when == "lower":
            color = BRAND["good"] if delta_val <= 0 else BRAND["bad"]
            arrow = "▼" if delta_val < 0 else ("▲" if delta_val > 0 else "—")
        else:
            color = BRAND["good"] if delta_val >= 0 else BRAND["bad"]
            arrow = "▲" if delta_val > 0 else ("▼" if delta_val < 0 else "—")
    except Exception:
        color = BRAND["neutral"]
        arrow = ""

    return dbc.Card(
        dbc.CardBody([
            html.Div(label, className="kpi-label mb-2"),
            html.Div(value, className="kpi-value", style={"color": BRAND["primary"]}),
            html.Div([
                html.Span(f"{arrow} {delta_text}", style={"color": color, "fontWeight": 600}),
                html.Span(f" {delta_context}",
                          style={"color": BRAND["ink_soft"], "fontWeight": 400}),
            ], className="kpi-delta mt-2"),
        ]),
        className="shadow-sm h-100",
    )


# %% [7C] Date Slider Range
all_dates = df_panel_ur[df_panel_ur["date"] >= START_DATE]["date"].dropna().sort_values().unique()
date_origin = pd.Timestamp("2000-01-01")
min_month = 0
max_month = int((pd.Timestamp(all_dates[-1]) - date_origin).days / 30.44) if len(all_dates) > 0 else 300


# %% [7D] Navbar
navbar = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Div("GEORGIA LABOR MARKET",
                     style={"color": "rgba(255,255,255,0.65)",
                            "fontSize": "0.7rem",
                            "fontWeight": 600,
                            "letterSpacing": "0.15em",
                            "marginBottom": "2px"}),
            html.H4("Research Dashboard",
                    style={"color": "white", "margin": "0",
                           "fontWeight": 700, "letterSpacing": "-0.01em"}),
        ]),
        html.Div([
            html.Div("DATA SOURCES",
                     style={"color": "rgba(255,255,255,0.55)",
                            "fontSize": "0.65rem",
                            "fontWeight": 600,
                            "letterSpacing": "0.12em",
                            "textAlign": "right"}),
            html.Div("BLS LAUS · FRED · CPS · JOLTS · BEA",
                     style={"color": "rgba(255,255,255,0.85)",
                            "fontSize": "0.78rem", "textAlign": "right"}),
            html.Div(f"Updated {pd.Timestamp.today().strftime('%b %d, %Y')}",
                     style={"color": "rgba(255,255,255,0.55)",
                            "fontSize": "0.7rem", "textAlign": "right"}),
        ]),
    ], fluid=True,
       className="d-flex justify-content-between align-items-center"),
    color=BRAND["primary"], dark=True, sticky="top",
    style={"padding": "0.85rem 0"},
)


# %% [7E0] Tab 0 — Executive Summary
ga_us_ur_diff = curr_ga_ur - curr_us_ur
ga_us_lfpr_diff = curr_ga_lfpr - curr_us_lfpr

# Determine direction language for headlines
ur_verdict = (
    "outperforming" if ga_us_ur_diff < 0
    else ("matching" if abs(ga_us_ur_diff) < 0.15 else "trailing")
)
lfpr_verdict = (
    "above" if ga_us_lfpr_diff > 0
    else ("near" if abs(ga_us_lfpr_diff) < 0.5 else "below")
)


def takeaway_card(number, headline, body):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(number,
                          style={"fontSize": "1.6rem", "fontWeight": 700,
                                 "color": BRAND["accent"], "marginRight": "0.6rem"}),
                html.Span(headline,
                          style={"fontSize": "1rem", "fontWeight": 600,
                                 "color": BRAND["primary"]}),
            ], className="mb-2"),
            html.P(body, className="mb-0",
                   style={"color": BRAND["ink"], "fontSize": "0.92rem",
                          "lineHeight": "1.55"}),
        ]),
        className="takeaway-card shadow-sm h-100",
    )


tab0_content = dbc.Container([
    # Hero header strip
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("EXECUTIVE SUMMARY", className="kpi-label mb-2"),
                html.H2("Georgia's labor market: from regional laggard to top-quartile performer",
                        style={"fontWeight": 700, "color": BRAND["primary"],
                               "letterSpacing": "-0.02em", "marginBottom": "0.5rem"}),
                html.P(f"A 25-year retrospective of unemployment, participation, payrolls, and "
                       f"industry composition. Updated {pd.Timestamp.today().strftime('%B %Y')}.",
                       style={"color": BRAND["ink_soft"], "fontSize": "1rem"}),
            ], className="py-3"),
        ], width=12),
    ]),

    # KPIs
    dbc.Row([
        dbc.Col(kpi_card(
            "Georgia Unemployment Rate",
            f"{curr_ga_ur:.1f}%",
            f"{ga_us_ur_diff:+.1f}pp",
            f"vs US ({curr_us_ur:.1f}%)",
            good_when="lower",
        ), width=12, md=4, className="mb-3"),
        dbc.Col(kpi_card(
            "Labor Force Participation",
            f"{curr_ga_lfpr:.1f}%",
            f"{ga_us_lfpr_diff:+.1f}pp",
            f"vs US ({curr_us_lfpr:.1f}%)",
            good_when="higher",
        ), width=12, md=4, className="mb-3"),
        dbc.Col(kpi_card(
            "Payroll Growth (YoY)",
            f"{ga_payroll_yoy:+.1f}%",
            f"{ga_payroll_yoy - us_payroll_yoy:+.1f}pp",
            f"vs US ({us_payroll_yoy:+.1f}%)",
            good_when="higher",
        ), width=12, md=4, className="mb-3"),
    ], className="mb-4"),

    # Three big takeaways
    html.Div("KEY FINDINGS", className="kpi-label mb-3"),
    dbc.Row([
        dbc.Col(takeaway_card(
            "01",
            "Convergence with the national economy",
            f"Georgia's UR has flipped from a +1.5pp deficit during the GFC to "
            f"{ur_verdict} the national average. The state now ranks in the top quartile "
            f"of US states on unemployment — a structural improvement, not a cyclical blip.",
        ), width=12, lg=4, className="mb-3"),
        dbc.Col(takeaway_card(
            "02",
            "A persistent participation gap",
            f"LFPR sits {lfpr_verdict} the US average — a structural gap visible across "
            f"every business cycle since 2000. Prime-age participation has yet to fully "
            f"recover from COVID, suggesting a labor supply ceiling that constrains "
            f"future growth.",
        ), width=12, lg=4, className="mb-3"),
        dbc.Col(takeaway_card(
            "03",
            "Service-sector–led growth",
            "Professional Services and Education/Health drive secular employment growth. "
            "Shift-share decomposition shows Georgia's outperformance is driven primarily "
            "by competitive advantages within industries, not favorable industry mix alone.",
        ), width=12, lg=4, className="mb-3"),
    ], className="mb-4"),

    # Headline chart — the spread story
    html.Div("THE HEADLINE CHART", className="kpi-label mb-2 mt-3"),
    dbc.Card(
        dbc.CardBody(dcc.Graph(figure=build_p9(), config={"displayModeBar": False})),
        className="shadow-sm mb-4",
    ),

    # Methodology / source note
    dbc.Card(
        dbc.CardBody([
            html.Div("METHODOLOGY", className="kpi-label mb-2"),
            html.P([
                "Built in Python (pandas, plotly, statsmodels, dash). Data pulled from "
                "BLS Local Area Unemployment Statistics, FRED, BLS Current Population Survey, "
                "and BLS Job Openings and Labor Turnover Survey. Econometric work includes "
                "OLS with HC3 robust standard errors, STL decomposition, AR(1) time-series "
                "forecasting, shift-share decomposition, and Welch's t-tests across business "
                "cycles. See the ",
                html.B("Econometric Models"), " tab for detailed specifications.",
            ], style={"fontSize": "0.88rem", "color": BRAND["ink_soft"],
                       "lineHeight": "1.6", "marginBottom": "0"}),
        ]),
        className="shadow-sm mb-4",
        style={"backgroundColor": "#FAFBFC"},
    ),
], fluid=True, className="mt-4 mb-4", style={"maxWidth": "1400px"})


# %% [7E] Tab 1 — State Overview
tab1_content = dbc.Container([
    html.Div("PEER COMPARISON", className="kpi-label mb-3 mt-3"),

    # Date range slider
    dbc.Card([
        dbc.CardBody([
            html.Label("Date Range Filter", className="fw-bold mb-2",
                       style={"fontSize": "0.85rem", "color": BRAND["primary"]}),
            dcc.RangeSlider(
                id="date-range-slider",
                min=min_month, max=max_month,
                value=[min_month, max_month],
                marks={
                    m: {"label": (date_origin + pd.DateOffset(months=m)).strftime("%Y"),
                        "style": {"fontSize": "0.7rem", "color": BRAND["ink_soft"]}}
                    for m in range(min_month, max_month + 1, 60)
                },
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ])
    ], className="mb-3 shadow-sm"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p1a-chart", figure=build_p1a(),
                          config={"displayModeBar": False}), width=12, md=6),
        dbc.Col(dcc.Graph(id="p1b-chart", figure=build_p1b(),
                          config={"displayModeBar": False}), width=12, md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p1c-chart", figure=build_p1c(),
                          config={"displayModeBar": False}), width=12, md=6),
        dbc.Col(dcc.Graph(id="p1d-chart", figure=build_p1d(),
                          config={"displayModeBar": False}), width=12, md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p9-chart", figure=build_p9(),
                          config={"displayModeBar": False}), width=12),
    ]),

    html.Div("SOUTHEAST STATE RANKINGS", className="kpi-label mb-2 mt-4"),
    html.P("Composite ranking across unemployment, participation, and payroll growth (1 = best).",
           style={"color": BRAND["ink_soft"], "fontSize": "0.88rem"}),
    dash_table.DataTable(
        id="table5-rankings",
        columns=[
            {"name": "State", "id": "label"},
            {"name": "UR (%)", "id": "ur", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "UR Rank", "id": "ur_rank", "type": "numeric"},
            {"name": "LFPR (%)", "id": "lfpr", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "LFPR Rank", "id": "lfpr_rank", "type": "numeric"},
            {"name": "Payroll YoY %", "id": "payroll_yoy_pct", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Payroll Rank", "id": "payroll_rank", "type": "numeric"},
            {"name": "Overall", "id": "overall_rank", "type": "numeric"},
        ],
        data=table5.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "10px 8px",
                    "fontSize": "13px", "fontFamily": "Inter, sans-serif",
                    "border": f"1px solid {BRAND['border']}"},
        style_header={"backgroundColor": BRAND["primary"], "color": "white",
                      "fontWeight": 600, "textAlign": "center",
                      "fontSize": "12px", "letterSpacing": "0.04em",
                      "textTransform": "uppercase"},
        style_data_conditional=[
            {"if": {"filter_query": "{overall_rank} = 1"},
             "backgroundColor": "#E8F5EE", "fontWeight": 600},
            {"if": {"filter_query": "{label} = Georgia"},
             "backgroundColor": "#FFF4E8", "fontWeight": 600},
        ],
        page_size=10,
    ),
], fluid=True, className="mt-3 mb-4", style={"maxWidth": "1400px"})


# %% [7F] Tab 2 — Demographics
tab2_content = dbc.Container([
    html.Div("LABOR FORCE COMPOSITION", className="kpi-label mb-3 mt-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p2-chart", figure=build_p2(),
                          config={"displayModeBar": False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p3-chart", figure=build_p3(),
                          config={"displayModeBar": False}), width=12),
    ]),

    dbc.Card([
        dbc.CardBody([
            html.Label("Demographic group", className="fw-bold mb-2",
                       style={"fontSize": "0.85rem", "color": BRAND["primary"]}),
            dcc.Dropdown(
                id="demographic-dropdown",
                options=[
                    {"label": "Sex / Overall", "value": "Sex/Overall"},
                    {"label": "Race / Ethnicity", "value": "Race/Ethnicity"},
                    {"label": "Education", "value": "Education"},
                ],
                value="Sex/Overall",
                clearable=False,
                style={"width": "320px"},
            ),
        ])
    ], className="mb-2 shadow-sm"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p4-chart", figure=build_p4(),
                          config={"displayModeBar": False}), width=12),
    ]),
], fluid=True, className="mt-3 mb-4", style={"maxWidth": "1400px"})


# %% [7G] Tab 3 — Economic Development
tab3_content = dbc.Container([
    html.Div("GEORGIA ECONOMIC DEVELOPMENT", className="kpi-label mb-3 mt-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p5a-chart", figure=build_p5a(),
                          config={"displayModeBar": False}), width=12, md=6),
        dbc.Col(dcc.Graph(id="p5b-chart", figure=build_p5b(),
                          config={"displayModeBar": False}), width=12, md=6),
    ]),

    dbc.Card([
        dbc.CardBody([
            html.Label("Industry chart view", className="fw-bold mb-2",
                       style={"fontSize": "0.85rem", "color": BRAND["primary"]}),
            dbc.RadioItems(
                id="industry-chart-type",
                options=[
                    {"label": "Stacked area", "value": "area"},
                    {"label": "Line chart", "value": "line"},
                    {"label": "Bar chart (latest)", "value": "bar"},
                ],
                value="area",
                inline=True,
                inputClassName="me-1",
                labelClassName="me-3",
            ),
        ])
    ], className="mb-2 mt-2 shadow-sm"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p6-chart", figure=build_p6(),
                          config={"displayModeBar": False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p5c-chart", figure=build_p5c(),
                          config={"displayModeBar": False}), width=12),
    ]),

    html.Div(f"GA INDUSTRY EMPLOYMENT — AS OF {latest_ind_date.date()}",
             className="kpi-label mb-2 mt-4"),
    dash_table.DataTable(
        id="table4-industry",
        columns=[
            {"name": "Industry", "id": "label"},
            {"name": "Employment (K)", "id": "value", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "Share (%)", "id": "share", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "YoY Change (K)", "id": "yoy_chg", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
            {"name": "YoY Growth (%)", "id": "yoy_pct", "type": "numeric",
             "format": dash_table.Format.Format(precision=1, scheme=dash_table.Format.Scheme.fixed)},
        ],
        data=table4.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "10px 8px",
                    "fontSize": "13px", "fontFamily": "Inter, sans-serif",
                    "border": f"1px solid {BRAND['border']}"},
        style_header={"backgroundColor": BRAND["primary"], "color": "white",
                      "fontWeight": 600, "fontSize": "12px",
                      "letterSpacing": "0.04em", "textTransform": "uppercase"},
        style_data_conditional=[
            {"if": {"filter_query": "{yoy_pct} > 0", "column_id": "yoy_pct"},
             "color": BRAND["good"], "fontWeight": 600},
            {"if": {"filter_query": "{yoy_pct} < 0", "column_id": "yoy_pct"},
             "color": BRAND["bad"], "fontWeight": 600},
        ],
        page_size=10,
    ),
], fluid=True, className="mt-3 mb-4", style={"maxWidth": "1400px"})


# %% [7H] Tab 4 — National Labor Market
tab4_content = dbc.Container([
    html.Div("NATIONAL LABOR MARKET DYNAMICS", className="kpi-label mb-3 mt-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p7-chart", figure=build_p7(),
                          config={"displayModeBar": False}), width=12, lg=7),
        dbc.Col(dcc.Graph(id="tightness-chart", figure=build_tightness_ts(),
                          config={"displayModeBar": False}), width=12, lg=5),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p8a-chart", figure=build_p8a(),
                          config={"displayModeBar": False}), width=12, md=6),
        dbc.Col(dcc.Graph(id="p8b-chart", figure=build_p8b(),
                          config={"displayModeBar": False}), width=12, md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p10-chart", figure=build_p10(),
                          config={"displayModeBar": False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="p12-chart", figure=build_p12(),
                          config={"displayModeBar": False}), width=12),
    ]),
], fluid=True, className="mt-3 mb-4", style={"maxWidth": "1400px"})


# %% [7I] Tab 5 — Econometric Models
if m1 is not None:
    ols_html = html.Div([
        html.Div("MODEL 1 — OLS: GA UR DETERMINANTS (HC3 ROBUST SE)",
                 className="kpi-label mb-2"),
        html.P([
            html.B("Specification: "),
            html.Code("GA_UR ~ US_UR + JOR + AHE_YoY + Trend",
                      style={"backgroundColor": "#F1F3F5", "padding": "2px 6px",
                              "borderRadius": "3px"}),
        ], style={"fontSize": "0.9rem"}),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in m1_table.columns],
            data=m1_table.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "10px 8px",
                        "fontSize": "13px", "fontFamily": "Inter, sans-serif",
                        "border": f"1px solid {BRAND['border']}"},
            style_header={"backgroundColor": BRAND["primary"], "color": "white",
                          "fontWeight": 600, "fontSize": "12px",
                          "letterSpacing": "0.04em", "textTransform": "uppercase"},
            style_data_conditional=[
                {"if": {"filter_query": '{Sig} contains "***"'},
                 "backgroundColor": "#E8F5EE", "fontWeight": 600},
                {"if": {"filter_query": '{Sig} contains "**"'},
                 "backgroundColor": "#F0F8F3"},
            ],
        ),
        html.Div([
            html.Span(f"R² = {m1.rsquared:.4f}", className="me-4"),
            html.Span(f"Adj R² = {m1.rsquared_adj:.4f}", className="me-4"),
            html.Span(f"N = {int(m1.nobs)}", className="me-4"),
            html.Span(f"Sample: {df_m1['date'].min().date()} to {df_m1['date'].max().date()}"),
        ], className="mt-2", style={"fontSize": "0.85rem", "color": BRAND["ink_soft"]}),
    ])
else:
    ols_html = html.P("OLS model unavailable (insufficient data).")

tab5_content = dbc.Container([
    html.Div("ECONOMETRIC MODELS & RESEARCH", className="kpi-label mb-3 mt-3"),

    dbc.Card(dbc.CardBody(ols_html), className="shadow-sm mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="m2-stl-chart", figure=build_m2_stl(),
                          config={"displayModeBar": False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="m3-ar1-chart", figure=build_m3_ar1(),
                          config={"displayModeBar": False}), width=12),
    ]),

    html.Div(f"MODEL 4 — SHIFT-SHARE DECOMPOSITION ({ss_start.date()} → {ss_end.date()})",
             className="kpi-label mb-2 mt-4"),
    html.P(f"National total employment growth over period: {nat_total_growth * 100:.2f}%.  "
           f"Competitive Effect = state-specific outperformance net of national and industry mix effects.",
           style={"color": BRAND["ink_soft"], "fontSize": "0.88rem"}),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in ss_table.columns],
        data=ss_table.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "10px 8px",
                    "fontSize": "13px", "fontFamily": "Inter, sans-serif",
                    "border": f"1px solid {BRAND['border']}"},
        style_header={"backgroundColor": BRAND["primary"], "color": "white",
                      "fontWeight": 600, "fontSize": "12px",
                      "letterSpacing": "0.04em", "textTransform": "uppercase"},
        style_data_conditional=[
            {"if": {"filter_query": "{Competitive Effect} > 0", "column_id": "Competitive Effect"},
             "color": BRAND["good"], "fontWeight": 600},
            {"if": {"filter_query": "{Competitive Effect} < 0", "column_id": "Competitive Effect"},
             "color": BRAND["bad"], "fontWeight": 600},
        ],
    ),

    dbc.Row([
        dbc.Col(dcc.Graph(id="p11-chart", figure=build_p11(),
                          config={"displayModeBar": False}), width=12),
    ], className="mt-4"),

    html.Div("WELCH T-TEST — GA UR vs US UR BY BUSINESS CYCLE",
             className="kpi-label mb-2 mt-4"),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df_welch.columns],
        data=df_welch.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "10px 8px",
                    "fontSize": "13px", "fontFamily": "Inter, sans-serif",
                    "border": f"1px solid {BRAND['border']}"},
        style_header={"backgroundColor": BRAND["primary"], "color": "white",
                      "fontWeight": 600, "fontSize": "12px",
                      "letterSpacing": "0.04em", "textTransform": "uppercase"},
        style_data_conditional=[
            {"if": {"filter_query": '{Sig} contains "***"'},
             "backgroundColor": "#E8F5EE", "fontWeight": 600},
        ],
    ),
], fluid=True, className="mt-3 mb-4", style={"maxWidth": "1400px"})


# %% [7J] Full App Layout
app.layout = html.Div([
    navbar,
    dbc.Container(
        dbc.Tabs([
            dbc.Tab(tab0_content, label="Executive Summary", tab_id="tab-0"),
            dbc.Tab(tab1_content, label="State Overview", tab_id="tab-1"),
            dbc.Tab(tab2_content, label="Labor Force Demographics", tab_id="tab-2"),
            dbc.Tab(tab3_content, label="Georgia Economic Development", tab_id="tab-3"),
            dbc.Tab(tab4_content, label="National Labor Market", tab_id="tab-4"),
            dbc.Tab(tab5_content, label="Econometric Models", tab_id="tab-5"),
        ], id="tabs", active_tab="tab-0", className="mt-2"),
        fluid=True,
        style={"maxWidth": "1400px"},
    ),

    # Footer
    html.Footer([
        html.Hr(style={"borderColor": BRAND["border"], "marginTop": "2rem"}),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div("GEORGIA LABOR MARKET RESEARCH DASHBOARD",
                             style={"fontSize": "0.7rem", "fontWeight": 600,
                                    "letterSpacing": "0.12em",
                                    "color": BRAND["primary"]}),
                    html.Div(f"{SCRIPT_VERSION}  ·  John Brassard  ·  "
                             f"{pd.Timestamp.today().strftime('%B %Y')}",
                             style={"fontSize": "0.78rem",
                                    "color": BRAND["ink_soft"]}),
                ], width=12, md=6),
                dbc.Col([
                    html.Div("DATA SOURCES",
                             style={"fontSize": "0.7rem", "fontWeight": 600,
                                    "letterSpacing": "0.12em",
                                    "color": BRAND["primary"]}),
                    html.Div("BLS LAUS · FRED · BLS CPS · BLS JOLTS · BLS CES · BEA",
                             style={"fontSize": "0.78rem",
                                    "color": BRAND["ink_soft"]}),
                ], width=12, md=6,
                   style={"textAlign": "right"}),
            ], className="py-3"),
        ], fluid=True, style={"maxWidth": "1400px"}),
    ], style={"backgroundColor": "white", "marginTop": "2rem"}),
], style={"backgroundColor": BRAND["bg_soft"], "minHeight": "100vh"})


# %% [8] Callbacks
# =============================================================================
# CALLBACKS
# =============================================================================

# %% [8A] Callback — Shared Date Range Slider
@app.callback(
    [Output("p1a-chart", "figure"),
     Output("p1b-chart", "figure"),
     Output("p1c-chart", "figure"),
     Output("p1d-chart", "figure"),
     Output("p9-chart", "figure")],
    Input("date-range-slider", "value"),
)
def update_date_range(slider_val):
    start = date_origin + pd.DateOffset(months=int(slider_val[0]))
    end = date_origin + pd.DateOffset(months=int(slider_val[1]))
    return (
        build_p1a(date_min=start, date_max=end),
        build_p1b(date_min=start, date_max=end),
        build_p1c(date_min=start, date_max=end),
        build_p1d(date_min=start, date_max=end),
        build_p9(date_min=start, date_max=end),
    )


# %% [8B] Callback — Demographic Group Dropdown
@app.callback(
    Output("p4-chart", "figure"),
    Input("demographic-dropdown", "value"),
)
def update_p4(group_type):
    return build_p4(group_type)


# %% [8C] Callback — Industry Chart Type Toggle
@app.callback(
    Output("p6-chart", "figure"),
    Input("industry-chart-type", "value"),
)
def update_p6(chart_type):
    return build_p6(chart_type)


# %% [9] Run Dashboard
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Georgia Labor Market Research Dashboard")
    print("  Open browser to: http://localhost:8050")
    print("=" * 60 + "\n")
    # Use PORT env var if set (Render sets this); otherwise default to 8050
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, port=port, host="0.0.0.0")
