{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import numpy as np\
import pandas as pd\
import streamlit as st\
import matplotlib.pyplot as plt\
from pandas_datareader.data import DataReader\
\
st.set_page_config(page_title="Mortgage Spread Dashboard", layout="wide")\
\
# ----------------------------\
# Helpers\
# ----------------------------\
def fred(series_id: str, start: str) -> pd.Series:\
    api_key = os.getenv("FRED_API_KEY")\
    if not api_key:\
        st.error("Missing FRED_API_KEY. Add it in Streamlit Cloud \uc0\u8594  App settings \u8594  Secrets.")\
        st.stop()\
    s = DataReader(series_id, "fred", start=start, api_key=api_key)[series_id]\
    s.name = series_id\
    return s\
\
def percentile_rank(series: pd.Series, value: float) -> float:\
    s = series.dropna().values\
    if len(s) == 0:\
        return np.nan\
    return float((s < value).mean() * 100.0)\
\
def zscore(series: pd.Series, value: float) -> float:\
    s = series.dropna()\
    if len(s) < 2:\
        return np.nan\
    return float((value - s.mean()) / s.std(ddof=0))\
\
def format_pp(x):\
    return "\'97" if pd.isna(x) else f"\{x:.2f\} pp"\
\
# ----------------------------\
# UI\
# ----------------------------\
st.title("Mortgage Spread: 30Y Fixed Mortgage \'96 10Y Treasury (with Recession Overlays)")\
\
with st.sidebar:\
    st.header("Settings")\
    start_date = st.date_input("Start date", value=pd.Timestamp("1990-01-01")).strftime("%Y-%m-%d")\
    roll_weeks = st.slider("Rolling average (weeks)", min_value=1, max_value=52, value=13)\
    show_raw = st.checkbox("Show raw spread line", value=True)\
    show_roll = st.checkbox("Show rolling average", value=True)\
    regime_threshold_wide = st.slider("\'93Wide\'94 regime threshold (pp)", 0.5, 4.0, 2.0, 0.1)\
    regime_threshold_tight = st.slider("\'93Tight\'94 regime threshold (pp)", -2.0, 2.0, 0.5, 0.1)\
\
# ----------------------------\
# Data\
# ----------------------------\
# FRED series IDs:\
# MORTGAGE30US = 30-Year Fixed Rate Mortgage Average in the United States (weekly, %)\
# DGS10        = 10-Year Treasury Constant Maturity Rate (daily, %)\
# USREC        = NBER based Recession Indicators for the United States from the Peak through the Trough (monthly, 0/1)\
mort = fred("MORTGAGE30US", start_date)\
dgs10 = fred("DGS10", start_date)\
usrec = fred("USREC", start_date)\
\
df = pd.concat([mort, dgs10, usrec], axis=1).sort_index()\
df.columns = ["Mortgage30Y", "Treasury10Y", "USREC"]\
\
# Align frequencies\
df["Mortgage30Y"] = df["Mortgage30Y"].ffill()  # weekly -> daily forward-fill\
df["USREC"] = df["USREC"].ffill()              # monthly -> daily forward-fill\
df["Spread"] = df["Mortgage30Y"] - df["Treasury10Y"]\
\
# Rolling average (weeks -> approx days)\
window_days = int(roll_weeks * 7)\
df["Spread_Roll"] = df["Spread"].rolling(window_days, min_periods=max(7, window_days // 4)).mean()\
\
# Latest stats\
latest = df["Spread"].dropna().iloc[-1] if df["Spread"].dropna().shape[0] else np.nan\
latest_date = df["Spread"].dropna().index[-1] if df["Spread"].dropna().shape[0] else None\
\
hist = df["Spread"].dropna()\
mean_spread = hist.mean() if len(hist) else np.nan\
median_spread = hist.median() if len(hist) else np.nan\
pct = percentile_rank(hist, latest) if not pd.isna(latest) else np.nan\
z = zscore(hist, latest) if not pd.isna(latest) else np.nan\
\
# Regime\
def regime(val: float) -> str:\
    if pd.isna(val):\
        return "\'97"\
    if val >= regime_threshold_wide:\
        return "Wide"\
    if val <= regime_threshold_tight:\
        return "Tight"\
    return "Normal"\
\
regime_now = regime(latest)\
\
# ----------------------------\
# Top metrics\
# ----------------------------\
c1, c2, c3, c4, c5 = st.columns(5)\
c1.metric("Latest spread", format_pp(latest), help="Mortgage30Y \uc0\u8722  Treasury10Y")\
c2.metric("As-of date", "\'97" if latest_date is None else latest_date.strftime("%Y-%m-%d"))\
c3.metric("Percentile (history)", "\'97" if pd.isna(pct) else f"\{pct:.0f\}th")\
c4.metric("Z-score (history)", "\'97" if pd.isna(z) else f"\{z:.2f\}")\
c5.metric("Regime", regime_now, help="Based on thresholds set in the sidebar")\
\
st.caption("Data source: FRED series MORTGAGE30US, DGS10, USREC. Spread in percentage points (pp).")\
\
# ----------------------------\
# Plot with recession shading\
# ----------------------------\
fig = plt.figure()\
ax = plt.gca()\
\
# Recession shading (USREC == 1)\
# We find contiguous recession intervals and shade them.\
rec = (df["USREC"] == 1).astype(int)\
rec_changes = rec.diff().fillna(0)\
\
start_idxs = df.index[rec_changes == 1].tolist()\
end_idxs = df.index[rec_changes == -1].tolist()\
\
# Handle recession already in progress at start or end\
if len(df) and rec.iloc[0] == 1:\
    start_idxs = [df.index[0]] + start_idxs\
if len(start_idxs) > len(end_idxs):\
    end_idxs = end_idxs + [df.index[-1]]\
\
for s, e in zip(start_idxs, end_idxs):\
    ax.axvspan(s, e, alpha=0.2)\
\
if show_raw:\
    ax.plot(df.index, df["Spread"], label="Spread (daily aligned)")\
if show_roll:\
    ax.plot(df.index, df["Spread_Roll"], label=f"Rolling avg (~\{roll_weeks\} weeks)")\
\
ax.axhline(mean_spread, linestyle="--", linewidth=1, label="Historical mean")\
ax.axhline(median_spread, linestyle=":", linewidth=1, label="Historical median")\
\
ax.set_title("Mortgage Spread with NBER Recession Shading")\
ax.set_xlabel("Date")\
ax.set_ylabel("Spread (pp)")\
ax.grid(True)\
ax.legend()\
plt.tight_layout()\
\
st.pyplot(fig)\
\
# ----------------------------\
# Extra analytics\
# ----------------------------\
st.subheader("Analytics")\
\
colA, colB = st.columns(2)\
\
with colA:\
    st.write("**Summary stats (since start date)**")\
    stats = pd.Series(\{\
        "Mean (pp)": mean_spread,\
        "Median (pp)": median_spread,\
        "Min (pp)": hist.min() if len(hist) else np.nan,\
        "Max (pp)": hist.max() if len(hist) else np.nan,\
        "Std dev (pp)": hist.std(ddof=0) if len(hist) else np.nan,\
        "Latest (pp)": latest,\
        "Percentile": pct,\
        "Z-score": z\
    \})\
    st.dataframe(stats.to_frame("Value").style.format("\{:.3f\}"))\
\
with colB:\
    st.write("**Recent readings**")\
    st.dataframe(df[["Mortgage30Y", "Treasury10Y", "Spread", "Spread_Roll", "USREC"]].tail(30).style.format(\{\
        "Mortgage30Y": "\{:.3f\}",\
        "Treasury10Y": "\{:.3f\}",\
        "Spread": "\{:.3f\}",\
        "Spread_Roll": "\{:.3f\}",\
        "USREC": "\{:.0f\}",\
    \}))\
\
# Download\
st.subheader("Download")\
csv = df[["Mortgage30Y", "Treasury10Y", "Spread", "Spread_Roll", "USREC"]].dropna().to_csv(index=True).encode("utf-8")\
st.download_button("Download CSV", csv, file_name="mortgage_spread.csv", mime="text/csv")}