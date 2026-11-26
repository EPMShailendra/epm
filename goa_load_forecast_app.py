"""
Goa Day-Ahead Load Forecast Tool  -  streamlit run goa_load_forecast_app.py
Issued at 07:00 (or whenever your data ends) for the NEXT day, 96 x 15-min blocks.
Needs forecast_core.py in the same folder.
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecast_core as fc

st.set_page_config(page_title="Goa Day-Ahead Load Forecast", layout="wide")
BLOCKS = fc.BLOCKS


@st.cache_data(show_spinner="Reading workbook...")
def read_workbook(raw: bytes):
    load, rem = fc.load_goa_workbook(io.BytesIO(raw))
    return load, fc.classify_remarks(rem)


@st.cache_data(show_spinner="Reading file...")
def read_tidy(raw: bytes, name: str, ts_col: str, load_col: str):
    df = pd.read_csv(io.BytesIO(raw)) if name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    return fc.load_tidy(df, ts_col, load_col)


@st.cache_data(show_spinner="Fetching weather (Open-Meteo)...", ttl=3 * 3600)
def get_weather(lat, lon, start, end):
    return fc.fetch_weather(lat, lon, start, end)


@st.cache_data(show_spinner="Backtesting as if issued each morning...")
def run_backtest(load, flags, weather, t0, t1, k, downweight, holidays):
    return fc.backtest_issue(load, flags, weather, t0, t1, k, downweight, holidays)


@st.cache_data(show_spinner="Training on all data and forecasting...")
def run_forecast(load, flags, weather, holidays, downweight, rel_resid):
    return fc.forecast_issue(load, flags, weather, holidays, downweight, rel_resid)


def mu(s: pd.Series) -> float:
    return float(s.sum() * 0.25 / 1000)


st.title("Goa Day-Ahead Load Forecast")

with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Historical load file", type=["xlsx", "xls", "csv"],
                          help="The FY-sheet Goa workbook, or any CSV/Excel with a timestamp and a load column. "
                               "Fill the current day up to 06:45 before the 07:00 run.")
    st.header("2. Options")
    use_weather = st.checkbox("Use weather (Open-Meteo)", value=False,
                              help="Temperature, humidity and rain for the target day. Needs internet.")
    lat = st.number_input("Latitude", value=15.50, format="%.3f")
    lon = st.number_input("Longitude", value=73.83, format="%.3f")
    downweight = st.checkbox("Down-weight outage / rain-trip days in training", value=True)
    holiday_text = st.text_area("Extra holidays (YYYY-MM-DD, one per line)",
                                help="Moving festivals (Diwali, Eid, Ganesh Chaturthi...) for the target day and "
                                     "the day before it. Fixed-date holidays are added automatically.")
    bt_days = st.slider("Backtest window (days)", 14, 90, 30)

if up is None:
    st.info("Upload your Goa historical load workbook in the sidebar to begin.")
    st.stop()

raw = up.getvalue()
flags = None
try:
    if not up.name.lower().endswith((".xlsx", ".xls")):
        raise ValueError("tidy")
    load, flags = read_workbook(raw)
except Exception:
    head = pd.read_csv(io.BytesIO(raw), nrows=5) if up.name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw), nrows=5)
    cols = list(head.columns)
    with st.sidebar:
        st.caption("Not the FY-sheet layout - reading it as a plain table.")
        ts_col = st.selectbox("Timestamp column", cols)
        load_col = st.selectbox("Load column", cols, index=min(1, len(cols) - 1))
    load = read_tidy(raw, up.name, ts_col, load_col)

load = load.loc[:load.last_valid_index()]
issue_day, k, target_day = fc.detect_issue_state(load)
last_ts = load.index[-1]

extra_holidays = []
for line in holiday_text.splitlines():
    if line.strip():
        try:
            extra_holidays.append(pd.Timestamp(line.strip()).normalize())
        except Exception:
            st.sidebar.warning(f"Could not read date: {line}")
extra_holidays = tuple(extra_holidays)

now = pd.Timestamp.now()
if (now - last_ts) > pd.Timedelta(hours=36):
    st.warning(f"Data ends **{last_ts:%d %b %Y %H:%M}**, so this forecast is for **{target_day:%d %b %Y}**. "
               f"Add the latest days to forecast tomorrow.")
elif k == 0:
    st.info("Only complete days are in the file (no morning data for the issue day). The forecast still works "
            "but is slightly less accurate than with data up to 06:45.")
st.caption(f"Data known to **{last_ts:%d %b %H:%M}** ({k} blocks of {issue_day:%d %b} available) "
           f"-> forecast for **{target_day:%A %d %b %Y}**, 96 blocks.")

weather = None
if use_weather:
    try:
        weather = get_weather(lat, lon, load.index[0].date(), (target_day + pd.Timedelta(days=1)).date())
    except Exception as e:
        st.warning(f"Weather unavailable ({e}). Continuing without weather.")

full_days = load.resample("D").count()
last_full = full_days[full_days == BLOCKS].index[-1]
bt_end = last_full
bt_start = bt_end - pd.Timedelta(days=bt_days - 1)

actual, pred, naive = run_backtest(load, flags, weather, bt_start, bt_end, k, downweight, extra_holidays)
rel_resid = ((actual - pred) / pred).dropna()
fcst, info = run_forecast(load, flags, weather, extra_holidays, downweight, rel_resid)
m_model, m_naive = fc.metrics(actual, pred), fc.metrics(actual, naive.dropna())

c = st.columns(5)
c[0].metric("Forecast peak", f"{fcst.forecast_mw.max():.0f} MW", f"{fcst.forecast_mw.idxmax():%H:%M}")
c[1].metric("Forecast minimum", f"{fcst.forecast_mw.min():.0f} MW", f"{fcst.forecast_mw.idxmin():%H:%M}")
c[2].metric("Forecast energy", f"{mu(fcst.forecast_mw):.2f} MU")
c[3].metric(f"Backtest MAPE ({bt_days}d)", f"{m_model['MAPE %']:.1f}%",
            f"{m_model['MAPE %'] - m_naive['MAPE %']:+.1f} pts vs last week", delta_color="inverse")
c[4].metric("Daily peak error", f"{m_model['Daily peak error %']:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Backtest", "History explorer", "Data & holidays"])

with tab1:
    hist = load.iloc[-BLOCKS * 3 - k:]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist, name="Actual (known)", line=dict(color="#1f77b4", width=1.6))
    if "p90_mw" in fcst:
        fig.add_scatter(x=fcst.index, y=fcst.p90_mw, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fcst.index, y=fcst.p10_mw, fill="tonexty", fillcolor="rgba(255,127,14,0.20)",
                        line=dict(width=0), name="P10-P90 band", hoverinfo="skip")
    fig.add_scatter(x=fcst.index, y=fcst.forecast_mw, name="Forecast", line=dict(color="#ff7f0e", width=2))
    fig.add_vline(x=last_ts, line_dash="dot", line_color="gray")
    fig.update_layout(height=470, yaxis_title="MW", hovermode="x unified", legend=dict(orientation="h", y=1.08),
                      margin=dict(t=30))
    st.plotly_chart(fig)

    out = fcst.copy()
    out.insert(0, "block", np.arange(1, BLOCKS + 1))
    out.insert(1, "time", [f"{t:%H:%M}-{(t + fc.STEP):%H:%M}" for t in out.index])
    out.index.name = "timestamp"
    out = out.round(1)
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV", out.to_csv().encode(), f"goa_dayahead_{target_day:%Y-%m-%d}.csv", "text/csv")
    buf = io.BytesIO()
    out.to_excel(buf, sheet_name="forecast")
    d2.download_button("Download Excel", buf.getvalue(), f"goa_dayahead_{target_day:%Y-%m-%d}.xlsx")
    st.dataframe(out, height=320)

with tab2:
    st.caption(f"Each target day {bt_start:%d %b}-{bt_end:%d %b %Y} is forecast using only data that existed "
               f"at 07:00 the day before (model trained on earlier days only)."
               f"{' Weather is actual observed weather here, so real error will be a bit higher.' if weather is not None else ''}")
    st.dataframe(pd.DataFrame({"Model": m_model, "Last-week same block": m_naive}).round(2))
    fig2 = go.Figure()
    fig2.add_scatter(x=actual.index, y=actual, name="Actual", line=dict(width=1.3))
    fig2.add_scatter(x=pred.index, y=pred, name="Model", line=dict(width=1.3))
    fig2.update_layout(height=420, yaxis_title="MW", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2)
    a, b = st.columns(2)
    err = ((pred - actual).abs() / actual * 100).dropna()
    by_hour = err.groupby(err.index.hour).mean()
    f3 = go.Figure(go.Bar(x=by_hour.index, y=by_hour.values))
    f3.update_layout(title="MAPE by hour of day (%)", height=320, xaxis_title="Hour", margin=dict(t=40))
    a.plotly_chart(f3)
    by_day = err.resample("D").mean()
    f4 = go.Figure(go.Bar(x=by_day.index, y=by_day.values))
    f4.update_layout(title="MAPE by day (%) - spikes = holidays, rain, outages", height=320, margin=dict(t=40))
    b.plotly_chart(f4)

with tab3:
    daily = load.resample("D").agg(["mean", "max", "min"])
    daily.columns = ["avg", "peak", "min"]
    f5 = go.Figure()
    f5.add_scatter(x=daily.index, y=daily.peak, name="Daily peak", line=dict(width=1))
    f5.add_scatter(x=daily.index, y=daily.avg, name="Daily average", line=dict(width=1))
    f5.add_scatter(x=daily.index, y=daily.avg.rolling(30, min_periods=7).mean(), name="30-day avg",
                   line=dict(width=2.5, color="black"))
    f5.update_layout(height=380, yaxis_title="MW", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(f5)
    a, b = st.columns(2)
    yearly = daily.groupby(daily.index.year).agg(avg=("avg", "mean"), peak=("peak", "max"))
    yearly["energy (MU/day)"] = yearly["avg"] * 24 / 1000
    a.subheader("By calendar year")
    a.dataframe(yearly.round(1))
    mon = daily.avg.groupby([daily.index.year, daily.index.month]).mean().unstack(0).round(0)
    mon.index.name = "month"
    b.subheader("Monthly average MW")
    b.dataframe(mon)
    st.subheader("Typical daily profile")
    yr = st.selectbox("Year", sorted(set(load.index.year), reverse=True))
    sub = load[load.index.year == yr].dropna()
    prof = pd.DataFrame({"v": sub.values, "blk": sub.index.hour * 4 + sub.index.minute // 15,
                         "wk": np.where(sub.index.dayofweek >= 5, "Weekend", "Weekday")})
    pr = prof.groupby(["wk", "blk"]).v.mean().unstack(0)
    f6 = go.Figure()
    for col in pr.columns:
        f6.add_scatter(x=pr.index / 4, y=pr[col], name=col)
    f6.update_layout(height=340, xaxis_title="Hour of day", yaxis_title="MW", hovermode="x unified")
    st.plotly_chart(f6)

with tab4:
    qc = fc.qc_report(load.dropna())
    q = st.columns(4)
    q[0].metric("Days of data", qc["days"])
    q[1].metric("From", f"{qc['start']:%d %b %Y}")
    q[2].metric("To", f"{qc['end']:%d %b %Y}")
    q[3].metric("Missing blocks", qc["missing_blocks"])
    if qc["odd_days"]:
        st.write("Unusual days: " + ", ".join(str(d) for d in qc["odd_days"]))
    if flags is not None:
        st.write(f"Holidays in remarks: **{int(flags.is_holiday.sum())}** | "
                 f"outage / rain / shutdown days: **{int(flags.is_event.sum())}**")
        st.dataframe(flags.reset_index(), height=360)
    else:
        st.info("No remarks row in this file, so holidays are not learned from history.")
