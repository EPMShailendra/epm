"""
Core engine for the Goa load forecast tool (no Streamlit imports, so it can be tested alone).

Data format supported natively: the "Goa_Historical_data.xlsx" layout
  - one sheet per financial year (Apr-Mar)
  - row 1: remarks (holidays / events), row 2: weekday, row 3: dates (from column C)
  - rows 4-99: 96 blocks of 15 minutes ("00:00-00:15" ... "23:45-24:00"), load in MW
Any tidy CSV/Excel with a timestamp column and a load column is also supported.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BLOCKS = 96  # 15-min blocks per day
STEP = pd.Timedelta("15min")

HOLIDAY_WORDS = [
    "gudi padwa", "ambedkar", "good friday", "labour day", "id ul", "ramzan", "bakrid",
    "independence", "ganesh", "gandhi", "dusherra", "dussehra", "diwali", "sfx", "liberation",
    "x'mas", "xmas", "republic", "holi", "ram navami", "mahashivratri", "ayodha",
    "parliamentary election", "zilla panchayat",
]
EVENT_WORDS = ["shut", "rain", "tripping", "load loss", "thndrstrm", "thunder"]
WEATHER_EVENT_WORDS = ["hottest", "humid"]

# Fixed-date holidays (same date every year) used to pre-fill the forecast window
FIXED_HOLIDAYS = {(1, 26): "Republic Day", (5, 1): "Labour day", (8, 15): "Independence Day",
                  (10, 2): "Gandhi Jayanti", (12, 19): "Liberation Day", (12, 25): "X'Mas"}


# --------------------------------------------------------------------------- loading
def load_goa_workbook(src) -> tuple[pd.Series, pd.DataFrame]:
    """Parse the FY-sheet workbook -> (15-min load series in MW, daily remarks frame)."""
    from openpyxl import load_workbook

    wb = load_workbook(src, data_only=True, read_only=True)
    days: dict[pd.Timestamp, pd.Series] = {}
    remarks: dict[pd.Timestamp, str] = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 99:
            continue
        dates, rem, blocks = rows[2], rows[0], rows[3:3 + BLOCKS]
        for j in range(2, len(dates)):
            d = dates[j]
            if d is None or not hasattr(d, "year"):
                continue
            d = pd.Timestamp(d).normalize()
            vals = pd.to_numeric(pd.Series([b[j] if j < len(b) else None for b in blocks]), errors="coerce")
            if vals.notna().sum() == 0:
                continue
            days[d] = pd.Series(vals.values, index=pd.date_range(d, periods=BLOCKS, freq="15min"))  # later sheet wins
            r = rem[j] if j < len(rem) else None
            if r:
                remarks[d] = str(r)
    if not days:
        raise ValueError("No data found - is this the FY-sheet Goa workbook layout?")
    load = pd.concat([days[k] for k in sorted(days)]).astype(float)
    load.name = "load_mw"
    rem_df = pd.DataFrame({"remark": pd.Series(remarks)})
    rem_df.index.name = "date"
    return load, rem_df


def load_tidy(df: pd.DataFrame, ts_col: str, load_col: str) -> pd.Series:
    s = pd.Series(pd.to_numeric(df[load_col], errors="coerce").values,
                  index=pd.to_datetime(df[ts_col], errors="coerce"))
    s = s[s.index.notna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample("15min").mean().interpolate(limit=4)
    s.name = "load_mw"
    return s


def classify_remarks(rem: pd.DataFrame) -> pd.DataFrame:
    """Turn free-text remarks into flags: holiday / outage-or-rain event / weather note."""
    out = pd.DataFrame(index=rem.index)
    t = rem["remark"].str.lower()
    out["is_holiday"] = t.apply(lambda x: any(w in x for w in HOLIDAY_WORDS)).astype(int)
    out["is_event"] = t.apply(lambda x: any(w in x for w in EVENT_WORDS)).astype(int)
    out["remark"] = rem["remark"]
    return out


def qc_report(load: pd.Series) -> dict:
    full = pd.date_range(load.index[0], load.index[-1], freq="15min")
    daily = load.resample("D").mean()
    med = daily.rolling(15, center=True, min_periods=5).median()
    odd = daily[(daily < 0.8 * med) | (daily > 1.25 * med)]
    return {
        "start": load.index[0], "end": load.index[-1], "days": int(daily.notna().sum()),
        "missing_blocks": int(len(full) - load.reindex(full).notna().sum()),
        "min_mw": float(load.min()), "max_mw": float(load.max()),
        "odd_days": [d.date() for d in odd.index],
    }


# --------------------------------------------------------------------------- weather
def fetch_weather(lat: float, lon: float, start: date, end: date, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Hourly temp/humidity/rain from Open-Meteo (archive + forecast), returned at 15-min."""
    import requests

    today = pd.Timestamp.now().normalize().date()
    parts = []
    hourly = "temperature_2m,relative_humidity_2m,precipitation"
    arch_end = min(end, today - pd.Timedelta(days=2).to_pytimedelta())
    if start <= arch_end:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
            "latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(arch_end),
            "hourly": hourly, "timezone": tz})
        r.raise_for_status()
        parts.append(r.json()["hourly"])
    if end > arch_end:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
            "latitude": lat, "longitude": lon, "hourly": hourly, "timezone": tz,
            "past_days": 7, "forecast_days": 10})
        r.raise_for_status()
        parts.append(r.json()["hourly"])
    frames = []
    for h in parts:
        frames.append(pd.DataFrame({"temp": h["temperature_2m"], "humidity": h["relative_humidity_2m"],
                                    "rain": h["precipitation"]}, index=pd.to_datetime(h["time"])))
    w = pd.concat(frames)
    w = w[~w.index.duplicated(keep="first")].sort_index()
    w = w.resample("15min").interpolate(limit=3)
    return w


# --------------------------------------------------------------------------- features
def holiday_series(index: pd.DatetimeIndex, flags: pd.DataFrame | None, extra_dates=()) -> pd.Series:
    """Daily 0/1 holiday flag for every date in `index` (history flags + user dates + fixed dates)."""
    days = pd.DatetimeIndex(index.normalize().unique())
    h = pd.Series(0, index=days)
    if flags is not None and len(flags):
        known = flags["is_holiday"].reindex(days)
        h.loc[known.index[known == 1]] = 1
    for d in extra_dates:
        d = pd.Timestamp(d).normalize()
        if d in h.index:
            h[d] = 1
    # fixed-date holidays only where we have no remark info (i.e. the future)
    if flags is not None and len(flags):
        unknown = days[days > flags.index.max()]
    else:
        unknown = days
    for d in unknown:
        if (d.month, d.day) in FIXED_HOLIDAYS:
            h[d] = 1
    return h


def build_features(load: pd.Series, hol_daily: pd.Series, weather: pd.DataFrame | None,
                   day_ahead: bool):
    """`load` spans history + NaN future. Every feature uses only information that is known
    at forecast time for horizons <= 7 days (lags >= 7 days; lag 1 day only in day-ahead mode)."""
    idx = load.index
    f = pd.DataFrame(index=idx)
    mins = idx.hour * 60 + idx.minute
    f["block"] = (mins // 15).astype(int)
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    doy = idx.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)

    h = hol_daily.reindex(idx.normalize()).values
    f["is_holiday"] = h
    hd = hol_daily.copy()
    f["day_before_holiday"] = hd.shift(-1, fill_value=0).reindex(idx.normalize()).values
    f["day_after_holiday"] = hd.shift(1, fill_value=0).reindex(idx.normalize()).values

    # Level and same-block history (all >= 7 days old => valid for a 7-day horizon).
    # Load-derived features are expressed RELATIVE to a recent level so the model keeps working
    # as the system grows (trees cannot extrapolate absolute MW beyond what they saw in training).
    lag7 = load.shift(7 * BLOCKS)
    level = lag7.rolling(28 * BLOCKS, min_periods=7 * BLOCKS).mean()  # 28 days ending 7 days ago
    lags_wk = pd.concat([lag7, load.shift(14 * BLOCKS), load.shift(21 * BLOCKS), load.shift(28 * BLOCKS)], axis=1)
    if day_ahead:
        base = load.shift(BLOCKS).rolling(7 * BLOCKS, min_periods=BLOCKS).mean()  # last 7 days
        f["lag_1d"] = load.shift(BLOCKS) / base
        f["lag_2d"] = load.shift(2 * BLOCKS) / base
        f["level_ratio"] = base / level  # recent growth vs 4-week-old level
    else:
        base = level
    f["lag_7d"] = lag7 / base
    f["lag_14d"] = load.shift(14 * BLOCKS) / base
    f["lag_wk_mean"] = lags_wk.mean(axis=1) / base
    f["lag_wk_max"] = lags_wk.max(axis=1) / base

    if weather is not None:
        w = weather.reindex(idx)
        f["temp"] = w["temp"]
        f["humidity"] = w["humidity"]
        f["rain"] = w["rain"]
        f["temp_24h_mean"] = w["temp"].rolling(BLOCKS, min_periods=8).mean()
        f["heat_index"] = w["temp"] + 0.1 * w["humidity"]
    return f, base


def new_model(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=48,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def _fit(X: pd.DataFrame, y_ratio: pd.Series, w: pd.Series | None):
    m = new_model()
    m.fit(X, y_ratio, sample_weight=w)
    return m


def train_weights(index: pd.DatetimeIndex, flags: pd.DataFrame | None, downweight_events: bool) -> pd.Series:
    w = pd.Series(1.0, index=index)
    if downweight_events and flags is not None and len(flags):
        ev = flags["is_event"].reindex(index.normalize()).fillna(0).values
        w[ev == 1] = 0.2
    return w


# --------------------------------------------------------------------------- backtest
def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    a, p = actual.align(pred, join="inner")
    err = p - a
    daily_a, daily_p = a.resample("D").mean(), p.resample("D").mean()
    peak_a, peak_p = a.resample("D").max(), p.resample("D").max()
    return {
        "MAPE %": float((err.abs() / a).mean() * 100),
        "MAE (MW)": float(err.abs().mean()),
        "RMSE (MW)": float(np.sqrt((err ** 2).mean())),
        "Daily energy error %": float(((daily_p - daily_a).abs() / daily_a).mean() * 100),
        "Daily peak error %": float(((peak_p - peak_a).abs() / peak_a).mean() * 100),
        "Bias (MW)": float(err.mean()),
    }


def backtest(load: pd.Series, flags: pd.DataFrame | None, weather: pd.DataFrame | None,
             test_start: pd.Timestamp, test_end: pd.Timestamp, day_ahead: bool,
             downweight_events: bool = True, extra_holidays=()):
    """Train on everything before test_start, predict [test_start, test_end].
    Returns (actual, model_pred, naive_pred)."""
    hol = holiday_series(load.index, flags, extra_holidays)
    f, base = build_features(load, hol, weather, day_ahead)
    y = load / base
    w = train_weights(load.index, flags, downweight_events)
    tr = (load.index < test_start) & y.notna() & base.notna()
    te = (load.index >= test_start) & (load.index <= test_end + pd.Timedelta("23h45min")) & load.notna() & base.notna()
    m = _fit(f[tr], y[tr], w[tr])
    pred = pd.Series(m.predict(f[te]) * base[te], index=f.index[te])
    naive = load.shift(7 * BLOCKS)[te]
    return load[te], pred, naive


# --------------------------------------------------------------------------- forecast
def forecast(load: pd.Series, flags: pd.DataFrame | None, weather: pd.DataFrame | None,
             horizon_days: int, extra_holidays=(), downweight_events: bool = True,
             band_residuals: pd.Series | None = None):
    """Forecast the `horizon_days` days after the last timestamp in `load` (15-min blocks)."""
    day_ahead = horizon_days == 1
    last = load.index[-1]
    fut_idx = pd.date_range(last + STEP, periods=horizon_days * BLOCKS, freq="15min")
    full = pd.concat([load, pd.Series(np.nan, index=fut_idx)])
    full.name = "load_mw"
    hol = holiday_series(full.index, flags, extra_holidays)
    f, base = build_features(full, hol, weather, day_ahead)
    y = full / base
    w = train_weights(full.index, flags, downweight_events)
    tr = y.notna() & base.notna() & (full.index <= last)
    m = _fit(f[tr], y[tr], w[tr])
    pred = pd.Series(m.predict(f.loc[fut_idx]) * base.loc[fut_idx], index=fut_idx, name="forecast_mw")
    out = pd.DataFrame({"forecast_mw": pred})
    if band_residuals is not None and len(band_residuals) > 50:
        # multiplicative error band by block-of-day from the backtest
        rel = (band_residuals).dropna()
        blk = (rel.index.hour * 4 + rel.index.minute // 15)
        q10 = rel.groupby(blk).quantile(0.10)
        q90 = rel.groupby(blk).quantile(0.90)
        fb = fut_idx.hour * 4 + fut_idx.minute // 15
        out["p10_mw"] = pred.values * (1 + np.minimum(q10.reindex(fb).values, 0))
        out["p90_mw"] = pred.values * (1 + np.maximum(q90.reindex(fb).values, 0))
    return out


# =========================================================================== ISSUE-TIME DAY-AHEAD
# Day-ahead forecast issued at a fixed time (e.g. 07:00) for the NEXT calendar day.
# `k` = number of 15-min blocks of the issue day that are already known (07:00 -> k=28,
# only complete days known -> k=0). Every feature below uses data up to that cut-off only.

def build_features_issue(load: pd.Series, hol_daily: pd.Series, weather: pd.DataFrame | None, k: int):
    idx = load.index
    f = pd.DataFrame(index=idx)
    mins = idx.hour * 60 + idx.minute
    f["block"] = (mins // 15).astype(int)
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    doy = idx.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    dn = idx.normalize()
    f["is_holiday"] = hol_daily.reindex(dn).values
    f["day_before_holiday"] = hol_daily.shift(-1, fill_value=0).reindex(dn).values
    f["day_after_holiday"] = hol_daily.shift(1, fill_value=0).reindex(dn).values  # issue day is a holiday

    # last known timestamp for every target row: (target day start) - 24h + k blocks - 1 block
    origin = dn - pd.Timedelta(days=1) + k * STEP - STEP
    r = load.rolling(7 * BLOCKS, min_periods=BLOCKS).mean()
    base = pd.Series(r.reindex(origin).values, index=idx)
    base_prev = pd.Series(r.shift(7 * BLOCKS).reindex(origin).values, index=idx)
    f["level_ratio"] = base / base_prev  # recent growth/season drift

    if k > 0:  # morning of the issue day
        m = load.rolling(k, min_periods=k).mean()
        m_now = m.reindex(origin).values
        m_wk = m.shift(7 * BLOCKS).reindex(origin).values
        f["morning_vs_base"] = m_now / base.values
        f["morning_vs_lastweek"] = m_now / m_wk

    if k > 0:  # same block yesterday is only known for blocks that already happened today
        known_1d = (f["block"].values < k)
        f["lag_1d"] = (load.shift(BLOCKS) / base).where(known_1d)
    f["lag_2d"] = load.shift(2 * BLOCKS) / base
    f["lag_7d"] = load.shift(7 * BLOCKS) / base
    f["lag_14d"] = load.shift(14 * BLOCKS) / base
    lags_wk = pd.concat([load.shift(n * BLOCKS) for n in (7, 14, 21, 28)], axis=1)
    f["lag_wk_mean"] = lags_wk.mean(axis=1) / base
    f["lag_wk_max"] = lags_wk.max(axis=1) / base

    if weather is not None:
        w = weather.reindex(idx)
        for v in ("temp", "humidity", "rain", "wind", "temp_spread", "rain_spread", "wind_spread"):
            if v in w and w[v].notna().any():
                f[v] = w[v]
        if "temp" in f:
            f["temp_24h_mean"] = w["temp"].rolling(BLOCKS, min_periods=8).mean()
            f["temp_72h_mean"] = w["temp"].rolling(3 * BLOCKS, min_periods=24).mean()
            if "humidity" in f:
                f["heat_index"] = w["temp"] + 0.1 * w["humidity"]
        if "rain" in f:  # rain is mm/h; one block = 0.25 h
            f["rain_6h"] = w["rain"].rolling(24, min_periods=4).sum() / 4
            f["rain_24h"] = w["rain"].rolling(BLOCKS, min_periods=8).sum() / 4
    return f, base


def detect_issue_state(load: pd.Series):
    """From the last valid timestamp work out (issue_day, k, target_day)."""
    t0 = load.last_valid_index()
    nxt = t0 + STEP
    issue_day = nxt.normalize()
    k = int((nxt - issue_day) / STEP)  # 0 if t0 was 23:45
    return issue_day, k, issue_day + pd.Timedelta(days=1)


def backtest_issue(load: pd.Series, flags, weather, test_start, test_end, k: int,
                   downweight_events: bool = True, extra_holidays=()):
    """Train on target days before test_start; predict each target day in [test_start, test_end]
    exactly as it would have been issued the previous day with k blocks known."""
    load = load.dropna()
    hol = holiday_series(load.index, flags, extra_holidays)
    f, base = build_features_issue(load, hol, weather, k)
    y = load / base
    w = train_weights(load.index, flags, downweight_events)
    tr = (load.index < test_start) & y.notna()
    te = (load.index >= test_start) & (load.index < test_end + pd.Timedelta(days=1)) & y.notna()
    m = _fit(f[tr], y[tr], w[tr])
    pred = pd.Series(m.predict(f[te]) * base[te], index=f.index[te])
    naive = load.shift(7 * BLOCKS)[te]
    return load[te], pred, naive


def forecast_issue(load: pd.Series, flags, weather, extra_holidays=(), downweight_events: bool = True,
                   rel_resid: pd.Series | None = None):
    """Forecast the full next day (96 blocks) after the issue day implied by the data."""
    load = load.loc[:load.last_valid_index()]
    issue_day, k, target = detect_issue_state(load)
    fut_idx = pd.date_range(target, periods=BLOCKS, freq="15min")
    # the issue day's unknown remainder is NaN; target day is NaN
    gap_idx = pd.date_range(load.index[-1] + STEP, fut_idx[-1], freq="15min")
    full = pd.concat([load, pd.Series(np.nan, index=gap_idx)])
    hol = holiday_series(full.index, flags, extra_holidays)
    f, base = build_features_issue(full, hol, weather, k)
    y = full / base
    w = train_weights(full.index, flags, downweight_events)
    tr = y.notna()
    m = _fit(f[tr], y[tr], w[tr])
    pred = pd.Series(m.predict(f.loc[fut_idx]) * base.loc[fut_idx], index=fut_idx, name="forecast_mw")
    out = pd.DataFrame({"forecast_mw": pred})
    if rel_resid is not None and len(rel_resid) > 50:
        blk = rel_resid.index.hour * 4 + rel_resid.index.minute // 15
        q10 = rel_resid.groupby(blk).quantile(0.10)
        q90 = rel_resid.groupby(blk).quantile(0.90)
        fb = fut_idx.hour * 4 + fut_idx.minute // 15
        out["p10_mw"] = pred.values * (1 + np.minimum(q10.reindex(fb).values, 0))
        out["p90_mw"] = pred.values * (1 + np.maximum(q90.reindex(fb).values, 0))
    return out, {"issue_day": issue_day, "k": k, "target_day": target, "last_data": load.index[-1]}


# =========================================================================== WEATHER SOURCES
# Parameters used everywhere: temp (degC), humidity (%), rain (mm/h), wind (m/s, 10 m).
# Sources: ECMWF IFS via Open-Meteo (live API) and NCMRWF NCUM-G 12 km via a file you supply.
WX_VARS = ("temp", "humidity", "rain", "wind")
_OM_VARS = {"temperature_2m": "temp", "relative_humidity_2m": "humidity",
            "precipitation": "rain", "wind_speed_10m": "wind"}


def _om_to_df(h: dict) -> pd.DataFrame:
    df = pd.DataFrame(index=pd.to_datetime(h["time"]))
    for k, new in _OM_VARS.items():
        key = k if k in h else next((c for c in h if c.startswith(k)), None)  # multi-model keys carry a suffix
        df[new] = pd.to_numeric(pd.Series(h[key], index=df.index), errors="coerce") if key else np.nan
    return df


def fetch_ecmwf(lat: float, lon: float, start, end, model: str = "ecmwf_ifs025",
                tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """Hourly ECMWF IFS fields from Open-Meteo: archived forecasts for past days (no look-ahead from
    analysis), live forecast for today onwards. Open-Meteo is free for NON-COMMERCIAL use only."""
    import requests

    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    today = pd.Timestamp.now().normalize()
    hourly = ",".join(_OM_VARS)
    base = {"latitude": lat, "longitude": lon, "hourly": hourly, "models": model,
            "wind_speed_unit": "ms", "timezone": tz}
    parts = []
    hist_end = min(end, today - pd.Timedelta(days=1))
    cur = start
    while cur <= hist_end:
        stop = min(cur + pd.Timedelta(days=364), hist_end)
        r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", timeout=120,
                         params={**base, "start_date": cur.date().isoformat(), "end_date": stop.date().isoformat()})
        r.raise_for_status()
        parts.append(_om_to_df(r.json()["hourly"]))
        cur = stop + pd.Timedelta(days=1)
    if end >= today:
        fd = int(max(2, min(16, (end - today).days + 2)))
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60,
                         params={**base, "past_days": 1, "forecast_days": fd})
        r.raise_for_status()
        parts.append(_om_to_df(r.json()["hourly"]))
    if not parts:
        raise ValueError("no ECMWF data requested")
    w = pd.concat(parts)
    return w[~w.index.duplicated(keep="last")].sort_index()  # forecast API wins on overlap


def fetch_ecmwf_cached(cache_dir, lat, lon, start, end, model="ecmwf_ifs025", tz="Asia/Kolkata") -> pd.DataFrame:
    """Same as fetch_ecmwf but keeps settled (past) days in a CSV so each run only downloads the new days."""
    from pathlib import Path

    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"ecmwf_{model}_{lat:.2f}_{lon:.2f}.csv"
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    today = pd.Timestamp.now().normalize()
    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    if cached is not None and len(cached) and cached.index[0] <= start + pd.Timedelta(days=1):
        fetch_from = max(start, cached.index[-1].normalize() - pd.Timedelta(days=2))
    else:
        cached, fetch_from = None, start
    new = fetch_ecmwf(lat, lon, fetch_from, end, model, tz)
    settled = new[new.index < today]
    merged = pd.concat([cached, settled]) if cached is not None else settled
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(path)
    out = pd.concat([merged, new[new.index >= today]])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.loc[start:end + pd.Timedelta(hours=23)]


def load_weather_file(src, name: str = "weather.csv", assume_utc: bool = False, tz: str = "Asia/Kolkata"):
    """Read a point time series (e.g. NCMRWF NCUM-G 12 km extracted at Goa) from CSV/Excel.
    Needs a time column plus any of: temperature (degC or K), humidity (% or fraction),
    rain/precipitation (mm per hour), wind speed (m/s) or u10/v10 components.
    Returns (frame with temp/humidity/rain/wind, notes about unit conversions)."""
    df = pd.read_csv(src) if str(name).lower().endswith(".csv") else pd.read_excel(src)
    lc = {c: str(c).strip().lower() for c in df.columns}
    notes: list[str] = []

    tcol = next((c for c, l in lc.items() if l in ("time", "timestamp", "datetime", "date_time", "valid_time", "date")),
                df.columns[0])
    t = pd.to_datetime(df[tcol], errors="coerce")
    if assume_utc:
        t = t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
        notes.append("times converted from UTC to local time")
    out = pd.DataFrame(index=t)

    def pick(*keys, exact=()):
        for c, l in lc.items():
            if c == tcol:
                continue
            if l in exact or any(k in l for k in keys):
                return c
        return None

    c_t = pick("temp", "t2m", "2t_", "air_temperature", exact=("2t", "tas", "t_2m", "t"))
    c_h = pick("humid", exact=("rh", "rh2m", "r2"))
    c_r = pick("rain", "precip", exact=("tp", "apcp", "prcp"))
    c_w = pick("windspeed", "wind_speed", "wind speed", exact=("ws", "wspd", "wind", "si10", "w10"))
    c_u, c_v = pick(exact=("u10", "u10m", "10u", "u_10m")), pick(exact=("v10", "v10m", "10v", "v_10m"))

    def num(c):
        return pd.to_numeric(df[c], errors="coerce").values

    if c_t:
        v = num(c_t)
        if np.nanmedian(v) > 100:
            v = v - 273.15
            notes.append("temperature converted from Kelvin")
        out["temp"] = v
    if c_h:
        v = num(c_h)
        if np.nanmax(v) <= 1.5:
            v = v * 100
            notes.append("humidity converted from fraction to %")
        out["humidity"] = v
    if c_r:
        out["rain"] = num(c_r)
    if c_w:
        out["wind"] = num(c_w)
    elif c_u and c_v:
        out["wind"] = np.sqrt(num(c_u) ** 2 + num(c_v) ** 2)
        notes.append("wind speed computed from u10/v10")
    out = out[out.index.notna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    found = [v for v in WX_VARS if v in out]
    if not found:
        raise ValueError("No temperature / humidity / rain / wind columns recognised in the file")
    notes.append("columns found: " + ", ".join(found))
    return out, notes


def combine_weather(sources: dict, target_day=None):
    """Merge several sources ({'ecmwf': df, 'ncmrwf': df}) onto a 15-min grid.
    Returns (frame, notes). Frame has the consensus (mean) of temp/humidity/rain/wind, the spread
    between sources, and each source's own columns (suffix _<name>) for display only.
    A source with no data on the forecast day is dropped, so the forecast never runs on blanks."""
    notes, ready = [], {}
    for name, df in sources.items():
        if df is None or not len(df):
            continue
        if target_day is not None:
            td = pd.Timestamp(target_day).normalize()
            day = df.loc[td: td + pd.Timedelta(hours=23, minutes=59)]
            if day.dropna(how="all").empty:
                notes.append(f"{name}: no data for forecast day {td:%d %b} - source not used")
                continue
        g = df.resample("15min").mean().interpolate(limit=3)
        ready[name] = g
    if not ready:
        return None, notes
    idx = sorted(set().union(*[g.index for g in ready.values()]))
    idx = pd.DatetimeIndex(idx)
    out = pd.DataFrame(index=pd.date_range(idx.min(), idx.max(), freq="15min"))
    for v in WX_VARS:
        cols = []
        for name, g in ready.items():
            if v in g:
                out[f"{v}_{name}"] = g[v].reindex(out.index)
                cols.append(f"{v}_{name}")
        if cols:
            out[v] = out[cols].mean(axis=1)
            if len(cols) > 1 and v in ("temp", "rain", "wind"):
                out[f"{v}_spread"] = out[cols].max(axis=1) - out[cols].min(axis=1)
    notes.append("sources used: " + ", ".join(ready))
    return out, notes


def weather_coverage(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in w.columns:
        if c.endswith("_spread"):
            continue
        s = w[c].dropna()
        if len(s):
            rows.append({"series": c, "from": s.index[0], "to": s.index[-1], "share_filled_%": round(100 * len(s) / len(w), 1)})
    return pd.DataFrame(rows)
