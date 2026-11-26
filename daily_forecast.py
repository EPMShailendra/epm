"""
Run the Goa day-ahead forecast from the command line (for a 07:00 daily schedule).

    python daily_forecast.py --data "C:/data/Goa_Historical_data.xlsx" --out "C:/data/forecasts"
    python daily_forecast.py --data Goa_Historical_data.xlsx --out forecasts --weather --holidays 2026-11-09,2026-11-10

Before it runs, the workbook must contain the current day filled up to 06:45 (or complete days only).
Output: goa_dayahead_<date>.csv and .xlsx (96 blocks: forecast, P10, P90) plus goa_dayahead_latest.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import forecast_core as fc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to the Goa historical workbook")
    p.add_argument("--out", default="forecasts", help="output folder")
    p.add_argument("--weather", action="store_true", help="add Open-Meteo weather (needs internet)")
    p.add_argument("--lat", type=float, default=15.50)
    p.add_argument("--lon", type=float, default=73.83)
    p.add_argument("--holidays", default="", help="comma separated extra holiday dates YYYY-MM-DD")
    p.add_argument("--backtest-days", type=int, default=30)
    p.add_argument("--max-age-hours", type=float, default=36, help="refuse to run if data is older than this")
    a = p.parse_args()

    load, rem = fc.load_goa_workbook(a.data)
    flags = fc.classify_remarks(rem)
    load = load.loc[:load.last_valid_index()]
    issue_day, k, target = fc.detect_issue_state(load)
    age = pd.Timestamp.now() - load.index[-1]
    if age > pd.Timedelta(hours=a.max_age_hours):
        print(f"ERROR: data ends {load.index[-1]} ({age}) - update the workbook first.", file=sys.stderr)
        sys.exit(2)

    hols = tuple(pd.Timestamp(x.strip()) for x in a.holidays.split(",") if x.strip())
    weather = None
    if a.weather:
        try:
            weather = fc.fetch_weather(a.lat, a.lon, load.index[0].date(), (target + pd.Timedelta(days=1)).date())
        except Exception as e:
            print(f"WARNING: weather unavailable ({e}); continuing without it.", file=sys.stderr)

    counts = load.resample("D").count()
    bt_end = counts[counts == fc.BLOCKS].index[-1]
    bt_start = bt_end - pd.Timedelta(days=a.backtest_days - 1)
    act, pred, naive = fc.backtest_issue(load, flags, weather, bt_start, bt_end, k, True, hols)
    rel = ((act - pred) / pred).dropna()
    out, info = fc.forecast_issue(load, flags, weather, hols, True, rel)
    m = fc.metrics(act, pred)

    out = out.copy()
    out.insert(0, "block", range(1, fc.BLOCKS + 1))
    out.insert(1, "time", [f"{t:%H:%M}-{(t + fc.STEP):%H:%M}" for t in out.index])
    out.index.name = "timestamp"
    out = out.round(1)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"goa_dayahead_{target:%Y-%m-%d}"
    out.to_csv(f"{stem}.csv")
    out.to_excel(f"{stem}.xlsx", sheet_name="forecast")
    out.to_csv(folder / "goa_dayahead_latest.csv")

    f = out.forecast_mw
    print(f"Forecast for {target:%A %d %b %Y} (data to {load.index[-1]:%d %b %H:%M})")
    print(f"  peak {f.max():.0f} MW at {f.idxmax():%H:%M} | min {f.min():.0f} MW at {f.idxmin():%H:%M} | "
          f"energy {f.sum() * 0.25 / 1000:.2f} MU")
    print(f"  backtest {a.backtest_days}d: MAPE {m['MAPE %']:.2f}% | daily peak err {m['Daily peak error %']:.2f}%")
    print(f"  saved: {stem}.csv / .xlsx")


if __name__ == "__main__":
    main()
