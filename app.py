"""AURUM - Delhi Parking Occupancy Forecasting (Streamlit dashboard).

Dashboard for the AURUM project report (HCLTech BTech Embedded Systems & IoT).
It covers the report's six tabs: Forecast, Week heatmap, Live, Map, Camera, Model.

IMPORTANT - what the data is
* The report's model is trained on 30 car parks in Birmingham, UK (Oct-Dec 2016).
  Delhi places only *borrow* a similar lot's demand pattern and use indicative capacities.
* The ESP32 sensor nodes are a proposed design; no hardware readings exist yet.
* Without data/occupancy_birmingham.csv this app runs in DEMO mode: the daily curves
  below are hand-made archetype patterns calibrated to the report's Monday 2 Nov 2026
  example, NOT the trained gradient-boosting model. The Model tab can reproduce the
  report's evaluation as soon as the Birmingham CSV is placed in data/.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

IST = ZoneInfo("Asia/Kolkata")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AURUM_DATA_DIR", BASE_DIR / "data"))
OCC_CSV = DATA_DIR / "occupancy_birmingham.csv"
BAY_XML = DATA_DIR / "bay_annotations.xml"
DEFAULT_DB = os.getenv("AURUM_DB_PATH", str(BASE_DIR / "aurum.db"))

# ---- Values stated in the project report -------------------------------------------
WINDOW_START_MIN = 7 * 60            # 07:00
WINDOW_END_MIN = 17 * 60 + 30        # 17:30
SLOT_MINUTES = 30
N_SLOTS = 22                         # 07:00 ... 17:30 inclusive
SLOT_LABELS = [f"{(WINDOW_START_MIN + i * SLOT_MINUTES) // 60:02d}:"
               f"{(WINDOW_START_MIN + i * SLOT_MINUTES) % 60:02d}" for i in range(N_SLOTS)]
MAX_RATIO = 0.99                     # forecasts are capped at 99 % of capacity
SIGMA = 0.072                        # residual sigma, fraction of capacity
Z80 = 1.28                           # ~80 % range = +/- 1.28 sigma
MULTIPLIERS = {"rain": 1.15, "holiday": 0.86, "event": 1.30}
MAX_PAYLOAD_BYTES = 2048
MAX_OVER_CAPACITY = 1.10             # occupied must be 0-110 % of capacity

REPORT_METRICS = {
    "R2 (hold-out)": "0.84",
    "MAE": "7.6 pp of capacity",
    "Residual sigma": "0.072 of capacity",
    "Baseline: global mean": "MAE 24.1 pp",
    "Baseline: per-lot / weekday / hour average": "MAE 7.6 pp, R2 0.85",
    "Train rows (first 80% of days)": "28,586",
    "Test rows (last 20% of days)": "7,119",
}


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    capacity: int            # indicative capacity from the report
    lat: float               # approximate coordinates (display only)
    lon: float
    archetype: str           # demo-mode pattern family (assumption, see module docstring)
    monday_peak: float       # demo-mode peak occupancy ratio on a Monday


# Capacities are from the report. Place ids other than "igi" and the archetype assignment
# are placeholders - align them with the project's places.py. Monday peaks for the five
# places in the report's example (vehicles / capacity) are exact; the rest are assumed.
PLACES = [
    Place("cp", "Connaught Place Inner Circle", 1200, 28.6315, 77.2167, "office", 1076 / 1200),
    Place("citywalk", "Select Citywalk", 1500, 28.5286, 77.2194, "shopping", 1373 / 1500),
    Place("promenade", "DLF Promenade", 1800, 28.5415, 77.1560, "shopping", 0.88),
    Place("khan_market", "Khan Market", 600, 28.6005, 77.2270, "shopping", 0.85),
    Place("india_gate", "India Gate Lawns", 900, 28.6129, 77.2295, "tourist", 0.55),
    Place("nehru_place", "Nehru Place Business Hub", 1000, 28.5491, 77.2518, "office", 907 / 1000),
    Place("ndls", "New Delhi Railway Station", 1100, 28.6429, 77.2197, "transport", 0.80),
    Place("igi", "IGI Airport Terminal 3", 3500, 28.5562, 77.0999, "transport", 2873 / 3500),
    Place("bharat_mandapam", "Bharat Mandapam", 2500, 28.6176, 77.2434, "office", 0.60),
    Place("hauz_khas", "Hauz Khas Village", 450, 28.5535, 77.1947, "shopping", 0.70),
    Place("qutub", "Qutub Minar", 500, 28.5245, 77.1855, "tourist", 237 / 500),
    Place("red_fort", "Red Fort · Chandni Chowk", 700, 28.6562, 77.2410, "tourist", 0.60),
]
PLACE_BY_ID = {p.id: p for p in PLACES}

# Weekday factors (Mon..Sun) per archetype; Monday = 1 so monday_peak is the Monday peak.
DAY_FACTORS = {
    "office": [1.0, 1.0, 1.0, 1.0, 0.97, 0.45, 0.35],
    "shopping": [1.0, 1.0, 1.0, 1.0, 1.04, 1.10, 1.10],
    "tourist": [1.0, 1.0, 1.0, 1.0, 1.05, 1.12, 1.12],
    "transport": [1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.92],
}


# ---- Forecast engine (demo patterns) -------------------------------------------------
def _raw_shape(archetype: str, hour: np.ndarray) -> np.ndarray:
    if archetype == "office":
        g = np.exp(-0.5 * ((hour - 11.5) / 2.6) ** 2)
    elif archetype == "shopping":
        g = np.exp(-0.5 * ((hour - 15.0) / 2.4) ** 2)
    elif archetype == "tourist":
        g = np.exp(-0.5 * ((hour - 12.5) / 2.8) ** 2)
    else:  # transport
        g = 0.7 + 0.3 * np.exp(-0.5 * ((hour - 9.0) / 2.5) ** 2)
    return 0.35 + 0.65 * g


SLOT_HOURS = np.array([(WINDOW_START_MIN + i * SLOT_MINUTES) / 60 for i in range(N_SLOTS)])


def slot_index(minutes_since_midnight: int) -> tuple[int, bool]:
    """Return (slot 0-21, in_window). Times outside 07:00-17:30 hold the nearest slot."""
    in_window = WINDOW_START_MIN <= minutes_since_midnight <= WINDOW_END_MIN
    idx = (minutes_since_midnight - WINDOW_START_MIN) // SLOT_MINUTES
    return int(min(max(idx, 0), N_SLOTS - 1)), in_window


def base_ratio(place: Place, weekday: int, slot: int) -> float:
    shape = _raw_shape(place.archetype, SLOT_HOURS)
    shape = shape / shape.max()
    return float(place.monday_peak * DAY_FACTORS[place.archetype][weekday] * shape[slot])


def apply_conditions(ratio: float, rain: bool, holiday: bool, event: bool) -> float:
    if rain:
        ratio *= MULTIPLIERS["rain"]
    if holiday:
        ratio *= MULTIPLIERS["holiday"]
    if event:
        ratio *= MULTIPLIERS["event"]
    return float(min(max(ratio, 0.0), MAX_RATIO))


def forecast_slot(place_id: str, weekday: int, slot: int, rain=False, holiday=False, event=False) -> dict:
    if place_id not in PLACE_BY_ID:
        raise KeyError(f"Unknown place id: {place_id}")
    place = PLACE_BY_ID[place_id]
    ratio = apply_conditions(base_ratio(place, weekday, slot), rain, holiday, event)
    lo = min(max(ratio - Z80 * SIGMA, 0.0), MAX_RATIO)
    hi = min(max(ratio + Z80 * SIGMA, 0.0), MAX_RATIO)
    return {"ratio": ratio, "low": lo, "high": hi, "vehicles": round(ratio * place.capacity),
            "capacity": place.capacity}


def forecast_at(place_id: str, when: datetime, **conditions) -> dict:
    slot, in_window = slot_index(when.hour * 60 + when.minute)
    out = forecast_slot(place_id, when.weekday(), slot, **conditions)
    out["in_window"] = in_window
    out["slot"] = slot
    return out


def day_curve(place_id: str, weekday: int, **conditions) -> pd.DataFrame:
    rows = [forecast_slot(place_id, weekday, s, **conditions) for s in range(N_SLOTS)]
    df = pd.DataFrame(rows)
    df.insert(0, "slot", SLOT_LABELS)
    return df


def week_matrix(place_id: str, **conditions) -> np.ndarray:
    return np.array([[forecast_slot(place_id, d, s, **conditions)["ratio"] for s in range(N_SLOTS)]
                     for d in range(7)])


# ---- MQTT message contract (FR-1, FR-2) ----------------------------------------------
TOPIC_RE = re.compile(r"^aurum/parking/([A-Za-z0-9_\-]+)/occupancy$")


def validate_message(topic: str, payload: str | bytes) -> tuple[bool, str, dict | None]:
    """Validate a reading as the ingest service does. Returns (ok, reason, record)."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > MAX_PAYLOAD_BYTES:
        return False, f"payload larger than {MAX_PAYLOAD_BYTES} bytes", None
    m = TOPIC_RE.match(topic)
    if not m:
        return False, "topic must be aurum/parking/<place_id>/occupancy", None
    place_id = m.group(1)
    if place_id not in PLACE_BY_ID:
        return False, f"unknown place_id '{place_id}'", None
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return False, "payload is not valid JSON", None
    if not isinstance(body, dict):
        return False, "JSON payload must be an object", None
    if body.get("place_id") != place_id:
        return False, "place_id in payload must match the topic", None
    occupied = body.get("occupied")
    if isinstance(occupied, bool) or not isinstance(occupied, (int, float)):
        return False, "'occupied' must be a number", None
    capacity = body.get("capacity", PLACE_BY_ID[place_id].capacity)
    if isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or capacity <= 0:
        return False, "'capacity' must be a positive number", None
    if occupied < 0 or occupied > MAX_OVER_CAPACITY * capacity:
        return False, "'occupied' must be between 0 and 110% of capacity", None
    ts_raw = body.get("ts")
    if ts_raw is None:
        ts = datetime.now(timezone.utc)
    else:
        try:
            ts = datetime.fromisoformat(str(ts_raw))
        except ValueError:
            return False, "'ts' is not an ISO-8601 timestamp", None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)   # naive timestamps are assumed UTC
    return True, "valid", {"place_id": place_id, "occupied": float(occupied),
                           "capacity": float(capacity), "ts": ts.isoformat()}


# ---- Telemetry database (written by aurum.ingest / aurum.simulator) -------------------
def load_readings(db_path: str) -> tuple[pd.DataFrame, str]:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(), f"No database found at {path}."
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
            df = pd.read_sql_query("SELECT * FROM readings", con)
    except Exception as exc:  # noqa: BLE001 - show any DB problem to the user
        return pd.DataFrame(), f"Could not read the 'readings' table: {exc}"
    missing = {"place_id", "ts", "occupied"} - set(df.columns)
    if missing:
        return pd.DataFrame(), f"'readings' table lacks columns: {', '.join(sorted(missing))}"
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(IST)
    df = df.dropna(subset=["ts"])
    cap = df["capacity"] if "capacity" in df.columns else pd.Series(np.nan, index=df.index)
    df["capacity"] = cap.fillna(df["place_id"].map(lambda p: PLACE_BY_ID[p].capacity
                                                   if p in PLACE_BY_ID else np.nan))
    df["ratio"] = df["occupied"] / df["capacity"]
    return df.sort_values("ts"), ""


# ---- Camera annotations (CVAT polygons) ----------------------------------------------
def summarise_annotations(xml_path: Path) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    rows = []
    for img in root.iter("image"):
        labels = [p.get("label", "").lower() for p in img.findall("polygon")]
        rows.append({"frame": img.get("name", ""),
                     "free": sum("free" in x or "empty" in x for x in labels),
                     "occupied": sum("occupied" in x for x in labels),
                     "partial": sum("partial" in x for x in labels),
                     "bays": len(labels)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["occupancy_pct"] = 100 * df["occupied"] / df["bays"].replace(0, np.nan)
    return df


# ---- Model evaluation on the Birmingham data (reproduces report section 8) ------------
@st.cache_data(show_spinner="Training on Birmingham data ...")
def evaluate_birmingham(csv_path: str, mtime: float) -> dict:
    raw = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in raw.columns}
    need = ["systemcodenumber", "capacity", "occupancy", "lastupdated"]
    if any(n not in lower for n in need):
        raise ValueError("CSV needs columns SystemCodeNumber, Capacity, Occupancy, LastUpdated")
    df = raw.rename(columns={lower[n]: n for n in need})[need].copy()
    df["ts"] = pd.to_datetime(df["lastupdated"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df[(df["capacity"] > 0) & (df["occupancy"] >= 0) & (df["occupancy"] <= df["capacity"])]
    df["ratio"] = df["occupancy"] / df["capacity"]
    df["lot"] = df["systemcodenumber"].astype("category").cat.codes
    df["slot"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) // SLOT_MINUTES
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour
    df["date"] = df["ts"].dt.normalize()
    days = np.sort(df["date"].unique())
    cut = days[int(len(days) * 0.8)]                      # time split, no shuffling
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    feats = ["slot", "dow", "lot"]
    model = HistGradientBoostingRegressor(categorical_features=[False, False, True], random_state=42)
    model.fit(train[feats], train["ratio"])
    pred = np.clip(model.predict(test[feats]), 0, 1)
    resid = test["ratio"] - pred
    glob = float(train["ratio"].mean())
    keys = ["lot", "dow", "hour"]
    lookup = train.groupby(keys)["ratio"].mean().rename("avg")
    base = test.join(lookup, on=keys)["avg"].fillna(glob)
    return {
        "rows_total": len(df), "rows_train": len(train), "rows_test": len(test),
        "lots": int(df["lot"].nunique()), "first_day": str(days[0])[:10], "last_day": str(days[-1])[:10],
        "split_day": str(cut)[:10],
        "r2": float(r2_score(test["ratio"], pred)),
        "mae_pp": float(100 * mean_absolute_error(test["ratio"], pred)),
        "sigma": float(resid.std()),
        "global_mae_pp": float(100 * mean_absolute_error(test["ratio"], np.full(len(test), glob))),
        "lookup_mae_pp": float(100 * mean_absolute_error(test["ratio"], base)),
        "lookup_r2": float(r2_score(test["ratio"], base)),
    }


# ---- Streamlit UI ---------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def curve_figure(curve: pd.DataFrame, title: str, readings: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["slot"], y=100 * curve["high"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve["slot"], y=100 * curve["low"], mode="lines", line=dict(width=0),
                             fill="tonexty", name="~80% range", fillcolor="rgba(120,120,120,0.25)"))
    fig.add_trace(go.Scatter(x=curve["slot"], y=100 * curve["ratio"], mode="lines+markers", name="Forecast"))
    if readings is not None and not readings.empty:
        fig.add_trace(go.Scatter(x=readings["ts"].dt.strftime("%H:%M"), y=100 * readings["ratio"],
                                 mode="markers", name="Live reading", marker=dict(size=9, symbol="diamond")))
    fig.update_layout(title=title, yaxis_title="Occupancy (% of capacity)", xaxis_title="Half-hour slot (IST)",
                      yaxis_range=[0, 105], margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
    return fig


def main() -> None:
    st.set_page_config(page_title="AURUM - Delhi Parking Occupancy Forecasting", layout="wide")
    st.title("AURUM — Delhi Parking Occupancy Forecasting")
    st.caption("MQTT-connected parking sensor network with a machine-learning occupancy forecaster "
               "for 12 Delhi landmarks · HCLTech BTech Embedded Systems & IoT project")

    have_csv = OCC_CSV.exists()
    st.info(
        "**Prototype - not a validated Delhi forecast.** The model data comes from 30 car parks in "
        "Birmingham, UK (Oct-Dec 2016); Delhi places borrow a similar lot's pattern and use indicative "
        "capacities. The ESP32 sensor nodes are a proposed design, so no hardware readings are shown. "
        "Forecast curves on this page are demo archetype patterns, not the trained model."
    )

    # Sidebar controls
    st.sidebar.header("Forecast inputs")
    place_id = st.sidebar.selectbox("Place", [p.id for p in PLACES], format_func=lambda i: PLACE_BY_ID[i].name)
    now = datetime.now(IST)
    day = st.sidebar.date_input("Date", value=now.date())
    default_slot = slot_index(now.hour * 60 + now.minute)[0]
    hh0, mm0 = divmod(WINDOW_START_MIN + default_slot * SLOT_MINUTES, 60)
    chosen_time = st.sidebar.time_input("Time (IST, forecast window 07:00-17:30)",
                                        value=datetime(2000, 1, 1, hh0, mm0).time(), step=SLOT_MINUTES * 60)
    st.sidebar.markdown("**Conditions**")
    rain = st.sidebar.checkbox(f"Rain (×{MULTIPLIERS['rain']})")
    holiday = st.sidebar.checkbox(f"Public holiday (×{MULTIPLIERS['holiday']})")
    event = st.sidebar.checkbox(f"Exam / peak-event day (×{MULTIPLIERS['event']})")
    cond = dict(rain=rain, holiday=holiday, event=event)
    db_path = st.sidebar.text_input("Telemetry SQLite file", value=DEFAULT_DB)
    st.sidebar.markdown("### About")
    st.sidebar.write("Model (report): HistGradientBoostingRegressor on time of day, weekday and car-park "
                     "identity; forecasts are capped at 99% of capacity.")
    st.sidebar.write(f"Mode: {'Birmingham CSV found' if have_csv else 'DEMO (no Birmingham CSV)'}")

    place = PLACE_BY_ID[place_id]
    when = datetime(day.year, day.month, day.day, chosen_time.hour, chosen_time.minute)
    fc = forecast_at(place_id, when, **cond)
    slot_label = SLOT_LABELS[fc["slot"]]

    tab_fc, tab_week, tab_live, tab_map, tab_cam, tab_model = st.tabs(
        ["Forecast", "Week heatmap", "Live", "Map", "Camera", "Model"])

    # --- Forecast
    with tab_fc:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast occupancy", pct(fc["ratio"]))
        c2.metric("Expected vehicles", f"{fc['vehicles']:,} / {fc['capacity']:,}")
        c3.metric("~80% range", f"{pct(fc['low'])} – {pct(fc['high'])}")
        c4.metric("Free bays (approx.)", f"{fc['capacity'] - fc['vehicles']:,}")
        st.caption(f"{place.name} · {day:%A %d %b %Y} · {slot_label} IST")
        if not fc["in_window"]:
            st.warning(f"{chosen_time:%H:%M} is outside the 07:00-17:30 forecast window. The nearest "
                       f"in-window slot ({slot_label}) is shown and flagged as a hold, not a forecast.")

        readings, _ = load_readings(db_path)
        overlay = None
        if not readings.empty:
            overlay = readings[(readings["place_id"] == place_id) & (readings["ts"].dt.date == day)]
        curve = day_curve(place_id, day.weekday(), **cond)
        st.plotly_chart(curve_figure(curve, f"{place.name}: forecast for {day:%A %d %b}", overlay))
        st.caption("The ~80% range is ±1.28σ of hold-out residuals (σ = 0.072 of capacity); "
                   "it is not a calibrated interval.")

        st.subheader(f"Peak half-hour by place · {day:%A %d %b %Y}")
        peaks = []
        for p in PLACES:
            c = day_curve(p.id, day.weekday(), **cond)
            i = int(c["ratio"].idxmax())
            peaks.append({"Place": p.name, "Capacity": p.capacity, "Peak slot": c.loc[i, "slot"],
                          "Peak occupancy": pct(c.loc[i, "ratio"]), "Vehicles": int(c.loc[i, "vehicles"]),
                          "_ratio": c.loc[i, "ratio"]})
        peak_df = pd.DataFrame(peaks).sort_values("_ratio", ascending=False).drop(columns="_ratio")
        st.dataframe(peak_df, hide_index=True)

    # --- Week heatmap
    with tab_week:
        z = 100 * week_matrix(place_id, **cond)
        fig = go.Figure(go.Heatmap(z=z, x=SLOT_LABELS, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                   colorscale="YlOrRd", zmin=0, zmax=100, colorbar=dict(title="%")))
        fig.update_layout(title=f"Weekly forecast heatmap · {place.name}", margin=dict(l=10, r=10, t=50, b=10),
                          yaxis_autorange="reversed")
        st.plotly_chart(fig)

    # --- Live
    with tab_live:
        st.subheader("Live readings")
        readings, msg = load_readings(db_path)
        if readings.empty:
            st.info((msg or "The readings table is empty.") +
                    " Start the simulator to create demo telemetry: "
                    "`python -m aurum.simulator --offline --backfill 3` (from the full AURUM package), "
                    "or point the sidebar to your ingest database. No data is invented here.")
        else:
            latest = readings.sort_values("ts").groupby("place_id").tail(1).copy()
            latest["Place"] = latest["place_id"].map(lambda i: PLACE_BY_ID[i].name if i in PLACE_BY_ID else i)
            latest["Occupancy"] = latest["ratio"].map(pct)
            st.dataframe(latest[["Place", "ts", "occupied", "capacity", "Occupancy"]]
                         .rename(columns={"ts": "Latest reading", "occupied": "Occupied"}), hide_index=True)
            sel = readings[readings["place_id"] == place_id].tail(500)
            if sel.empty:
                st.write("No readings for the selected place yet.")
            else:
                st.line_chart(sel.set_index("ts")["ratio"].mul(100).rename("Occupancy (%)"))

        st.subheader("MQTT message validator")
        st.caption("Checks a reading the way the ingest service does: topic, JSON shape, 0-110% range, "
                   "≤ 2048 bytes. Invalid messages are dropped with a reason and never stored.")
        topic = st.text_input("Topic", value=f"aurum/parking/{place_id}/occupancy")
        payload = st.text_area("Payload (JSON)", value=json.dumps(
            {"place_id": place_id, "occupied": int(0.5 * place.capacity), "capacity": place.capacity,
             "ts": datetime.now(IST).replace(microsecond=0).isoformat()}, indent=2), height=150)
        ok, reason, record = validate_message(topic, payload)
        if ok:
            st.success("Accepted")
            st.json(record)
        else:
            st.error(f"Rejected: {reason}")

    # --- Map
    with tab_map:
        st.write(f"Capacity-weighted view of all 12 places · forecast for {day:%a %d %b}, {slot_label} IST")
        rows = []
        for p in PLACES:
            f = forecast_at(p.id, when, **cond)
            rows.append({"Place": p.name, "lat": p.lat, "lon": p.lon, "size": p.capacity,
                         "Forecast occupancy": pct(f["ratio"]), "Capacity": p.capacity})
        map_df = pd.DataFrame(rows)
        st.map(map_df, latitude="lat", longitude="lon", size="size")
        st.dataframe(map_df[["Place", "Capacity", "Forecast occupancy"]], hide_index=True)
        st.caption("Coordinates are approximate and for display only.")

    # --- Camera
    with tab_cam:
        st.subheader("Camera frames (CVAT bay annotations)")
        st.caption("Display only - not used for training. Report figures: 30 annotated frames, "
                   "903 labelled bays, mean occupancy 69%.")
        if BAY_XML.exists():
            try:
                cam = summarise_annotations(BAY_XML)
                a, b, c = st.columns(3)
                a.metric("Frames", len(cam))
                b.metric("Labelled bays", int(cam["bays"].sum()))
                c.metric("Mean occupancy", f"{cam['occupancy_pct'].mean():.0f}%")
                st.bar_chart(cam.set_index("frame")["occupancy_pct"].rename("Occupancy (%)"))
                st.dataframe(cam, hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse {BAY_XML.name}: {exc}")
        else:
            st.info(f"Place `bay_annotations.xml` in `{DATA_DIR}` to compute these from the annotations.")

    # --- Model
    with tab_model:
        st.subheader("Model and results (project report, section 8)")
        st.write("A HistGradientBoostingRegressor predicts the occupancy ratio (occupied ÷ capacity) from "
                 "time of day, weekday and car-park identity. The first 80% of days train the model, the "
                 "last 20% test it (no shuffling); the final model is refit on all data.")
        st.table(pd.DataFrame({"Metric": list(REPORT_METRICS), "Reported value": list(REPORT_METRICS.values())}))
        st.write("**Honest reading:** far better than a constant guess (MAE 7.6 vs 24.1 pp), but about the "
                 "same as a per-lot / weekday / hour average, because the occupancy pattern is very regular. "
                 "Its value is a compact, smooth predictor that can be retrained on real Delhi data.")
        st.write(f"**Condition multipliers** (from dwell-time ratios in 10,000 parking sessions, not separately "
                 f"validated against occupancy): rain ×{MULTIPLIERS['rain']}, public holiday "
                 f"×{MULTIPLIERS['holiday']}, exam / peak-event day ×{MULTIPLIERS['event']}.")
        st.markdown("**Reproduce the evaluation**")
        if have_csv:
            try:
                r = evaluate_birmingham(str(OCC_CSV), OCC_CSV.stat().st_mtime)
                st.table(pd.DataFrame({
                    "Metric": ["Rows (clean)", "Train / test rows", "Car parks", "Split date",
                               "R2", "MAE (pp)", "Residual sigma", "Global-mean MAE (pp)",
                               "Lookup MAE (pp) / R2"],
                    "Computed now": [f"{r['rows_total']:,}", f"{r['rows_train']:,} / {r['rows_test']:,}",
                                     r["lots"], r["split_day"], f"{r['r2']:.2f}", f"{r['mae_pp']:.1f}",
                                     f"{r['sigma']:.3f}", f"{r['global_mae_pp']:.1f}",
                                     f"{r['lookup_mae_pp']:.1f} / {r['lookup_r2']:.2f}"],
                }))
                st.caption("Hyper-parameters and cleaning rules here are a best reconstruction; small differences "
                           "from the reported numbers are expected until they are aligned with aurum/model.py.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not evaluate {OCC_CSV.name}: {exc}")
        else:
            st.info(f"Place `occupancy_birmingham.csv` (columns SystemCodeNumber, Capacity, Occupancy, "
                    f"LastUpdated) in `{DATA_DIR}` to recompute the metrics.")
        st.markdown("**Limitations:** different city and year (Birmingham 2016); indicative Delhi capacities and "
                    "place-to-lot mapping; only 07:00-17:30 is forecast (other times hold the nearest slot and "
                    "are flagged); 11 weeks of history, so no seasonality; ESP32 node not built or field-tested.")

    st.caption("Safety: software demonstration only - it does not control any hardware or power equipment.")


if __name__ == "__main__":
    main()
