# AURUM — Delhi Parking Occupancy Forecasting

An MQTT-connected parking sensor network with a machine-learning occupancy forecaster, live dashboards and cloud-ready deployment for **12 Delhi landmarks** (airport, malls, markets, monuments, railway station).

HCLTech · BTech Learner Programme · Embedded Systems & IoT project (v1.0, November 2026)

| | |
|---|---|
| Author | Satyam Gupta (individual project) |
| Programme | BTech · Electrical and Electronics Engineering |
| Institution | ABES Engineering College |
| Mentor | Mr. Puneet |
| Demo video | https://share.zight.com/01a19792-a053-792a-ad5e-f6f39d39340f |
| Source code | _GitHub link to be added_ |

> **Scope note — please read.** The training occupancy data (35,717 raw readings, 35,705 valid) comes from **30 car parks in Birmingham, UK** (4 Oct – 19 Dec 2016). Each Delhi place borrows the demand pattern of a similar lot and uses an *indicative* capacity. The Delhi ESP32 sensor hardware is a **proposed design**; in this submission the sensor side is exercised by a software simulator. Forecasts are a working prototype, **not a validated Delhi forecast**.

## The problem

Drivers in Delhi cannot tell in advance whether a car park at a destination will be full. AURUM collects bay occupancy over MQTT, stores it, visualises it live, and forecasts occupancy for any place, weekday and half-hour slot between **07:00 and 17:30** (22 slots).

Success metrics (from the report): hold-out R² ≥ 0.80 and MAE ≤ 10 percentage points (achieved **R² 0.84, MAE 7.6 pp**); invalid MQTT messages rejected and never stored; all 12 places × 22 slots forecast within 0–99 % of capacity; full pipeline starts with one command (`docker compose up`).

## What is in this repository

- `app.py` — Streamlit dashboard with six tabs: **Forecast**, **Week heatmap**, **Live**, **Map**, **Camera**, **Model**.
- `requirements.txt` — Python dependencies.
- `README.md` — this file.

The complete project described in the report (the `aurum/` package, tests, Docker Compose stack, Grafana provisioning, CI/CD and cloud scripts, `docs/`) is documented in the project report, sections 7 and 10.

### How `app.py` behaves

| Situation | What you see |
|---|---|
| No data files (default) | **Demo mode.** Daily curves are hand-made archetype patterns, calibrated to the report's Monday 2 Nov 2026 example. They are *not* the trained model. |
| `data/occupancy_birmingham.csv` present | The **Model** tab retrains a gradient-boosting model on a time split and shows the computed R², MAE, residual σ and baselines next to the reported values. |
| `data/bay_annotations.xml` present | The **Camera** tab summarises free / occupied / partial bays per frame. |
| A telemetry SQLite file with a `readings` table (path set in the sidebar, default `aurum.db`) | The **Live** tab lists the latest reading per place, and readings are overlaid on the forecast curve. |

The **Live** tab also contains an MQTT message validator that applies the same rules as the ingest service (topic, JSON shape, 0–110 % of capacity, ≤ 2048 bytes). Environment variables `AURUM_DATA_DIR` and `AURUM_DB_PATH` override the default locations.

## Quick start

Requires Python 3.10 or newer (the report uses 3.12).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

Full package and stack (as in the report, from the complete repository):

```bash
pip install -r requirements-dev.txt
python -m aurum.simulator --offline --backfill 3
streamlit run streamlit_app.py

# full stack: Mosquitto, PostgreSQL, InfluxDB, Grafana, ingest, dashboard, simulator
cp .env.example .env && docker compose --profile demo up -d --build
```

## System design

1. **Sensors** (proposed ESP32 nodes, or `aurum.simulator`) publish JSON to an MQTT broker (HiveMQ Cloud / EMQX Cloud on port 8883, or local Mosquitto).
2. **`aurum.ingest`** subscribes, validates each message and writes to SQLite (default) or PostgreSQL and, when configured, InfluxDB.
3. **Grafana** reads InfluxDB and PostgreSQL for live panels.
4. **Streamlit** loads the trained model, forecasts any place / date / time and overlays live readings.

### MQTT message contract

```
Topic  : aurum/parking/<place_id>/occupancy
Payload: {"place_id": "igi", "occupied": 1840, "capacity": 3500,
          "ts": "2026-11-02T13:00:00+05:30"}
```

`capacity` and `ts` are optional (defaults: catalogue capacity, receive time). `occupied` must be between 0 and 110 % of capacity. Invalid messages are dropped with a logged reason.

### Technology stack

| Layer | Choice |
|---|---|
| Edge firmware (proposed) | ESP32, C/C++ or MicroPython, Wi-Fi, MQTT over TLS |
| Broker | HiveMQ Cloud or EMQX Cloud (8883); Mosquitto locally |
| Back end | Python 3.12, paho-mqtt 2.x, SQLite / PostgreSQL, InfluxDB 2.7 |
| ML | scikit-learn `HistGradientBoostingRegressor`, Pandas, NumPy, joblib |
| Dashboards | Streamlit + Plotly; Grafana |
| DevOps | Git, Docker Compose, GitHub Actions, Azure and AWS scripts |

## Forecasting model and results

A `HistGradientBoostingRegressor` predicts the occupancy ratio (occupied ÷ capacity) from time of day, weekday and car-park identity. The first 80 % of days (28,586 rows, to 5 Dec 2016) train the model; the last 20 % (7,119 rows) test it, with no shuffling. The final model is refit on all data.

| Metric (hold-out) | Value |
|---|---|
| R² | 0.84 |
| Mean absolute error | 7.6 percentage points of capacity |
| Residual σ (used for the ~80 % range, ±1.28σ) | 0.072 of capacity |
| Baseline: global mean | MAE 24.1 pp |
| Baseline: per-lot / weekday / hour average | MAE 7.6 pp, R² 0.85 |

**Honest reading:** the model is far better than a constant guess, but on this data it performs about the same as a simple lookup of the average per lot, weekday and hour. Its value is a compact, smooth predictor that can be retrained once real Delhi data arrives. The ~80 % range is not a calibrated interval.

**Condition effects** (dwell-time ratios from 10,000 parking sessions; not separately validated against occupancy): rain ×1.15, public holiday ×0.86, exam / peak-event day ×1.30. Forecasts are capped at 99 % of capacity, and times outside 07:00–17:30 are flagged.

Example forecast, Monday 2 Nov 2026 (peak half-hour):

| Place | Capacity | Peak occupancy | Vehicles |
|---|---|---|---|
| Select Citywalk | 1,500 | 92 % | 1,373 |
| Nehru Place Business Hub | 1,000 | 91 % | 907 |
| Connaught Place Inner Circle | 1,200 | 90 % | 1,076 |
| IGI Airport Terminal 3 | 3,500 | 82 % | 2,873 |
| Qutub Minar | 500 | 47 % | 237 |

## Places covered (indicative capacities)

Connaught Place Inner Circle 1,200 · Select Citywalk 1,500 · DLF Promenade 1,800 · Khan Market 600 · India Gate Lawns 900 · Nehru Place Business Hub 1,000 · New Delhi Railway Station 1,100 · IGI Airport Terminal 3 3,500 · Bharat Mandapam 2,500 · Hauz Khas Village 450 · Qutub Minar 500 · Red Fort · Chandni Chowk 700.

## Datasets

| File | Content | Use |
|---|---|---|
| `occupancy_birmingham.csv` | 35,705 valid readings, 30 car parks, 4 Oct – 19 Dec 2016 | Train and test the model |
| `parking_sessions.csv` | 10,000 sessions with weather, holiday, exam-day and dwell time | Condition multipliers only |
| `bay_annotations.xml` | 30 camera frames, 903 labelled bays (CVAT polygons) | Camera tab (display, not training) |

## Proposed edge node (not yet built)

One ESP32 monitors two bays with HC-SR04 ultrasonic sensors, a 0.96" SSD1306 OLED and two status LEDs.

| Signal | Pin | Note |
|---|---|---|
| HC-SR04 #1 TRIG / ECHO | GPIO5 / GPIO18 | ECHO via 1 kΩ / 2 kΩ divider (5 V → 3.3 V) |
| HC-SR04 #2 TRIG / ECHO | GPIO19 / GPIO23 | ECHO via 1 kΩ / 2 kΩ divider |
| OLED SDA / SCL | GPIO21 / GPIO22 | I²C, address 0x3C, 400 kHz |
| Green LED (bay free) | GPIO26 | 220 Ω series |
| Red LED (bay full) | GPIO27 | 220 Ω series |

Estimated bill of materials: ₹1,270 (ESP32 DevKit v1 ₹450, 2× HC-SR04 ₹180, OLED ₹180, LED kit ₹40, resistors ₹20, 5 V 2 A supply ₹250, jumper wires + breadboard ₹150). Prices are indicative; verify before purchase.

Firmware design: median of 5 ultrasonic readings per bay; green LED when free bays remain, red when full; publish every 10 s with QoS 1; reconnect with back-off on Wi-Fi or broker loss; Wi-Fi and broker secrets loaded from a config file excluded from Git.

## Testing

The report documents 26 pytest tests (all passing on Python 3.11) covering the model, places/config, MQTT validation, the database layer, ingest, InfluxDB fail-safe behaviour, the simulator and camera annotations. Reproduce with `pytest --cov=aurum` in the full repository; linting uses `ruff`.

Field tests still to perform on hardware: power-cycle recovery and Wi-Fi reconnect; ultrasonic noise at different distances and angles (calibrate the occupied threshold per bay); message loss and duplicates at QoS 1.

## Deploy

- **Streamlit Community Cloud:** push this repository to GitHub, choose **Create app**, select the repository and branch, set `app.py` as the main file, and deploy.
- **Full stack:** Docker Compose, GitHub Actions CI (lint, test, Docker build, image publish) and Azure / AWS deploy scripts (`infra/azure/deploy.sh`, `infra/aws/deploy.sh`) are described in the report, section 10.

## Security

- No hard-coded credentials; `.env.example` uses `change-me` placeholders and real values belong in `.env` (git-ignored).
- MQTT payloads are validated and SQL is parameterised.
- The bundled Mosquitto config is anonymous and for **development only**; use authenticated HiveMQ / EMQX credentials in production.

## Limitations and future work

- Birmingham 2016 data stands in for Delhi; traffic, climate and holidays differ.
- Delhi capacities and the place-to-lot mapping are assumptions.
- Only 07:00–17:30 is forecast; 11 weeks of history means no seasonality or yearly holidays.
- The ESP32 node is a design; it has not been built or field-tested.
- Path to production: collect several months of real Delhi counts over MQTT at all hours, add weather and holiday feeds, retrain per place and compare against a "same hour last week" baseline, then monitor forecast vs actual in Grafana and retrain on a schedule.

## Safety

This is a software demonstration. It does not control any hardware, power equipment or vehicle barriers.

## References

- Birmingham car-park occupancy dataset, 4 Oct – 19 Dec 2016.
- HCLTech Project Creation Guide — Embedded Systems & IoT, v1.0 (September 2026).
- AURUM project report v1.0 and repository docs: `docs/MODEL_CARD.md`, `docs/ARCHITECTURE.md`, `docs/MQTT_SETUP.md`.
