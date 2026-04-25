# ML Aeronautics — Aircraft Trajectory Reconstruction

ADS-B / ADS-C gap-filling pipeline that reconstructs missing trajectory points
using a bidirectional LSTM, with great-circle interpolation as a fallback.
Includes a FastAPI service, training scripts, evaluation scripts, and an
emissions-sensitivity analysis.

## Project layout

| Path | Purpose |
| --- | --- |
| [data_pipeline.py](data_pipeline.py) | Pulls flights from the OpenSky Trino database, cleans them, and splits into train/test parquets under `data/clean/`. |
| [baseline.py](baseline.py) | Great-circle interpolation baseline. |
| [model.py](model.py) | LSTM trajectory model definition + helpers. |
| [train_lstm.py](train_lstm.py) | Trains the bidirectional LSTM on `data/clean/tracks/`. |
| [evaluate.py](evaluate.py) / [evaluate_all.py](evaluate_all.py) | Compares baseline vs LSTM on held-out flights. |
| [phase6_emissions.py](phase6_emissions.py) | Distance + CO2 sensitivity from reconstructed tracks. |
| [api.py](api.py) | FastAPI service exposing reconstruction endpoints. |
| [Dockerfile](Dockerfile) / [docker-compose.yml](docker-compose.yml) | Container build + run for the API. |
| [weights/](weights/) | Trained model weights (`lstm_residual_trajectory.keras`). |

## Prerequisites

- Python 3.11
- An OpenSky account with Trino access (only required for `data_pipeline.py`
  and the `/opensky/*` API endpoints — pre-built parquets and weights work
  without it)
- Docker + Docker Compose (optional, for containerized runs)

## Setup

### 1. Clone and create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure credentials

Create a `.env` file at the project root with your OpenSky credentials:

```env
OPENSKY_CLIENT_ID=your-client-id
OPENSKY_CLIENT_SECRET=your-client-secret
OPENSKY_USERNAME=your-username
OPENSKY_PASSWORD=your-password
```

These are loaded automatically by `python-dotenv` and used by the data
pipeline and the live OpenSky API endpoints.

## Running the pipeline

### Build the dataset

Pulls flights from the default European airports for the last 2 days and
writes parquet files to `data/clean/tracks/` (train) and
`data/clean/test_tracks/` (held-out test set):

```bash
python data_pipeline.py
```

Common flags:

```bash
python data_pipeline.py --airports EDDF EGLL LFPG --days 5 --test-ratio 0.2 --clean
```

### Train the LSTM

Trains on `data/clean/tracks/` (80/20 train/val) and benchmarks on
`data/clean/test_tracks/`. Writes weights to
`weights/lstm_residual_trajectory.keras`:

```bash
python train_lstm.py
```

### Evaluate

Single flight (auto-picks the longest test track if no path given):

```bash
python evaluate.py
python evaluate.py data/clean/test_tracks/<file>.parquet
```

Run the full evaluation suite across all held-out flights:

```bash
python evaluate_all.py
```

### Emissions sensitivity

```bash
python phase6_emissions.py
```

Writes `phase6_emissions_results.csv` summarizing per-flight distance and CO2
deltas between baseline and LSTM reconstructions.

## Running the API

### Local (uvicorn)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose

```bash
docker compose up --build
```

The service is then available at http://localhost:8000.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Demo web UI. |
| `GET` | `/health` | Liveness check. |
| `GET` | `/flights` | List local flight IDs (from `data/clean/tracks/`). |
| `GET` | `/opensky/flights` | List recent flights from OpenSky. |
| `GET` | `/reconstruct/{flight_id}` | Reconstruct a local flight. |
| `GET` | `/opensky/reconstruct/{icao24}` | Pull from OpenSky and reconstruct. |
| `POST` | `/reconstruct` | Reconstruct from a posted track. |
| `GET` | `/demo/{flight_id}` | Demo payload (original + reconstructed). |

Interactive docs: http://localhost:8000/docs

## Notes

- `docker-compose.yml` bind-mounts `./data/clean/tracks` and `./weights` read-only,
  so you can swap data or weights without rebuilding the image.
- If `weights/lstm_residual_trajectory.keras` is missing at API startup,
  reconstruction falls back to great-circle interpolation.
- The reconstruction model expects at least `2 * CONTEXT_LENGTH + GAP_LENGTH`
  (= 50) points per flight.
