from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

TICKER = "GC=F"
INTERVAL_SOURCE = "1h"
RESAMPLE_RULE = "4h"

START_DATE = "2010-01-01"
CHUNK_DAYS = 700

# real: last ~730 days of actual 1h Yahoo data (most accurate, fewer trades)
# extended: synthetic 4h from daily since 2010 (multi-year, ~600 trades)
# hybrid: synthetic history + recent real 1h overlay
DATA_MODE = "extended"

MONTE_CARLO_SIMULATIONS = 10_000
RANDOM_SEED = 42

INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01
