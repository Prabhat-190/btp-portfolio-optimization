"""NSE universe, dates, and BTP experiment settings."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# Sector-balanced Nifty 50 names with long, clean NSE history.
# LICI.NS was dropped: IPO only in May 2022, so 2020–2022 is empty.
NSE_UNIVERSE = {
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Banking"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Banking"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "sector": "Telecom"},
    "INFY.NS": {"name": "Infosys", "sector": "IT"},
    "ITC.NS": {"name": "ITC", "sector": "FMCG"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG"},
    "LT.NS": {"name": "Larsen & Toubro", "sector": "Capital Goods"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Pharma"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto"},
}

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "Nifty 50"

# Current Indian sample: COVID crash through latest NSE close.
START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()

# India 10Y G-Sec / policy corridor, used as daily rf = annual / 252.
RISK_FREE_ANNUAL = 0.065
TRADING_DAYS = 252

# Long-only, fully invested, concentration cap (not the old 57% synthetic cap).
MAX_WEIGHT = 0.30
MIN_HISTORY_FRAC = 0.90

# Walk-forward: 2y estimate window, quarterly out-of-sample hold.
TRAIN_DAYS = 504
TEST_DAYS = 63
CVAR_ALPHA = 0.05

# NSGA-III
NSGA_PARTITIONS = 5
NSGA_GENERATIONS = 80
NSGA_SEED = 42
N_PARETO_KEEP = 24
