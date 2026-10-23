"""Central Configuration for VoiceOps 2026 Capstone (ZentroMart Support Agent).

All paths, frozen reference dates, constants, and API defaults are centralized here.
Reference date is frozen at 2026-10-20 per dataset specifications.
"""
import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Base directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicitly load .env from project root regardless of execution working directory
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()  # Fallback to default search

DATASET_ROOT = PROJECT_ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
DATA_DIR = DATASET_ROOT / "data"
KB_DIR = DATASET_ROOT / "kb"
PRACTICE_DIR = DATASET_ROOT / "practice"
TOOLS_DIR = DATASET_ROOT / "tools"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure runtime directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Fixed Constants
COMPANY_NAME = "ZentroMart"
REFERENCE_DATE = date(2026, 10, 20)
REFERENCE_DATE_STR = "2026-10-20"

# Calendar & Logistics Rules (KB-00.3)
BUSINESS_DAYS_PER_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
KNOWN_HOLIDAYS = [
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-19",  # Warehouse closure
    "2026-11-09",  # Warehouse closure
]

# Identity Verification Gate Rules (AGT-IDV)
MAX_VERIFICATION_RETRIES = 1
PHONE_SUFFIX_LENGTH = 4
PINCODE_LENGTH = 6

# Escalation & Ticket Rules (AGT-ESC, tools.py)
MIN_TICKET_SUMMARY_LENGTH = 20
HIGH_VALUE_THRESHOLD_INR = 10000
REPEAT_CONTACT_THRESHOLD = 2

# Speech Layer Configuration (Gnani.ai)
DEFAULT_STT_LANGUAGE = "en-IN"
ALLOWED_STT_LANGUAGES = [
    "en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN",
    "ml-IN", "mr-IN", "bn-IN", "gu-IN", "pa-IN"
]
TTS_MODEL_V25 = "timbre-v2.5"
TTS_MODEL_V30 = "timbre-v3.0"
DEFAULT_TTS_MODEL = TTS_MODEL_V25
DEFAULT_TTS_VOICE = "Kaveri"
MAX_AUDIO_TURN_SECONDS = 60

# Model Settings
# gemini-3.5-flash-lite is the active verified production model on the current API tier
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_TEMPERATURE = 0.0  # Zero temperature for reproducible benchmarking

# Embedding Settings
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
