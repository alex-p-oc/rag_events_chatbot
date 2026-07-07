"""
Centralized path configuration for the project
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Project Sub-directories
DATA_SUBDIR = BASE_DIR / "data"
VECTOR_STORE_SUBDIR = BASE_DIR / "vector_store"
LOGS_SUBDIR = BASE_DIR / "logs"

# Data files
CSV_EVENTS_FILE = DATA_SUBDIR / "events.csv"

# Log files
INTERACTIONS_LOG_FILE = LOGS_SUBDIR / "interaction_logs.jsonl"

# Vector store files
FAISS_INDEX_DIR= VECTOR_STORE_SUBDIR / "faiss_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
STORE_FILE = VECTOR_STORE_SUBDIR / "store.pkl"