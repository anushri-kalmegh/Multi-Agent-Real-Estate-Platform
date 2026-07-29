"""Shared, location-independent paths for PropWise AI."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RAG_DIR = PROJECT_ROOT / "rag"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATABASE_PATH = DATA_DIR / "propwise.db"
CHROMA_DIR = RAG_DIR / "chroma"
ROI_MODEL_PATH = MODEL_DIR / "roi_scenario_model.pkl"
ROI_MODEL_METADATA_PATH = MODEL_DIR / "roi_scenario_model.json"
APP_DATABASE_PATH = DATA_DIR / "propwise_app.db"
LOG_DIR = PROJECT_ROOT / "logs"
