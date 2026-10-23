from src.config import (
    COMPANY_NAME, REFERENCE_DATE_STR, DATASET_ROOT, DATA_DIR, KB_DIR,
    PRACTICE_DIR, TOOLS_DIR, RESULTS_DIR, KNOWN_HOLIDAYS
)

def test_config_paths_exist():
    assert DATASET_ROOT.exists(), f"Dataset root not found: {DATASET_ROOT}"
    assert DATA_DIR.exists(), f"Data dir not found: {DATA_DIR}"
    assert KB_DIR.exists(), f"KB dir not found: {KB_DIR}"
    assert PRACTICE_DIR.exists(), f"Practice dir not found: {PRACTICE_DIR}"
    assert TOOLS_DIR.exists(), f"Tools dir not found: {TOOLS_DIR}"
    assert RESULTS_DIR.exists(), f"Results dir not found: {RESULTS_DIR}"

def test_config_constants():
    assert COMPANY_NAME == "ZentroMart"
    assert REFERENCE_DATE_STR == "2026-10-20"
    assert len(KNOWN_HOLIDAYS) == 4
    assert "2026-10-19" in KNOWN_HOLIDAYS
