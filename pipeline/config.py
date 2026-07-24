"""
Shared configuration for all feature extraction scripts.
Import this in every script to keep paths and constants consistent.
"""

from pathlib import Path

# ── Directory paths ───────────────────────────────────────────────────────────
# BASE_DIR resolves to the pipeline/ folder regardless of working directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
HOSP_DIR = DATA_DIR / "hosp"
CORE_DIR = DATA_DIR / "core"
ICU_DIR = DATA_DIR / "icu"
OUTPUT_DIR = BASE_DIR / "output"

# admissions.csv.gz and patients.csv.gz live in core/ on PhysioNet
# but some distributions (e.g. Mika's zip) put them in hosp/.
# This helper picks the right location automatically.
def _resolve(filename: str) -> Path:
    """Return path to a file that may be in core/ or hosp/."""
    if (CORE_DIR / filename).exists():
        return CORE_DIR / filename
    return HOSP_DIR / filename

ADMISSIONS_PATH = _resolve("admissions.csv.gz")
PATIENTS_PATH = _resolve("patients.csv.gz")
TRANSFERS_PATH = _resolve("transfers.csv.gz")

# RXCUI → ATC mapping file (download separately if missing)
# Source: https://github.com/MIT-LCP/mimic-code
MAPPING_PATH = DATA_DIR / "RXCUI2atc4.csv"

# ── Cohort thresholds ─────────────────────────────────────────────────────────
THRESHOLD = 7  # LOS > THRESHOLD days → prolonged_stay = 1
OBS_WINDOW = 48  # hours of observation used for time-series features
MIN_LOS_DAYS = 2  # minimum LOS to ensure a full 48h observation window
MIN_AGE = 18

# ── Vital sign itemids from chartevents ──────────────────────────────────────
# Each entry: itemid → (feature_name, unit_flag)
# unit_flag "F" = Fahrenheit (will be converted to Celsius)
VITAL_ITEMIDS = {
    220045: ("heart_rate", None),
    # Blood pressure — invasive (arterial line)
    220050: ("sbp", None),
    220051: ("dbp", None),
    220052: ("map", None),
    # Blood pressure — non-invasive (NIBP cuff)
    220179: ("sbp", None),
    220180: ("dbp", None),
    220181: ("map", None),
    220210: ("resp_rate", None),
    220277: ("spo2", None),
    # Temperature — both units present in MIMIC-IV
    223761: ("temperature_f", "F"),  # Fahrenheit → converted to Celsius
    223762: ("temperature_c", "C"),  # already Celsius
    220621: ("glucose", None),
    225664: ("glucose", None),
    # Glasgow Coma Scale (3 components)
    220739: ("gcs_eye", None),
    223900: ("gcs_verbal", None),
    223901: ("gcs_motor", None),
}

# Itemids for urine output (from outputevents, not chartevents)
URINE_ITEMIDS = [226559, 226560, 226561, 226584, 226563, 226564,
                 226565, 226567, 226557, 227488, 227489]

# ── Physiological range filters ───────────────────────────────────────────────
# Values outside these ranges are treated as measurement errors and dropped.
RANGE_FILTERS = {
    "heart_rate": (0, 300),
    "sbp": (0, 300),
    "dbp": (0, 200),
    "map": (0, 250),
    "resp_rate": (0, 80),
    "spo2": (0, 100),
    "temperature": (25, 45),  # Celsius after conversion
    "glucose": (0, 1000),
    "gcs_eye": (1, 4),
    "gcs_verbal": (1, 5),
    "gcs_motor": (1, 6),
    "gcs_total": (3, 15),
    "urine_output": (0, 5000),
}

TS_FEATURES = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
    "temperature", "glucose", "gcs_eye", "gcs_verbal", "gcs_motor",
    "gcs_total", "urine_output",
]

# ── ICD-9 numeric ranges → disease category ──────────────────────────────────
ICD9_RANGES = [
    ("001", "139", "infectious"),
    ("140", "239", "neoplasms"),
    ("240", "279", "endocrine"),
    ("280", "289", "blood"),
    ("290", "319", "mental"),
    ("320", "389", "nervous"),
    ("390", "459", "circulatory"),
    ("460", "519", "respiratory"),
    ("520", "579", "digestive"),
    ("580", "629", "genitourinary"),
    ("630", "679", "pregnancy"),
    ("680", "709", "skin"),
    ("710", "739", "musculoskeletal"),
    ("740", "759", "congenital"),
    ("760", "779", "perinatal"),
    ("780", "799", "ill_defined"),
    ("800", "999", "injury"),
    ("E", "V", "supplementary"),
]

# ICD-10 first-character(s) → disease category
ICD10_MAP = {
    "A": "infectious", "B": "infectious",
    "C": "neoplasms",
    "D0": "neoplasms", "D1": "neoplasms", "D2": "neoplasms",
    "D3": "neoplasms", "D4": "neoplasms",
    "D5": "blood", "D6": "blood", "D7": "blood",
    "D8": "blood", "D9": "blood",
    "E": "endocrine", "F": "mental", "G": "nervous",
    "H0": "nervous", "H1": "nervous", "H2": "nervous",
    "H3": "nervous", "H4": "nervous", "H5": "nervous",
    "H6": "nervous", "H7": "nervous", "H8": "nervous",
    "H9": "nervous",
    "I": "circulatory", "J": "respiratory", "K": "digestive",
    "L": "skin", "M": "musculoskeletal", "N": "genitourinary",
    "O": "pregnancy", "P": "perinatal", "Q": "congenital",
    "R": "ill_defined",
    "S": "injury", "T": "injury",
    "V": "supplementary", "W": "supplementary", "X": "supplementary",
    "Y": "supplementary", "Z": "supplementary", "U": "supplementary",
}

ICD_CATEGORIES = [
    "infectious", "neoplasms", "endocrine", "blood", "mental",
    "nervous", "circulatory", "respiratory", "digestive", "genitourinary",
    "pregnancy", "skin", "musculoskeletal", "congenital", "perinatal",
    "ill_defined", "injury", "supplementary",
]

# ── ATC Level 1 drug classes ──────────────────────────────────────────────────
ATC1_CODES = list("ABCDGHJLMNPRSV")
ATC1_COL_NAMES = {
    "A": "atc_alimentary",
    "B": "atc_blood",
    "C": "atc_cardiovascular",
    "D": "atc_dermatological",
    "G": "atc_genitourinary",
    "H": "atc_hormonal",
    "J": "atc_antiinfective",
    "L": "atc_antineoplastic",
    "M": "atc_musculoskeletal",
    "N": "atc_nervous",
    "P": "atc_antiparasitic",
    "R": "atc_respiratory",
    "S": "atc_sensory",
    "V": "atc_various",
}
