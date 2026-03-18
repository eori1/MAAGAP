import os
import numpy as np

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REAL_DATA_FILE = os.path.join(BASE_DIR, "LIST-OF-ALL-ONGOING-PPAS-2026.xlsx")

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Risk Thresholds (from manuscript) ---
RISK_THRESHOLDS = {"Low": (0.0, 0.25), "Medium": (0.25, 0.50), "High": (0.50, 0.75), "Critical": (0.75, 1.0)}
RISK_LABELS = ["Low", "Medium", "High", "Critical"]

# --- Project Parameters (from manuscript delimitation) ---
INFRA_DURATION_MONTHS = 12
NON_INFRA_DURATION_MONTHS = 6

IMPLEMENTING_AGENCIES = [
    "PSWDO", "Provincial Accountant's Office", "OPPESM", "IPG-HRMDO",
    "PCDO", "PHO", "OLEDIPO", "PGENRO", "ICTMO",
    "Provincial Agriculture Office", "HMO", "IPG/HMO",
    "Provincial Engineering Office", "Provincial Planning Office",
]

AGENCY_INFRA_RATIO = {
    "PSWDO": 0.08, "Provincial Accountant's Office": 0.02, "OPPESM": 0.10,
    "IPG-HRMDO": 0.05, "PCDO": 0.10, "PHO": 0.30, "OLEDIPO": 0.20,
    "PGENRO": 0.40, "ICTMO": 0.15, "Provincial Agriculture Office": 0.35,
    "HMO": 0.80, "IPG/HMO": 0.85, "Provincial Engineering Office": 0.95,
    "Provincial Planning Office": 0.20,
}

AGENCY_CAPACITY_SCORE = {
    "PSWDO": 0.75, "Provincial Accountant's Office": 0.80, "OPPESM": 0.65,
    "IPG-HRMDO": 0.70, "PCDO": 0.60, "PHO": 0.72, "OLEDIPO": 0.55,
    "PGENRO": 0.68, "ICTMO": 0.78, "Provincial Agriculture Office": 0.62,
    "HMO": 0.70, "IPG/HMO": 0.67, "Provincial Engineering Office": 0.73,
    "Provincial Planning Office": 0.82,
}

CONTRACTORS = [
    "Juantong Agri-Ventures Trading", "3rd Dragon Builders Innovation",
    "Conquest Construction Supply", "CEBU ERNBRI IMPORT INC.",
    "Iloilo New Agri-Industrial Marketing", "RS AGRO INDUSTRIAL Corporation",
    "Red Hammer International Builders", "RMKP Construction",
    "A. Maquiling Construction and Supply", "Orchard Valley Enterprises",
    "Dreamer's Valley Construction", "PACEMCO Construction",
    "PCC Construction Corp", "JV Pacific Builders",
    "Metro Iloilo Construction", "Western Visayas Builders Inc.",
    "Panay Construction Services", "Golden West Engineering",
    "Island Builders Corporation", "Visayan Construction Group",
]

# Seeded reliability scores per contractor (0=unreliable, 1=perfect)
_rng = np.random.RandomState(SEED)
CONTRACTOR_RELIABILITY = {c: round(v, 2) for c, v in zip(CONTRACTORS, _rng.uniform(0.35, 0.95, len(CONTRACTORS)))}

ILOILO_MUNICIPALITIES = [
    "Iloilo City", "Oton", "Sta. Barbara", "Cabatuan", "Maasin",
    "Dumangas", "Pavia", "Leganes", "Zarraga", "San Miguel",
    "Barotac Nuevo", "Alimodian", "Leon", "Tubungan", "Igbaras",
    "Miagao", "San Joaquin", "Guimbal", "Tigbauan", "Anilao",
    "Banate", "Barotac Viejo", "Ajuy", "Concepcion", "Sara",
    "San Dionisio", "Batad", "Estancia", "Carles", "Balasan",
    "Lemery", "Dingle", "Pototan", "Janiuay", "Lambunao",
    "Calinog", "Bingawan", "Passi City", "San Enrique", "Dueñas",
]

FUNDING_SOURCES = [
    "General Fund", "20% NTA", "MOOE", "Supplemental Budget",
    "Special Budget", "HFEP", "GAD Fund", "LDRRMF", "SEF", "Trust Fund",
]

PROCUREMENT_MODES = [
    "Competitive Bidding", "Alternative Mode (Negotiated)",
    "Alternative Mode (Shopping)", "Direct Contracting", "Repeat Order",
]

# --- PAGASA Iloilo Climate Data (historical monthly averages) ---
ILOILO_MONTHLY_RAINFALL_MM = {
    1: 55, 2: 35, 3: 35, 4: 55, 5: 150,
    6: 300, 7: 400, 8: 400, 9: 350, 10: 300, 11: 200, 12: 100,
}

ILOILO_MONTHLY_TYPHOON_DAYS = {
    1: 0.3, 2: 0.2, 3: 0.1, 4: 0.2, 5: 0.5,
    6: 1.5, 7: 2.5, 8: 3.0, 9: 2.8, 10: 2.5, 11: 2.0, 12: 1.0,
}

# --- PSA Economic Indicators (approximate annual values) ---
PSA_CPI_ANNUAL = {
    2016: 94.5, 2017: 97.7, 2018: 100.0, 2019: 102.5, 2020: 105.1,
    2021: 109.8, 2022: 116.0, 2023: 123.1, 2024: 127.8, 2025: 132.5,
}

PSA_CMRPI_ANNUAL = {
    2016: 96.0, 2017: 98.5, 2018: 100.0, 2019: 103.2, 2020: 107.5,
    2021: 114.0, 2022: 125.3, 2023: 131.0, 2024: 135.8, 2025: 139.2,
}

# --- Model Hyperparameters ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LSTM_MAX_TIMESTEPS = 4
LSTM_EPOCHS = 60
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 64

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 15

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 10
XGB_LEARNING_RATE = 0.08

SYNTHETIC_NUM_PROJECTS = 3000
SYNTHETIC_YEARS = list(range(2016, 2026))
