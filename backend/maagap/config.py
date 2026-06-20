import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

# Set base seed immediately
SEED = 42
np.random.seed(SEED)

@dataclass(frozen=True)
class PathConfig:
    """Centralized path configurations."""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_dir: str = field(init=False)
    data_processed_dir: str = field(init=False)
    models_dir: str = field(init=False)
    outputs_dir: str = field(init=False)
    real_data_file: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data_raw_dir", os.path.join(self.base_dir, "data", "raw"))
        object.__setattr__(self, "data_processed_dir", os.path.join(self.base_dir, "data", "processed"))
        object.__setattr__(self, "models_dir", os.path.join(self.base_dir, "models"))
        object.__setattr__(self, "outputs_dir", os.path.join(self.base_dir, "outputs"))
        
        # Determine preferred real file
        preferred_files = [
            "Copy of 2022 conso Fund Transfer worksheet (2).xlsx",
            "LIST-OF-ALL-ONGOING-PPAS-2026.xlsx",
        ]
        real_file = os.path.join(self.data_raw_dir, preferred_files[0])
        for name in preferred_files:
            candidate = os.path.join(self.data_raw_dir, name)
            if os.path.exists(candidate):
                real_file = candidate
                break
        object.__setattr__(self, "real_data_file", real_file)
        
        for d in [self.data_raw_dir, self.data_processed_dir, self.models_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)

paths = PathConfig()

# Re-export key paths for backward-compatibility if needed
BASE_DIR = paths.base_dir
DATA_RAW_DIR = paths.data_raw_dir
DATA_PROCESSED_DIR = paths.data_processed_dir
MODELS_DIR = paths.models_dir
OUTPUTS_DIR = paths.outputs_dir
REAL_DATA_FILE = paths.real_data_file

@dataclass(frozen=True)
class RiskConfig:
    """Risk scoring thresholds and labels."""
    thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Low": (0.0, 0.30), 
        "Medium": (0.30, 0.70), 
        "High": (0.70, 0.90), 
        "Critical": (0.90, 1.0)
    })
    labels: List[str] = field(default_factory=lambda: ["Low", "Medium", "High", "Critical"])

risk_cfg = RiskConfig()
RISK_THRESHOLDS = risk_cfg.thresholds
RISK_LABELS = risk_cfg.labels

@dataclass(frozen=True)
class ModelHyperparameters:
    """Hyperparameters for machine learning models."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    lstm_max_timesteps: int = 4
    lstm_epochs: int = 60
    lstm_batch_size: int = 32
    lstm_units: int = 64

    rf_n_estimators: int = 300
    rf_max_depth: int = 15

    xgb_n_estimators: int = 300
    xgb_max_depth: int = 10
    xgb_learning_rate: float = 0.08

    use_cuda_if_available: bool = True
    xgb_try_gpu: bool = True

    random_search_n_iter: int = 15
    random_search_cv: int = 3

model_params = ModelHyperparameters()

# Provide direct mappings for compatibility while migrating
TRAIN_RATIO = model_params.train_ratio
VAL_RATIO = model_params.val_ratio
TEST_RATIO = model_params.test_ratio
LSTM_MAX_TIMESTEPS = model_params.lstm_max_timesteps
LSTM_EPOCHS = model_params.lstm_epochs
LSTM_BATCH_SIZE = model_params.lstm_batch_size
LSTM_UNITS = model_params.lstm_units
RF_N_ESTIMATORS = model_params.rf_n_estimators
RF_MAX_DEPTH = model_params.rf_max_depth
XGB_N_ESTIMATORS = model_params.xgb_n_estimators
XGB_MAX_DEPTH = model_params.xgb_max_depth
XGB_LEARNING_RATE = model_params.xgb_learning_rate
USE_CUDA_IF_AVAILABLE = model_params.use_cuda_if_available
XGB_TRY_GPU = model_params.xgb_try_gpu
RANDOM_SEARCH_N_ITER = model_params.random_search_n_iter
RANDOM_SEARCH_CV = model_params.random_search_cv

@dataclass(frozen=True)
class ProjectParams:
    """Parameters defining project characteristics for synthetic generation."""
    infra_duration_months: int = 12
    non_infra_duration_months: int = 6
    synthetic_num_projects: int = 3000
    synthetic_years: List[int] = field(default_factory=lambda: list(range(2016, 2026)))

    implementing_agencies: List[str] = field(default_factory=lambda: [
        "PSWDO", "Provincial Accountant's Office", "OPPESM", "IPG-HRMDO",
        "PCDO", "PHO", "OLEDIPO", "PGENRO", "ICTMO",
        "Provincial Agriculture Office", "HMO", "IPG/HMO",
        "Provincial Engineering Office", "Provincial Planning Office",
    ])

    agency_infra_ratio: Dict[str, float] = field(default_factory=lambda: {
        "PSWDO": 0.08, "Provincial Accountant's Office": 0.02, "OPPESM": 0.10,
        "IPG-HRMDO": 0.05, "PCDO": 0.10, "PHO": 0.30, "OLEDIPO": 0.20,
        "PGENRO": 0.40, "ICTMO": 0.15, "Provincial Agriculture Office": 0.35,
        "HMO": 0.80, "IPG/HMO": 0.85, "Provincial Engineering Office": 0.95,
        "Provincial Planning Office": 0.20,
    })

    agency_capacity_score: Dict[str, float] = field(default_factory=lambda: {
        "PSWDO": 0.75, "Provincial Accountant's Office": 0.80, "OPPESM": 0.65,
        "IPG-HRMDO": 0.70, "PCDO": 0.60, "PHO": 0.72, "OLEDIPO": 0.55,
        "PGENRO": 0.68, "ICTMO": 0.78, "Provincial Agriculture Office": 0.62,
        "HMO": 0.70, "IPG/HMO": 0.67, "Provincial Engineering Office": 0.73,
        "Provincial Planning Office": 0.82,
    })

    contractors: List[str] = field(default_factory=lambda: [
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
    ])

    iloilo_municipalities: List[str] = field(default_factory=lambda: [
        "Iloilo City", "Oton", "Sta. Barbara", "Cabatuan", "Maasin",
        "Dumangas", "Pavia", "Leganes", "Zarraga", "San Miguel",
        "Barotac Nuevo", "Alimodian", "Leon", "Tubungan", "Igbaras",
        "Miagao", "San Joaquin", "Guimbal", "Tigbauan", "Anilao",
        "Banate", "Barotac Viejo", "Ajuy", "Concepcion", "Sara",
        "San Dionisio", "Batad", "Estancia", "Carles", "Balasan",
        "Lemery", "Dingle", "Pototan", "Janiuay", "Lambunao",
        "Calinog", "Bingawan", "Passi City", "San Enrique", "Dueñas",
    ])

    funding_sources: List[str] = field(default_factory=lambda: [
        "General Fund", "20% NTA", "MOOE", "Supplemental Budget",
        "Special Budget", "HFEP", "GAD Fund", "LDRRMF", "SEF", "Trust Fund",
    ])

    procurement_modes: List[str] = field(default_factory=lambda: [
        "Competitive Bidding", "Alternative Mode (Negotiated)",
        "Alternative Mode (Shopping)", "Direct Contracting", "Repeat Order",
    ])

proj_params = ProjectParams()

# Extract for easy imports
INFRA_DURATION_MONTHS = proj_params.infra_duration_months
NON_INFRA_DURATION_MONTHS = proj_params.non_infra_duration_months
SYNTHETIC_NUM_PROJECTS = proj_params.synthetic_num_projects
SYNTHETIC_YEARS = proj_params.synthetic_years
IMPLEMENTING_AGENCIES = proj_params.implementing_agencies
AGENCY_INFRA_RATIO = proj_params.agency_infra_ratio
AGENCY_CAPACITY_SCORE = proj_params.agency_capacity_score
CONTRACTORS = proj_params.contractors
ILOILO_MUNICIPALITIES = proj_params.iloilo_municipalities
FUNDING_SOURCES = proj_params.funding_sources
PROCUREMENT_MODES = proj_params.procurement_modes

# Seeded reliability scores per contractor (0=unreliable, 1=perfect)
_rng = np.random.RandomState(SEED)
CONTRACTOR_RELIABILITY = {c: round(v, 2) for c, v in zip(CONTRACTORS, _rng.uniform(0.35, 0.95, len(CONTRACTORS)))}

@dataclass(frozen=True)
class EnvironmentData:
    """Historical climate and economic data."""
    iloilo_monthly_rainfall_mm: Dict[int, float] = field(default_factory=lambda: {
        1: 55, 2: 35, 3: 35, 4: 55, 5: 150,
        6: 300, 7: 400, 8: 400, 9: 350, 10: 300, 11: 200, 12: 100,
    })

    iloilo_monthly_typhoon_days: Dict[int, float] = field(default_factory=lambda: {
        1: 0.3, 2: 0.2, 3: 0.1, 4: 0.2, 5: 0.5,
        6: 1.5, 7: 2.5, 8: 3.0, 9: 2.8, 10: 2.5, 11: 2.0, 12: 1.0,
    })

    psa_cpi_annual: Dict[int, float] = field(default_factory=lambda: {
        2016: 94.5, 2017: 97.7, 2018: 100.0, 2019: 102.5, 2020: 105.1,
        2021: 109.8, 2022: 116.0, 2023: 123.1, 2024: 127.8, 2025: 132.5,
    })

    psa_cmrpi_annual: Dict[int, float] = field(default_factory=lambda: {
        2016: 96.0, 2017: 98.5, 2018: 100.0, 2019: 103.2, 2020: 107.5,
        2021: 114.0, 2022: 125.3, 2023: 131.0, 2024: 135.8, 2025: 139.2,
    })

env_data = EnvironmentData()
ILOILO_MONTHLY_RAINFALL_MM = env_data.iloilo_monthly_rainfall_mm
ILOILO_MONTHLY_TYPHOON_DAYS = env_data.iloilo_monthly_typhoon_days
PSA_CPI_ANNUAL = env_data.psa_cpi_annual
PSA_CMRPI_ANNUAL = env_data.psa_cmrpi_annual
