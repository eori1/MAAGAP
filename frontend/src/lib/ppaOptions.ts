// Real domain values for manual PPA entry, transcribed from
// backend/maagap/config.py (the source of truth -- keep these in sync
// if that file changes). Project type duration is a fixed manuscript rule,
// not a free-entry field: temporal feature engineering elsewhere assumes
// duration_months // 3 quarters, so an arbitrary duration would break that.
export const PROJECT_TYPES = [
  { value: "Infrastructure", label: "Infrastructure", durationMonths: 12 },
  { value: "Non-Infrastructure", label: "Non-Infrastructure", durationMonths: 6 },
] as const;

export const IMPLEMENTING_AGENCIES = [
  "PSWDO", "Provincial Accountant's Office", "OPPESM", "IPG-HRMDO",
  "PCDO", "PHO", "OLEDIPO", "PGENRO", "ICTMO",
  "Provincial Agriculture Office", "HMO", "IPG/HMO",
  "Provincial Engineering Office", "Provincial Planning Office",
];

export const FUNDING_SOURCES = [
  "General Fund", "20% NTA", "MOOE", "Supplemental Budget",
  "Special Budget", "HFEP", "GAD Fund", "LDRRMF", "SEF", "Trust Fund",
];

export const MUNICIPALITIES = [
  "Iloilo City", "Oton", "Sta. Barbara", "Cabatuan", "Maasin",
  "Dumangas", "Pavia", "Leganes", "Zarraga", "San Miguel",
  "Barotac Nuevo", "Alimodian", "Leon", "Tubungan", "Igbaras",
  "Miagao", "San Joaquin", "Guimbal", "Tigbauan", "Anilao",
  "Banate", "Barotac Viejo", "Ajuy", "Concepcion", "Sara",
  "San Dionisio", "Batad", "Estancia", "Carles", "Balasan",
  "Lemery", "Dingle", "Pototan", "Janiuay", "Lambunao",
  "Calinog", "Bingawan", "Passi City", "San Enrique", "Dueñas",
];
