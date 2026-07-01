import os
import pandas as pd
import random

# Absolute paths
BACKEND_PROJ = r"c:\Users\ASUS\Desktop\Tisis\backend\data\processed\ppdo_2026_cleaned.csv"
OUT_DIR = r"c:\Users\ASUS\Desktop\Tisis\frontend\public\data"

os.makedirs(OUT_DIR, exist_ok=True)

# Approximate central coordinates for Iloilo municipalities to bypass rate-limits and ensure speed
MUNI_COORDS = {
    "Oton": (10.6974, 122.4777),
    "Tigbauan": (10.6781, 122.3807),
    "Guimbal": (10.6657, 122.3216),
    "Miagao": (10.6432, 122.2359),
    "San Joaquin": (10.5925, 122.1408),
    "Igbaras": (10.7180, 122.2647),
    "Tubungan": (10.7719, 122.3023),
    "Leon": (10.7816, 122.3831),
    "Alimodian": (10.8285, 122.4339),
    "San Miguel": (10.7607, 122.4646),
    "Pavia": (10.7656, 122.5401),
    "Santa Barbara": (10.8242, 122.5348),
    "Leganes": (10.7885, 122.5888),
    "Zarraga": (10.8306, 122.6053),
    "New Lucena": (10.8804, 122.5857),
    "Cabatuan": (10.8788, 122.4844),
    "Maasin": (10.8887, 122.4307),
    "Janiuay": (10.9542, 122.5029),
    "Badiangan": (11.0028, 122.5647),
    "Mina": (10.9320, 122.5937),
    "Pototan": (10.9576, 122.6288),
    "Dingle": (11.0069, 122.6685),
    "Duenas": (11.0543, 122.6186),
    "Passi City": (11.1092, 122.6416),
    "San Enrique": (11.0583, 122.6847),
    "Banate": (11.0378, 122.8028),
    "Anilao": (10.9855, 122.7533),
    "Barotac Nuevo": (10.8920, 122.7013),
    "Dumangas": (10.8304, 122.7118),
    "Barotac Viejo": (11.0582, 122.8488),
    "San Rafael": (11.1648, 122.8361),
    "Lemery": (11.2334, 122.9157),
    "Ajuy": (11.1718, 123.0185),
    "Sara": (11.2618, 123.0185),
    "Concepcion": (11.2721, 123.1094),
    "San Dionisio": (11.3149, 123.0847),
    "Batad": (11.3653, 123.0373),
    "Estancia": (11.4542, 123.1517),
    "Balasan": (11.4586, 123.0561),
    "Carles": (11.5648, 123.1534),
    "Calinog": (11.1213, 122.5317),
    "Lambunao": (11.0560, 122.4828),
    "Bingawan": (11.2323, 122.6074),
    "Iloilo City": (10.7202, 122.5621)
}

print("Loading raw real data...")
df_proj = pd.read_csv(BACKEND_PROJ)

# Filter for strictly infrastructure projects (no laptops, chairs, etc)
df_proj = df_proj[df_proj['project_type'].str.contains('Infrastructure', case=False, na=False)]
# Exclude Non-Infrastructure explicitly
df_proj = df_proj[~df_proj['project_type'].str.contains('Non-Infrastructure', case=False, na=False)]

# Randomly sample 150 projects so the UI isn't overwhelmed
if len(df_proj) > 150:
    df_proj = df_proj.sample(n=150, random_state=42).copy()

# Ensure completely unique IDs so Map click events don't collide
df_proj['no.'] = range(1, len(df_proj) + 1)

# Generate diverse progress rates for the UI
df_proj['physical_accomplishment'] = [random.randint(5, 100) for _ in range(len(df_proj))]

print(f"Total projects loaded: {len(df_proj)}")

def geocode_location(loc):
    loc_str = str(loc)
    for muni, coords in MUNI_COORDS.items():
        if muni.lower() in loc_str.lower():
            lat = coords[0] + random.uniform(-0.015, 0.015)
            lng = coords[1] + random.uniform(-0.015, 0.015)
            return pd.Series([lat, lng, muni])
            
    # Default to a random KNOWN land municipality to avoid spawning in the ocean
    muni = random.choice(list(MUNI_COORDS.keys()))
    coords = MUNI_COORDS[muni]
    lat = coords[0] + random.uniform(-0.015, 0.015)
    lng = coords[1] + random.uniform(-0.015, 0.015)
    return pd.Series([lat, lng, muni])

print("Geocoding locations...")
df_proj[['lat', 'lng', 'inferred_muni']] = df_proj['location'].apply(geocode_location)

# Mock AI Predictions since the real CSV is just raw cleaned data without predictions
print("Generating Mock AI Predictions...")
df_proj['delay_probability'] = [random.uniform(0.05, 0.85) for _ in range(len(df_proj))]
df_proj['overrun_probability'] = [random.uniform(0.05, 0.65) for _ in range(len(df_proj))]

def assign_risk(prob):
    if prob > 0.6: return "Critical"
    elif prob > 0.4: return "High"
    elif prob > 0.2: return "Medium"
    return "Low"

df_proj['risk_category'] = df_proj['delay_probability'].apply(assign_risk)

out_proj = os.path.join(OUT_DIR, "demo_projects.csv")
df_proj.to_csv(out_proj, index=False)
print(f"Exported projects to {out_proj}")

print("Demo data preparation complete!")
