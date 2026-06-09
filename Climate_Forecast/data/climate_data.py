import os 
import requests
import pandas as pd


LATITUDE = 48.2082
LONGITUDE = 16.3738

print(f"Querying Open-Meteo Climate Reanalysis Archives for coordinates: {LATITUDE}°N, {LONGITUDE}°E...")

# Query the Open-Meteo API
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2023-01-01",
    "end_date": "2025-12-31", 
    "daily": ["temperature_2m_mean", "surface_pressure_mean", "relative_humidity_2m_mean"],
    "timezone": "auto"
}x

response = requests.get(url, params=params)
print(f"API Response Status: {response.status_code}")
# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    daily_data = data["daily"]
    print(f"Daily Data: {daily_data}")
    # Parse the incoming chronological arrays into a DataFrame
    df_raw = pd.DataFrame({
        "Date": pd.to_datetime(daily_data["time"]),
        "Temp": daily_data["temperature_2m_mean"],
        "Pressure": daily_data["surface_pressure_mean"],
        "Humidity": daily_data["relative_humidity_2m_mean"]
    })
    print(f"DataFrame: {df_raw}")
    # Extract Calendar Parameters to fit the model architecture columns
    df_final = pd.DataFrame({
        "Month": df_raw["Date"].dt.month,
        "Day": df_raw["Date"].dt.day,
        "Temp": df_raw["Temp"],
        "Pressure": df_raw["Pressure"],
        "Humidity": df_raw["Humidity"]
    })
    print(f"DataFrame: {df_final}")
    # Drop missing data gaps using interpolation
    df_final = df_final.interpolate(method="linear").round(2)
    print(f"DataFrame: {df_final}")
    # Commit to the data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "geocoded_data.csv")
    print(f"CSV Path: {csv_path}")
    df_final.to_csv(csv_path, index=False)
    print(f"DataFrame: {df_final}")
    print("-----------------------------------------------------")
    print("REAL-WORLD ATMOSPHERIC DATA RETRIEVED SUCCESSFULLY!")
    print("-----------------------------------------------------")
    print(f"Destination: {csv_path}")
    print(f"Length:      {len(df_final)} Consecutive Days Logged.")
    print("-----------------------------------------------------")
    print(df_final.head(10))
    
else:
    print(f"Connection Error: API returned status code {response.status_code}")