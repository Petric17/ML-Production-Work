import os 
import pandas as pd

raw_path = "data/real_climate.csv"
output_path = "data/geocoded_data.csv"

print(f'Loading raw data from {raw_path}...')
df = pd.read_csv(raw_path)

df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Latitude'] = -37.8136
df['Longitude'] = 144.9631
df['Location_Name'] = "Melbourne, Australia"

df = df[['Date', 'Location_Name', 'Latitude', 'Longitude', 'Month', 'Day', 'Temp']]
df.to_csv(output_path, index=False)

print(f'Geographical data added to the dataset and saved to {output_path}')
