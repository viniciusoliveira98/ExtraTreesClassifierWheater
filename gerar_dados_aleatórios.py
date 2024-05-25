import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate random weather data
def generate_weather_data(num_records, start_date):
    np.random.seed(42)  # For reproducibility
    date_range = [start_date + timedelta(days=i) for i in range(num_records)]
    rainfall = np.random.uniform(0, 25, num_records).round(1)
    temperature = np.random.uniform(13, 24, num_records).round(1)
    humidity = np.random.uniform(40, 90, num_records).round(0).astype(int)
    wind_speed = np.random.uniform(2, 10, num_records).round(1)
    weather_condition = np.where(rainfall > 0, 'Rainy', 'Sunny')
    
    data = {
        'date': date_range,
        'rainfall': rainfall,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'weather_condition': weather_condition
    }
    
    df = pd.DataFrame(data)
    return df

# Generate 25,000 records starting from 2022-01-01
num_records = 25000
start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
df_large = generate_weather_data(num_records, start_date)
df_large.head()  # Display the first few records to verify

# Save to a CSV file
csv_path = '.../large_weather_data.csv'
df_large.to_csv(csv_path, index=False)
csv_path
