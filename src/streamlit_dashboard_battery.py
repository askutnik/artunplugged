import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title='Solar Forecast Dashboard', layout='wide')
st.title("☀️ Solar Irradiance Forecast (PVGIS Data) + 🔋 Battery Simulation")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/pvgis_cleaned.csv', parse_dates=['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    return df

df = load_data()

# Show raw data toggle
if st.checkbox("Show raw data"):
    st.write(df.head())

# Features and target
X = df[['hour', 'month', 'temperature']]
y = df['GTI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display MSE
st.metric(label="Model Mean Squared Error", value=f"{mse:.2f}")

# Plot actual vs predicted
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(y_test, y_pred, alpha=0.5)
ax1.set_xlabel('Actual GTI (W/m²)')
ax1.set_ylabel('Predicted GTI (W/m²)')
ax1.set_title('Actual vs Predicted Solar Irradiance (GTI)')
ax1.grid(True)
st.pyplot(fig1)

# --- Battery Simulation ---
st.subheader("🔋 Battery Simulation Based on Predicted GTI")

# Simulate battery over time using predicted GTI from the full dataset
df['predicted_GTI'] = model.predict(X)

battery = []
level = 50  # start at 50%
max_capacity = 100
load_watts = 10
charge_efficiency = 0.9
conversion_factor = 0.1  # GTI to charging W

for gti in df['predicted_GTI']:
    charge_in = gti * conversion_factor * charge_efficiency
    net = charge_in - load_watts
    level += net * 0.05  # simple factor to simulate charging rate
    level = max(0, min(max_capacity, level))
    battery.append(level)

df['battery_level'] = battery

# Line plot
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df['timestamp'], df['battery_level'], label='Battery Level (%)', color='green')
ax2.set_xlabel('Time')
ax2.set_ylabel('Battery Level (%)')
ax2.set_title('Battery Level Simulation Over Time')
ax2.grid(True)
st.pyplot(fig2)
