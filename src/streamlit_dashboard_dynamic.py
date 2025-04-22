import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pvgis_fetcher import get_pvgis_data

st.set_page_config(page_title='Dynamic Solar Forecast Dashboard', layout='wide')
st.title("☀️ Dynamic Solar Irradiance Forecast + 🔋 Battery Simulation")

# Sidebar input for PVGIS parameters
st.sidebar.header("📍 PVGIS Input Parameters")
lat = st.sidebar.number_input("Latitude", value=51.17, format="%.5f")
lon = st.sidebar.number_input("Longitude", value=-0.83, format="%.5f")
angle = st.sidebar.slider("Tilt Angle (°)", min_value=0, max_value=90, value=35)
aspect = st.sidebar.slider("Azimuth (°)", min_value=-180, max_value=180, value=180)
year = st.sidebar.selectbox("Year", options=list(range(2018, 2023)), index=3)

# Button to fetch data
if st.sidebar.button("Fetch PVGIS Data"):
    with st.spinner("Fetching PVGIS data..."):
        df = get_pvgis_data(lat, lon, angle, aspect, year)

    st.success("Data fetched successfully!")
    st.write(df.head())

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    # Model training and prediction
    X = df[['hour', 'month', 'temperature']]
    y = df['GTI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    st.metric(label="Model Mean Squared Error", value=f"{mse:.2f}")

    # Actual vs predicted GTI
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.set_xlabel("Actual GTI (W/m²)")
    ax1.set_ylabel("Predicted GTI (W/m²)")
    ax1.set_title("Actual vs Predicted Solar Irradiance (GTI)")
    ax1.grid(True)
    st.pyplot(fig1)

    # Battery simulation
    st.subheader("🔋 Battery Simulation")

    df['predicted_GTI'] = model.predict(X)
    battery = []
    level = 50
    max_capacity = 100
    load_watts = 10
    charge_efficiency = 0.9
    conversion_factor = 0.1

    for gti in df['predicted_GTI']:
        charge_in = gti * conversion_factor * charge_efficiency
        net = charge_in - load_watts
        level += net * 0.05
        level = max(0, min(max_capacity, level))
        battery.append(level)

    df['battery_level'] = battery

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df['timestamp'], df['battery_level'], color='green')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Battery Level (%)")
    ax2.set_title("Battery Level Simulation Over Time")
    ax2.grid(True)
    st.pyplot(fig2)
