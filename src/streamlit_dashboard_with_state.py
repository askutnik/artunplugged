import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pvgis_fetcher import get_pvgis_data

st.set_page_config(page_title='Solar Forecast + Planning Tool', layout='wide')
st.title("☀️ Solar Forecast Dashboard + 🔋 Battery Simulation + 🔮 Future Planner")

# Tabs for different modes
tab1, tab2 = st.tabs(["📡 Live PVGIS Forecast", "🔮 Plan Ahead Forecast"])

# -----------------------------
# 📡 TAB 1: Fetch PVGIS Data
# -----------------------------
with tab1:
    st.sidebar.header("📍 PVGIS Input Parameters")
    lat = st.sidebar.number_input("Latitude", value=51.17, format="%.5f")
    lon = st.sidebar.number_input("Longitude", value=-0.83, format="%.5f")
    angle = st.sidebar.slider("Tilt Angle (°)", min_value=0, max_value=90, value=35)
    aspect = st.sidebar.slider("Azimuth (°)", min_value=-180, max_value=180, value=180)
    year = st.sidebar.selectbox("Year", options=list(range(2005, 2021)), index=15)

    if st.sidebar.button("Fetch PVGIS Data"):
        with st.spinner("Fetching PVGIS data..."):
            df = get_pvgis_data(lat, lon, angle, aspect, year)

        st.success("Data fetched successfully!")
        st.dataframe(df)

        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        X = df[['hour', 'month', 'temperature']]
        y = df['GTI']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        st.metric(label="Model Mean Squared Error", value=f"{mse:.2f}")

        # Store model in session state
        st.session_state.model = model

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.set_xlabel("Actual GTI (W/m²)")
        ax1.set_ylabel("Predicted GTI (W/m²)")
        ax1.set_title("Actual vs Predicted Solar Irradiance (GTI)")
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("🔋 Battery Simulation")

        df['predicted_GTI'] = model.predict(X)
        battery = []
        level = 50
        for gti in df['predicted_GTI']:
            charge_in = gti * 0.1 * 0.9
            net = charge_in - 10
            level += net * 0.05
            level = max(0, min(100, level))
            battery.append(level)
        df['battery_level'] = battery

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df['timestamp'], df['battery_level'], color='green')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Battery Level (%)")
        ax2.set_title("Battery Level Simulation Over Time")
        ax2.grid(True)
        st.pyplot(fig2)

# -----------------------------
# 🔮 TAB 2: Forecast Ahead
# -----------------------------
with tab2:
    st.subheader("📅 Forecast Solar Irradiance for a Future Time")
    st.markdown("Use your trained model to estimate GTI for a hypothetical time.")

    month = st.selectbox("Month", list(range(1, 13)), index=5)
    hour = st.slider("Hour of Day", 0, 23, 12)
    temp = st.number_input("Expected Temperature (°C)", value=18.0)

    model = st.session_state.get("model", None)

    if st.button("Predict Irradiance"):
        if model is None:
            st.warning("⚠️ Please train the model first by fetching PVGIS data in the first tab.")
        else:
            X_future = pd.DataFrame([[hour, month, temp]], columns=["hour", "month", "temperature"])
            predicted_gti = model.predict(X_future)[0]
            st.success(f"🔮 Predicted GTI: {predicted_gti:.2f} W/m²")

            sim_hours = 12
            battery = []
            level = 50
            for _ in range(sim_hours):
                charge_in = predicted_gti * 0.1 * 0.9
                net = charge_in - 10
                level += net * 0.05
                level = max(0, min(100, level))
                battery.append(level)

            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.plot(range(sim_hours), battery, color='purple')
            ax3.set_xlabel("Simulated Hours")
            ax3.set_ylabel("Battery Level (%)")
            ax3.set_title("🔋 Battery Level Forecast Over 12 Hours")
            ax3.grid(True)
            st.pyplot(fig3)
