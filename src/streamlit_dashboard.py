import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pvgis_fetch import get_pvgis_data
from geocode_location import geocode_location
from get_temps import get_temps

st.set_page_config(page_title='Art, Unplugged', layout='wide')
st.title("Art, Unplugged: Energy Production Visualisation and Forecasting")

tab1, tab2 = st.tabs(["PVGIS Data", "Forecast"])


# TAB 1: PVGIS Data
with tab1:
    st.header("Input Parameters")
    place_name = st.text_input("Location", value="Alice Holt Forest")
    if place_name:
        lat, lon = geocode_location(place_name)
        if lat and lon:
            st.success(f"Found coordinates: {lat:.4f}, {lon:.4f}")
        else:
            st.warning("Could not find location- try a more specific name")
    angle = st.slider("Tilt Angle", min_value=0, max_value=90, value=35)
    aspect = st.slider("Azimuth", min_value=-180, max_value=180, value=180)
    start_year = 2015
    end_year = 2020

    if st.button("Fetch PVGIS Data"):
        if end_year < start_year:
            st.warning("End year must be equal to or greater than start year.")
        else:
            from pvgis_fetch import get_pvgis_data
            with st.spinner("Fetching data..."):
                df = get_pvgis_data(lat, lon, angle, aspect, start_year, end_year)

            st.success(f"Data fetched for years {start_year}–{end_year}")

            st.subheader("Hourly Solar Irradiance")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(df['timestamp'], df['GTI'], color='skyblue')
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Global Tilted Irradiance (W/m²)")
            ax2.set_title("Hourly Solar Irradiance")
            ax2.grid(True)
            st.pyplot(fig2)

            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month

            X = df[['hour', 'month', 'temperature']]
            y = df['GTI']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # model logic
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)

            st.metric(label="Model R² Score", value=f"{r2:.3f}")

            st.session_state.model = model

            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(y_test, y_pred, alpha=0.5)
            ax1.set_xlabel("Actual GTI (W/m²)")
            ax1.set_ylabel("Predicted GTI (W/m²)")
            ax1.set_title("Actual vs Predicted Solar Irradiance (GTI)")
            ax1.grid(True)
            st.pyplot(fig1)

            


# TAB 2: Forecast 
with tab2:
    st.subheader("Forecast Solar Irradiance")
    st.markdown("Predict GTI and battery levels across a full 12-hour daytime window.")

    # Inputs
    month = st.selectbox("Month", list(range(1, 13)), index=5)
    battery_capacity_ah = st.number_input("Battery Capacity (Ah)", min_value=10, max_value=1000, value=100)
    system_voltage = st.number_input("System Voltage (V)", min_value=6, max_value=48, value=12)
    load_power_watts = st.number_input("Load Power (Watt)", min_value=1, max_value=500, value=10)

    model = st.session_state.get("model", None)

    if st.button("Forecast Day"):
        if model is None:
            st.warning("Please populate the model first by fetching PVGIS data in the first tab.")
        else:
            hours = list(range(8, 20))  # 8:00 to 19:00
            lat = st.session_state.get("latitude", 51.17)
            lon = st.session_state.get("longitude", -0.83)
            temps = get_temps(lat, lon, month=month, year=2023)
            if temps and len(temps) >= 20:
                forecast_temps = temps[8:20]
                gtis = [model.predict([[hour, month, temp]])[0] for hour, temp in zip(hours, forecast_temps)]

                battery_capacity_wh = battery_capacity_ah * system_voltage
                current_wh = 0
                battery_wh_series = []

                for gti in gtis:
                    charge_wh = gti * 0.1 * 0.9
                    net_wh = charge_wh - load_power_watts
                    current_wh += net_wh
                    current_wh = max(0, min(battery_capacity_wh, current_wh))
                    battery_wh_series.append(current_wh)

                ah_series = [round(wh / system_voltage, 2) for wh in battery_wh_series]
                battery_pct = [round(wh / battery_capacity_wh * 100, 2) for wh in battery_wh_series]

                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.set_xlabel("Hour")
                ax1.set_ylabel("Predicted GTI (W/m²)", color='tab:blue')
                ax1.plot(hours, gtis, color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')

                ax2 = ax1.twinx()
                ax2.set_ylabel("Battery Charge (%)", color='tab:orange')
                ax2.plot(hours, battery_pct, color='tab:orange')
                ax2.tick_params(axis='y', labelcolor='tab:orange')

                fig.suptitle("Forecasted GTI and Battery Charge")
                fig.tight_layout()
                st.pyplot(fig)

            else:
                st.error("Could not fetch temperature data. Try again later or check your Internet connection.")
