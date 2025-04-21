import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title='Solar Forecast Dashboard', layout='wide')
st.title("☀️ Solar Irradiance Forecast (PVGIS Data)")

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

# Feature selection and model training
X = df[['hour', 'month', 'temperature']]
y = df['GTI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display MSE
st.metric(label="Model Mean Squared Error", value=f"{mse:.2f}")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.set_xlabel('Actual GTI (W/m²)')
ax.set_ylabel('Predicted GTI (W/m²)')
ax.set_title('Actual vs Predicted Solar Irradiance (GTI)')
ax.grid(True)
st.pyplot(fig)
