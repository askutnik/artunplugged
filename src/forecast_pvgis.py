import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the cleaned PVGIS dataset
df = pd.read_csv('data/pvgis_cleaned.csv', parse_dates=['timestamp'])

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month

# Features and target
X = df[['hour', 'month', 'temperature']]
y = df['GTI']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual GTI (W/m²)')
plt.ylabel('Predicted GTI (W/m²)')
plt.title('Actual vs Predicted Solar Irradiance (GTI)')
plt.grid(True)
plt.tight_layout()
plt.show()
