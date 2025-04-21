import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/energy_log.csv', parse_dates=['timestamp'])

# Feature: Hour of the day
df['hour'] = df['timestamp'].dt.hour
X = df[['hour']]
y = df['solar_input']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Hour of Day')
plt.ylabel('Solar Input (W)')
plt.title('Actual vs Predicted Solar Input')
plt.legend()
plt.grid(True)
plt.show()
