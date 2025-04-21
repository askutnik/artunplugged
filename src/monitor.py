import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/energy_log.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

plt.plot(data['timestamp'], data['solar_input'], label='Solar Input (W)')
plt.plot(data['timestamp'], data['power_usage'], label='Power Usage (W)')
plt.xlabel('Time')
plt.ylabel('Watts')
plt.legend()
plt.title('Solar Power vs Consumption')
plt.grid(True)
plt.show()