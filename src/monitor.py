import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/energy_log.csv', parse_dates=['timestamp'])

# Plot 1: Solar Input vs Power Usage
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['solar_input'], label='Solar Input (W)', linewidth=2)
plt.plot(df['timestamp'], df['power_usage'], label='Power Usage (W)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Watts')
plt.title('Solar Input vs Power Usage')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Battery Level Over Time
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['battery_level'], color='green', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Battery Level (%)')
plt.title('Battery Level Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
