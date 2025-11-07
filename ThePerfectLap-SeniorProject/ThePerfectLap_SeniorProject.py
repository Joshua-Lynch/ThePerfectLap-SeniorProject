import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import os

#Handles caching so data is not re-downloaded each time
cache_dir = os.path.join(os.getcwd(), "cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Test Run: Plotting best lap times from Monaco GP 2024 Qualifying
fastf1.Cache.enable_cache("cache")
year, event, session = 2024, "Monaco Grand Prix", "Q"
print("Loading session data...")
session = fastf1.get_session(year, event, session)
session.load(laps=True, telemetry=True)

# Plot best lap times for all drivers in qualifying
laps = session.laps
best_lap = laps.groupby("Driver")["LapTime"].min().dropna().sort_values()
drivers = list(best_lap.index)
times = best_lap.dt.total_seconds()

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(drivers, times, color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Lap Time (seconds)")
plt.title(f"{event} {year} - Qualifying Best Lap Times")
plt.tight_layout()
plt.show()


