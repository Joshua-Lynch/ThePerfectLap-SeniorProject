import os
import tkinter as tk
from tkinter import ttk, messagebox
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------- Setup and Cache --------------------
cache_dir = os.path.join(os.getcwd(), "cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


# -------------------- Core Functions --------------------
def load_session(year, event, session_type):
    """Load the specified F1 session."""
    session = fastf1.get_session(int(year), event, session_type)
    session.load(laps=True, telemetry=True)
    return session


def get_driver_fastest_lap(session, driver):
    """Get fastest lap and telemetry for a driver."""
    lap = session.laps.pick_driver(driver).pick_fastest()
    tel = lap.get_telemetry().add_distance()
    return lap, tel


def compare_drivers(year, event, session_type, driver1, driver2):
    """Return dataframes, telemetry, and stats for both drivers."""
    session = load_session(year, event, session_type)
    lap1, tel1 = get_driver_fastest_lap(session, driver1)
    lap2, tel2 = get_driver_fastest_lap(session, driver2)

    stats = pd.DataFrame({
        "Driver": [driver1, driver2],
        "Lap Time (s)": [
            lap1["LapTime"].total_seconds(),
            lap2["LapTime"].total_seconds()
        ],
        "Sector 1 (s)": [
            lap1["Sector1Time"].total_seconds(),
            lap2["Sector1Time"].total_seconds()
        ],
        "Sector 2 (s)": [
            lap1["Sector2Time"].total_seconds(),
            lap2["Sector2Time"].total_seconds()
        ],
        "Sector 3 (s)": [
            lap1["Sector3Time"].total_seconds(),
            lap2["Sector3Time"].total_seconds()
        ]
    })
    delta = stats.iloc[0]["Lap Time (s)"] - stats.iloc[1]["Lap Time (s)"]
    return tel1, tel2, stats, delta


def plot_comparison(tel1, tel2, driver1, driver2, event, year):
    """Return a Matplotlib figure with both racing lines and speed trace."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))

    # Racing Line
    ax1.plot(tel1["X"], tel1["Y"], color="blue", label=driver1, linewidth=2)
    ax1.plot(tel2["X"], tel2["Y"], color="red", label=driver2, linewidth=2, alpha=0.8)
    ax1.set_title(f"{event} {year} – Racing Line Comparison")
    ax1.legend()
    ax1.axis("equal")

    # Speed Trace
    ax2.plot(tel1["Distance"], tel1["Speed"], color="blue", label=driver1)
    ax2.plot(tel2["Distance"], tel2["Speed"], color="red", label=driver2)
    ax2.set_title("Speed vs Distance")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Speed (km/h)")
    ax2.legend()

    fig.tight_layout()
    return fig


# -------------------- GUI Logic --------------------
def analyze_drivers():
    year = e_year.get().strip()
    event = e_event.get().strip()
    session_type = cb_sess.get().strip()
    driver1 = e_driver1.get().strip().upper()
    driver2 = e_driver2.get().strip().upper()

    if not (year and event and session_type and driver1 and driver2):
        messagebox.showwarning("Missing Info", "Please fill in all fields.")
        return

    try:
        tel1, tel2, stats, delta = compare_drivers(
            year, event, session_type, driver1, driver2
        )
    except Exception as e:
        messagebox.showerror("Error", f"Data load failed:\n{e}")
        return

    # Clear old plots
    for w in frame_plot.winfo_children():
        w.destroy()

    # Create plot
    fig = plot_comparison(tel1, tel2, driver1, driver2, event, year)
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Update table
    for i in tree.get_children():
        tree.delete(i)
    for _, row in stats.iterrows():
        tree.insert("", "end", values=list(row))
    lbl_delta.config(
        text=f"Lap Time Δ ({driver1} - {driver2}) = {delta:.3f}s",
        fg="green" if delta < 0 else "red"
    )


# -------------------- GUI Layout --------------------
root = tk.Tk()
root.title("F1 Driver Comparison Analyzer")
root.geometry("1200x750")

# Top Controls
tk.Label(root, text="Year:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
e_year = tk.Entry(root, width=6)
e_year.insert(0, "2024")
e_year.grid(row=0, column=1, sticky="w")

tk.Label(root, text="Event:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
e_event = tk.Entry(root, width=25)
e_event.insert(0, "Monaco Grand Prix")
e_event.grid(row=1, column=1, sticky="w")

tk.Label(root, text="Session:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
cb_sess = ttk.Combobox(root, values=["FP1", "FP2", "FP3", "Q", "R"], width=6)
cb_sess.set("Q")
cb_sess.grid(row=2, column=1, sticky="w")

tk.Label(root, text="Driver 1:").grid(row=0, column=2, sticky="e", padx=5)
e_driver1 = tk.Entry(root, width=8)
e_driver1.insert(0, "VER")
e_driver1.grid(row=0, column=3, sticky="w")

tk.Label(root, text="Driver 2:").grid(row=1, column=2, sticky="e", padx=5)
e_driver2 = tk.Entry(root, width=8)
e_driver2.insert(0, "LEC")
e_driver2.grid(row=1, column=3, sticky="w")

tk.Button(
    root,
    text="Compare Drivers",
    command=analyze_drivers,
    bg="#4CAF50",
    fg="white",
    font=("Segoe UI", 10, "bold"),
).grid(row=2, column=2, columnspan=2, pady=5)

# Plot Frame
frame_plot = tk.Frame(root, bg="#f2f2f2", relief="ridge", borderwidth=2)
frame_plot.grid(row=4, column=0, columnspan=6, sticky="nsew", padx=10, pady=10)
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(5, weight=1)

# Stats Table
cols = ("Driver", "Lap Time (s)", "Sector 1 (s)", "Sector 2 (s)", "Sector 3 (s)")
tree = ttk.Treeview(root, columns=cols, show="headings", height=3)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=150)
tree.grid(row=5, column=0, columnspan=6, padx=10, pady=5)

# Lap delta label
lbl_delta = tk.Label(root, text="Lap Time Δ: ", font=("Segoe UI", 10, "bold"))
lbl_delta.grid(row=6, column=0, columnspan=6, pady=8)

root.mainloop()
