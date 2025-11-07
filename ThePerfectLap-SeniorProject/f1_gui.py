import tkinter as tk
from tkinter import ttk, messagebox
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fastf1.Cache.enable_cache("cache")

def load_session(year, event, ses_type):
    s = fastf1.get_session(int(year), event, ses_type)
    s.load(laps=True, telemetry=True)
    return s

def compute_optimized_secs(laps_df):
    # “optimized” = sum of min sector times for that driver in this session
    mins = []
    for col in ("Sector1Time","Sector2Time","Sector3Time"):
        if col in laps_df.columns:
            t = laps_df[col].min()
            if pd.notna(t):
                mins.append(t.total_seconds())
    return sum(mins) if mins else None

def analyze():
    year = e_year.get().strip()
    event = e_event.get().strip()
    ses_type = cb_sess.get().strip()
    drv = e_driver.get().strip().upper() or None
    try:
        ses = load_session(year, event, ses_type)
    except Exception as ex:
        messagebox.showerror("Load error", str(ex)); return

    laps = ses.laps if drv is None else ses.laps.pick_driver(drv)
    if laps.empty:
        messagebox.showwarning("No data", "No laps found for that selection.")
        return

    fastest = laps.pick_fastest()
    if pd.isna(fastest["LapTime"]):
        messagebox.showwarning("No time", "Fastest lap has no LapTime."); return

    act_secs = fastest["LapTime"].total_seconds()
    opt_secs = compute_optimized_secs(laps)
    if opt_secs is None:
        messagebox.showwarning("No sectors", "Could not compute sector-based optimum."); return

    # clear previous plots
    for w in plot_frame.winfo_children():
        w.destroy()

    # bar chart: actual vs optimized
    fig1, ax1 = plt.subplots(figsize=(5,3))
    ax1.bar(["Actual", "Optimized"], [act_secs, opt_secs])
    ax1.set_ylabel("Seconds")
    ax1.set_title(f"{(drv or fastest['Driver'])} – {event} {year} ({ses_type})")
    fig1.tight_layout()
    c1 = FigureCanvasTkAgg(fig1, master=plot_frame)
    c1.draw(); c1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

    # racing line (if telemetry has X/Y)
    try:
        tel = fastest.get_telemetry()
        if {"X","Y"}.issubset(tel.columns):
            fig2, ax2 = plt.subplots(figsize=(5,3.8))
            ax2.plot(tel["X"], tel["Y"])
            ax2.set_title("Racing Line (Fastest Lap)")
            ax2.set_xlabel("X"); ax2.set_ylabel("Y")
            ax2.axis("equal")
            fig2.tight_layout()
            c2 = FigureCanvasTkAgg(fig2, master=plot_frame)
            c2.draw(); c2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
        else:
            messagebox.showinfo("Note", "XY telemetry not available for this session.")
    except Exception as ex:
        messagebox.showinfo("Telemetry", f"Could not draw racing line:\n{ex}")

# ---- GUI ----
root = tk.Tk()
root.title("F1 Lap Optimizer (FastF1)")
root.geometry("1200x650")

tk.Label(root, text="Year").grid(row=0, column=0, sticky="e", padx=6, pady=4)
e_year = tk.Entry(root, width=8); e_year.insert(0,"2024"); e_year.grid(row=0, column=1, sticky="w")

tk.Label(root, text="Event").grid(row=1, column=0, sticky="e", padx=6, pady=4)
e_event = tk.Entry(root, width=28); e_event.insert(0,"Bahrain Grand Prix"); e_event.grid(row=1, column=1, sticky="w")

tk.Label(root, text="Session").grid(row=2, column=0, sticky="e", padx=6, pady=4)
cb_sess = ttk.Combobox(root, width=6, values=["FP1","FP2","FP3","Q","R"]); cb_sess.set("Q"); cb_sess.grid(row=2, column=1, sticky="w")

tk.Label(root, text="Driver (optional, e.g. VER)").grid(row=3, column=0, sticky="e", padx=6, pady=4)
e_driver = tk.Entry(root, width=8); e_driver.grid(row=3, column=1, sticky="w")

ttk.Button(root, text="Analyze", command=analyze).grid(row=4, column=0, columnspan=2, pady=8)

plot_frame = tk.Frame(root); plot_frame.grid(row=5, column=0, columnspan=6, padx=10, pady=10, sticky="nsew")
root.grid_rowconfigure(5, weight=1); root.grid_columnconfigure(5, weight=1)

root.mainloop()
