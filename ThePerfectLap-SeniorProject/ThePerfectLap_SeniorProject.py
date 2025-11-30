import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

import fastf1
import fastf1.plotting  # optional, but harmless

# -----------------------------
# Cache & Matplotlib style
# -----------------------------
CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

plt.rcParams.update({
    "axes.facecolor": "#0d0d0d",
    "figure.facecolor": "#0d0d0d",
    "savefig.facecolor": "#0d0d0d",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#333333",
    "grid.color": "#333333"
})


# ============================================
# PHYSICS / MINIMUM-TIME OPTION A (core pieces)
# ============================================

def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Numeric curvature kappa(s) for a polyline (x(s), y(s))."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


def smooth_path_xy(x, y, window=31, poly=3):
    """Simple path smoothing on XY to reduce curvature spikes."""
    if len(x) < window + 2 or len(y) < window + 2:
        return x.copy(), y.copy()
    xs = savgol_filter(x, window, poly, mode="interp")
    ys = savgol_filter(y, window, poly, mode="interp")
    return xs, ys


def compute_vmax_lateral(kappa: np.ndarray, params: dict) -> np.ndarray:
    """
    Cornering limit from friction circle with aerodynamic load.
    a_lat_max = mu * (g + 0.5*rho*ClA/m * v^2)
    Solve for v in v^2 * |kappa| <= a_lat_max.
    """
    m = params["mass_kg"]
    mu = params["mu"]
    rho = params["air_density"]
    ClA = params["ClA"]  # effective downforce coefficient (Cl*A)

    k = 0.5 * rho * ClA / m
    kappa_abs = np.abs(kappa) + 1e-12

    denom = (kappa_abs - mu * k)
    v2 = np.where(
        denom > 1e-9,
        (mu * 9.81) / denom,
        params["vmax_straight"] ** 2
    )
    vmax = np.sqrt(np.clip(v2, 1.0, params["vmax_straight"] ** 2))
    return vmax


def forward_backward_speed_profile(s: np.ndarray,
                                   v_lat_cap: np.ndarray,
                                   params: dict) -> np.ndarray:
    """
    Classical forward-backward pass to enforce longitudinal accel/brake limits.
    """
    a_pos = params["a_long_accel"]
    a_neg = params["a_long_brake"]

    v = v_lat_cap.copy()

    # Forward pass (acceleration)
    for i in range(1, len(s)):
        ds = max(s[i] - s[i - 1], 1e-6)
        v_allowed = np.sqrt(v[i - 1] ** 2 + 2 * a_pos * ds)
        if v_allowed < v[i]:
            v[i] = v_allowed

    # Backward pass (braking)
    for i in range(len(s) - 2, -1, -1):
        ds = max(s[i + 1] - s[i], 1e-6)
        v_allowed = np.sqrt(v[i + 1] ** 2 + 2 * a_neg * ds)
        if v_allowed < v[i]:
            v[i] = v_allowed

    return v


def integrate_time(distance: np.ndarray, v: np.ndarray) -> (float, np.ndarray):
    """
    Integrate lap time and cumulative time along distance using trapezoidal rule.
    """
    v_clipped = np.clip(v, 1.0, None)
    dt = 2.0 * np.diff(distance) / (v_clipped[1:] + v_clipped[:-1])
    cum_t = np.concatenate([[0.0], np.cumsum(dt)])
    return float(cum_t[-1]), cum_t


def build_optimized_line_from_baseline(tel_df: pd.DataFrame,
                                       params: dict) -> pd.DataFrame:
    """
    Compute a min-time speed profile on a smoothed baseline line.
    tel_df is telemetry from get_telemetry().add_distance() (has X,Y,Distance,Speed).
    """
    s = np.asarray(tel_df["Distance"].values, dtype=float)
    x = np.asarray(tel_df["X"].values, dtype=float)
    y = np.asarray(tel_df["Y"].values, dtype=float)

    # Smooth path
    xs, ys = smooth_path_xy(x, y, window=params["smooth_window"], poly=3)

    # Curvature & lateral limit speed
    kappa = compute_curvature(xs, ys)
    vmax_lat = compute_vmax_lateral(kappa, params)

    # Longitudinal limits
    v_profile = forward_backward_speed_profile(s, vmax_lat, params)

    return pd.DataFrame({
        "Distance": s,
        "X": xs,
        "Y": ys,
        "Speed_opt": v_profile
    })


def suggest_optimal_setup(track_stats: dict) -> dict:
    """Heuristic car-setup suggestions from track characteristics (PoC level)."""
    avg_kappa = track_stats.get("avg_kappa", 0.0015)
    v_top = track_stats.get("top_speed", 310.0)
    brake_g = track_stats.get("brake_g", 3.5)

    if avg_kappa > 0.0020:
        fw, rw = 7, 12
    elif v_top > 320:
        fw, rw = 5, 9
    else:
        fw, rw = 6, 11

    ride_front = 28 if brake_g > 4.0 else 22
    ride_rear = ride_front + 5

    spring = "Medium-Stiff" if v_top > 320 else "Medium"
    arb = "Stiff" if avg_kappa < 0.0015 else "Medium"

    diff_on = 65 if avg_kappa > 0.0020 else 55
    diff_off = 45 if avg_kappa > 0.0020 else 40

    return {
        "Aero (FW/RW)": f"{fw}/{rw}",
        "Ride Height (F/R mm)": f"{ride_front} / {ride_rear}",
        "Springs": spring,
        "Anti-Roll Bars": arb,
        "Differential (On/Off %)": f"{diff_on}% / {diff_off}%"
    }


# ============================================
# DATA FETCH (FastF1)
# ============================================

def load_baseline_fastest_lap(year: int, event: str, session_code: str):
    """Loads session, returns (session_obj, fastest_lap_obj, telemetry_df_with_XY)."""
    session = fastf1.get_session(int(year), event, session_code)
    session.load(laps=True, telemetry=True)

    fastest_lap = session.laps.pick_fastest()
    tel = fastest_lap.get_telemetry().add_distance()  # has Distance, Speed, X, Y, etc.

    return session, fastest_lap, tel


def extract_world_record_metadata(session):
    """Metadata for the fastest lap within the session."""
    fastest_lap = session.laps.pick_fastest()
    driver_info = session.get_driver(fastest_lap["Driver"])

    driver_full_name = driver_info.get("FullName", "Unknown")
    driver_code = fastest_lap["Driver"]
    team_name = fastest_lap["Team"]
    car_number = fastest_lap["DriverNumber"]

    lap_time = fastest_lap["LapTime"]
    compound = fastest_lap["Compound"]
    tyre_life = fastest_lap["TyreLife"]
    track_status = fastest_lap["TrackStatus"]
    stint = fastest_lap["Stint"]

    info = {
        "Driver Name": driver_full_name,
        "Driver Code": driver_code,
        "Team": team_name,
        "Car Number": car_number,
        "Lap Time": lap_time,
        "Tyre Compound": compound,
        "Tyre Life (laps)": tyre_life,
        "Track Status": track_status,
        "Session Type": session.session_info.get("Name", ""),
        "Stint": stint
    }
    return info


def extract_car_telemetry_specs(session, fastest_lap):
    """
    Telemetry-based car performance metrics for the world record lap.
    IMPORTANT: resample X/Y onto the car-data distance grid so shapes match.
    """
    # Car data: Speed, Throttle, Brake, Gear, RPM, Distance
    car = fastest_lap.get_car_data().add_distance()
    speed = car["Speed"].to_numpy()        # km/h
    throttle = car["Throttle"].to_numpy()
    brake = car["Brake"].to_numpy()
    gear = car["nGear"].to_numpy()
    rpm = car["RPM"].to_numpy()
    s_car = car["Distance"].to_numpy()

    # Position / telemetry (higher rate): X, Y, Distance
    pos = fastest_lap.get_telemetry().add_distance()
    s_pos = pos["Distance"].to_numpy()
    x_pos = pos["X"].to_numpy()
    y_pos = pos["Y"].to_numpy()

    # Resample X,Y onto the car-data distance grid
    x = np.interp(s_car, s_pos, x_pos)
    y = np.interp(s_car, s_pos, y_pos)

    # Curvature and lateral G on same grid as speed
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    kappa = (dx * ddy - dy * ddx) / denom

    v_mps = speed / 3.6
    a_lat = v_mps ** 2 * np.abs(kappa)
    g_force = a_lat / 9.81

    specs = {
        "Top Speed (km/h)": float(np.nanmax(speed)),
        "Average Speed (km/h)": float(np.nanmean(speed)),
        "Max RPM": float(np.nanmax(rpm)),
        "Max Gear": int(np.nanmax(gear)),
        "Max Brake (%)": float(np.nanmax(brake)),
        "Avg Throttle (%)": float(np.nanmean(throttle)),
        "Max Lateral G": float(np.nanmax(g_force)),
        "Sample Count": int(len(speed))
    }
    return specs


def populate_world_record_panel(panel_widget, metadata, specs):
    """Write the World Record driver + car spec sheet into a Tkinter Text()."""
    panel_widget.configure(state="normal")
    panel_widget.delete("1.0", tk.END)

    panel_widget.insert(tk.END, "🏎 WORLD RECORD LAP CAR PROFILE\n", "header")
    panel_widget.insert(tk.END, "────────────────────────────────────\n", "divider")

    panel_widget.insert(
        tk.END,
        f"Driver: {metadata['Driver Name']} ({metadata['Driver Code']})\n",
        "driver"
    )
    panel_widget.insert(tk.END, f"Team: {metadata['Team']}\n", "team")
    panel_widget.insert(tk.END, f"Car Number: {metadata['Car Number']}\n\n", "normal")

    panel_widget.insert(tk.END, f"Lap Time: {metadata['Lap Time']}\n", "highlight")
    panel_widget.insert(tk.END, f"Session: {metadata['Session Type']}\n", "normal")
    panel_widget.insert(tk.END, f"Track Status: {metadata['Track Status']}\n", "normal")
    panel_widget.insert(
        tk.END,
        f"Tyre: {metadata['Tyre Compound']}  |  Life: {metadata['Tyre Life (laps)']} laps\n\n",
        "normal"
    )

    panel_widget.insert(tk.END, "📊 CAR PERFORMANCE SPECS\n", "header2")
    panel_widget.insert(tk.END, "────────────────────────────────────\n", "divider")

    for k, v in specs.items():
        panel_widget.insert(tk.END, f"{k}: {v}\n", "normal")

    panel_widget.configure(state="disabled")


# ============================================
# GUI (Tkinter + Matplotlib)
# ============================================

class F1MinTimeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("F1 Minimum-Time Optimizer — World Record vs AI (PoC)")
        self.geometry("1380x840")
        self.configure(bg="#0d0d0d")

        # --- Controls (top) ---
        ctrl = tk.Frame(self, bg="#0d0d0d")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        lbl_style = {"bg": "#0d0d0d", "fg": "white"}
        entry_bg = {"bg": "#1a1a1a", "fg": "white", "insertbackground": "white"}

        tk.Label(ctrl, text="Year", **lbl_style).grid(row=0, column=0, sticky="e", padx=6)
        self.ent_year = tk.Entry(ctrl, width=6, **entry_bg)
        self.ent_year.insert(0, "2024")
        self.ent_year.grid(row=0, column=1, sticky="w")

        tk.Label(ctrl, text="Event", **lbl_style).grid(row=0, column=2, sticky="e", padx=6)
        self.ent_event = tk.Entry(ctrl, width=26, **entry_bg)
        self.ent_event.insert(0, "Bahrain Grand Prix")
        self.ent_event.grid(row=0, column=3, sticky="w")

        tk.Label(ctrl, text="Session", **lbl_style).grid(row=0, column=4, sticky="e", padx=6)
        self.cb_session = ttk.Combobox(ctrl, values=["Q", "R", "FP1", "FP2", "FP3"], width=6)
        self.cb_session.set("Q")
        self.cb_session.grid(row=0, column=5, sticky="w")

        self.btn_go = tk.Button(
            ctrl, text="Optimize & Compare",
            command=self.run_optimization,
            bg="#bb0000", fg="white", activebackground="#ff3333"
        )
        self.btn_go.grid(row=0, column=6, padx=10)

        # --- Center: plots area (matplotlib) ---
        self.fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        gs = GridSpec(
            2, 2, figure=self.fig,
            height_ratios=[3.0, 1.4],
            width_ratios=[3.0, 1.2],
            hspace=0.25, wspace=0.25
        )

        self.ax_track = self.fig.add_subplot(gs[0, 0])
        self.ax_delta = self.fig.add_subplot(gs[1, 0])
        self.ax_setup = self.fig.add_subplot(gs[:, 1])
        self.ax_setup.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Sidebar panel ---
        self.side = tk.Frame(self, bg="#0d0d0d")
        self.side.pack(side=tk.RIGHT, fill=tk.Y, padx=8)

        # World record info panel
        tk.Label(
            self.side,
            text="World Record Car & Driver",
            font=("Segoe UI", 12, "bold"),
            bg="#0d0d0d", fg="#ffcc00"
        ).pack(anchor="w", pady=(8, 4))

        self.txt_worldrecord = tk.Text(
            self.side, width=34, height=18,
            bg="#1a1a1a", fg="white",
            insertbackground="white", relief="flat"
        )
        self.txt_worldrecord.pack(fill=tk.X, padx=2)

        # Configure tags for pretty styling
        self.txt_worldrecord.tag_configure("header", foreground="#ffcc00", font=("Segoe UI", 13, "bold"))
        self.txt_worldrecord.tag_configure("header2", foreground="#00ffff", font=("Segoe UI", 12, "bold"))
        self.txt_worldrecord.tag_configure("driver", foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        self.txt_worldrecord.tag_configure("team", foreground="#ff4444")
        self.txt_worldrecord.tag_configure("highlight", foreground="#00ff88", font=("Consolas", 11, "bold"))
        self.txt_worldrecord.tag_configure("divider", foreground="#555555")
        self.txt_worldrecord.tag_configure("normal", foreground="white")

        # Optimal setup panel
        tk.Label(
            self.side,
            text="Optimal Car Setup (PoC)",
            font=("Segoe UI", 12, "bold"),
            bg="#0d0d0d", fg="#00ffff"
        ).pack(anchor="w", pady=(10, 4))

        self.txt_setup = tk.Text(
            self.side, width=34, height=18,
            bg="#1a1a1a", fg="white",
            insertbackground="white", relief="flat"
        )
        self.txt_setup.pack(fill=tk.X, padx=2)

        # Stats table (lap & sectors)
        tk.Label(
            self.side,
            text="Lap & Sectors",
            font=("Segoe UI", 11, "bold"),
            bg="#0d0d0d", fg="#ffcc00"
        ).pack(anchor="w", pady=(10, 2))

        cols = ("Label", "S1 (s)", "S2 (s)", "S3 (s)", "Lap (s)")
        self.tree = ttk.Treeview(self.side, columns=cols, show="headings", height=4)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=85)
        self.tree.pack(fill=tk.X, pady=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1a1a1a",
            fieldbackground="#1a1a1a",
            foreground="white",
            rowheight=22
        )
        style.configure("Treeview.Heading", background="#333333", foreground="white")

        self.status = tk.Label(self.side, text="", bg="#0d0d0d", fg="white")
        self.status.pack(anchor="w", pady=6)

        self._clear_plots()

    # -------------
    # GUI helpers
    # -------------
    def _clear_plots(self):
        self.ax_track.clear()
        self.ax_delta.clear()
        self.ax_setup.clear()
        self.ax_setup.set_axis_off()

        self.ax_track.set_title(
            "Racing Line — World Record (Yellow) vs Optimized (Cyan)",
            color="white"
        )
        self.ax_track.set_aspect("equal", adjustable="box")
        self.ax_track.grid(True, alpha=0.25)

        self.ax_delta.set_title("Cumulative ΔT (Optimized − Baseline) vs Distance", color="white")
        self.ax_delta.set_xlabel("Distance (m)")
        self.ax_delta.set_ylabel("ΔT (s)")
        self.ax_delta.grid(True, alpha=0.25)

        self.txt_setup.delete("1.0", tk.END)
        self.txt_worldrecord.configure(state="normal")
        self.txt_worldrecord.delete("1.0", tk.END)
        self.txt_worldrecord.configure(state="disabled")

        for i in self.tree.get_children():
            self.tree.delete(i)
        self.status.config(text="")
        self.canvas.draw_idle()

    # -------------
    # Main action
    # -------------
    def run_optimization(self):
        self._clear_plots()
        year = self.ent_year.get().strip()
        event = self.ent_event.get().strip()
        ses = self.cb_session.get().strip()

        try:
            session, lap, tel = load_baseline_fastest_lap(int(year), event, ses)
            metadata = extract_world_record_metadata(session)
            specs = extract_car_telemetry_specs(session, lap)
            populate_world_record_panel(self.txt_worldrecord, metadata, specs)
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load session: {e}")
            return

        # Baseline data
        baseline_time = lap["LapTime"].total_seconds()
        S1 = lap["Sector1Time"].total_seconds()
        S2 = lap["Sector2Time"].total_seconds()
        S3 = lap["Sector3Time"].total_seconds()

        # Physics params (tweakable per car/track)
        params = {
            "mass_kg": 798.0,
            "mu": 1.85,
            "air_density": 1.18,
            "ClA": 3.4,
            "vmax_straight": 95.0,
            "a_long_accel": 6.0,
            "a_long_brake": 12.0,
            "smooth_window": 41
        }

        # Optimization (PoC): min-time on smoothed baseline line
        opt_df = build_optimized_line_from_baseline(tel, params)
        opt_time, opt_cum_t = integrate_time(
            opt_df["Distance"].values,
            opt_df["Speed_opt"].values
        )

        # ΔT curve: optimized − baseline (align by distance samples)
        try:
            base_s = tel["Distance"].values
            base_v = tel["Speed"].values.clip(min=1.0)
            _, base_cum_t = integrate_time(base_s, base_v)
            base_cum_t_interp = np.interp(opt_df["Distance"].values, base_s, base_cum_t)
            delta_t = opt_cum_t - base_cum_t_interp
        except Exception:
            delta_t = opt_cum_t - np.linspace(0, baseline_time, len(opt_cum_t))

        # Track lines
        x, y = tel["X"].values, tel["Y"].values
        xs, ys = opt_df["X"].values, opt_df["Y"].values

        self.ax_track.plot(x, y, color="#ffcc00", lw=2.4, label="World Record (session fastest)")
        self.ax_track.plot(xs, ys, color="#00ffff", lw=2.0, label="Optimized (PoC)")
        self.ax_track.legend(loc="upper right", frameon=True)

        # ΔT
        self.ax_delta.plot(opt_df["Distance"].values, delta_t, color="#00ffff", lw=2.0)

        # Setup suggestions
        kappa = compute_curvature(xs, ys)
        stats = {
            "avg_kappa": float(np.mean(np.abs(kappa))),
            "top_speed": float(np.max(opt_df["Speed_opt"].values * 3.6)),
            "brake_g": 3.8
        }
        setup = suggest_optimal_setup(stats)

        self.txt_setup.insert(tk.END, f"Track: {event} {year} ({ses})\n")
        self.txt_setup.insert(tk.END, f"Baseline Lap: {baseline_time:.3f} s\n")
        self.txt_setup.insert(tk.END, f"Optimized Lap: {opt_time:.3f} s\n")
        self.txt_setup.insert(tk.END, f"Δ Lap (Opt - Base): {(opt_time - baseline_time):.3f} s\n\n")
        self.txt_setup.insert(tk.END, "Recommended Setup:\n")
        for k, v in setup.items():
            self.txt_setup.insert(tk.END, f"  • {k}: {v}\n")

        # table rows
        self.tree.insert(
            "", "end",
            values=("Baseline", f"{S1:.3f}", f"{S2:.3f}", f"{S3:.3f}", f"{baseline_time:.3f}")
        )
        s_total = float(opt_df["Distance"].values[-1])
        thirds = [s_total / 3 * i for i in range(1, 4)]
        t_split = [float(np.interp(d, opt_df["Distance"].values, opt_cum_t)) for d in thirds]
        S1o, S2o, S3o = t_split[0], (t_split[1] - t_split[0]), (t_split[2] - t_split[1])
        self.tree.insert(
            "", "end",
            values=("Optimized", f"{S1o:.3f}", f"{S2o:.3f}", f"{S3o:.3f}", f"{opt_time:.3f}")
        )

        self.status.config(
            text="Done. Tip: tweak ClA and a_long_brake for different downforce/tyre scenarios."
        )
        self.canvas.draw_idle()


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    app = F1MinTimeGUI()
    app.mainloop()
