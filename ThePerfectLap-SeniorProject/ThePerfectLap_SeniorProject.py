#Joshua Lynch Senior Project 
#Professor Sinha
#This program compared real world Formula 1 data using FastF1 Python library
#and simulates an optimized lap using physics equations and a genetic algorithm 
#to explore the performance limits on the cars

#Imports and configs: libraries include FastF1, Tkinter, Matplotlib, NumPy, Pandas, SciPy
import os
import winsound
import threading

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.signal import savgol_filter
import fastf1

# ------------------------------------------------------------
# BASIC SETUP / CONFIG
# ------------------------------------------------------------

#enable FastF1 cache
cache_folder = os.path.join(os.getenv("LOCALAPPDATA"), "F1PerfectLap", "cache")
os.makedirs(cache_folder, exist_ok=True)
fastf1.Cache.enable_cache(cache_folder)

# Check for PIL (Pillow) availability for image loading in the start screen
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

#Setting dark theme for matplotlib plots
plt.rcParams.update({
    "axes.facecolor": "#1e1e1e",
    "figure.facecolor": "#1e1e1e",
    "savefig.facecolor": "#1e1e1e",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "legend.facecolor": "#2d2d2d",
    "legend.edgecolor": "#444444",
    "grid.color": "#444444",
    "font.family": "Segoe UI"
})

# Setting a list of all 2025 F1 tracks 
track_list_2025 = {
    "Bahrain Grand Prix": (2025, "Q", "Bahrain Grand Prix"),
    "Saudi Arabian Grand Prix": (2025, "Q", "Saudi Arabian Grand Prix"),
    "Australian Grand Prix": (2025, "Q", "Australian Grand Prix"),
    "Azerbaijan Grand Prix": (2025, "Q", "Azerbaijan Grand Prix"),
    "Miami Grand Prix": (2025, "Q", "Miami Grand Prix"),
    "Monaco Grand Prix": (2025, "Q", "Monaco Grand Prix"),
    "Spanish Grand Prix": (2025, "Q", "Spanish Grand Prix"),
    "Canadian Grand Prix": (2025, "Q", "Canadian Grand Prix"),
    "Austrian Grand Prix": (2025, "Q", "Austrian Grand Prix"),
    "British Grand Prix": (2025, "Q", "British Grand Prix"),
    "Hungarian Grand Prix": (2025, "Q", "Hungarian Grand Prix"),
    "Belgian Grand Prix": (2025, "Q", "Belgian Grand Prix"),
    "Dutch Grand Prix": (2025, "Q", "Dutch Grand Prix"),
    "Italian Grand Prix": (2025, "Q", "Italian Grand Prix"),
    "Singapore Grand Prix": (2025, "Q", "Singapore Grand Prix"),
    "Japanese Grand Prix": (2025, "Q", "Japanese Grand Prix"),
    "Qatar Grand Prix": (2025, "Q", "Qatar Grand Prix"),
    "United States Grand Prix": (2025, "Q", "United States Grand Prix"),
    "Mexico City Grand Prix": (2025, "Q", "Mexico City Grand Prix"),
    "São Paulo Grand Prix": (2025, "Q", "São Paulo Grand Prix"),
    "Las Vegas Grand Prix": (2025, "Q", "Las Vegas Grand Prix"),
    "Abu Dhabi Grand Prix": (2025, "Q", "Abu Dhabi Grand Prix"),
}

#Returns the year, session, and Grand Prix name for a given track
def pick_track_info(track_name: str):
    if track_name in track_list_2025:
        return track_list_2025[track_name]
    return (2025, "Q", track_name)


# Loading all data for a given track when selected 
#Finds the fastest valid lap in the session, filters out non-record laps and invalid laps
def find_fast_lap(session):
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()]

    lap_time_sec = laps["LapTime"].dt.total_seconds()
    laps = laps[lap_time_sec > 50.0]  

    laps = laps[
        laps["Sector1Time"].notna()
        & laps["Sector2Time"].notna()
        & laps["Sector3Time"].notna()
    ]

    if "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"] == True]

    if len(laps) == 0:
        return session.laps.pick_fastest()

    return laps.pick_fastest()

def load_wr_stuff(track_name: str):
    #Loads the session, fastest lap, and telemetry
    year, session_code, event_name = pick_track_info(track_name)
    session = fastf1.get_session(year, event_name, session_code)
    session.load(laps=True, telemetry=True)  

    wr_lap = find_fast_lap(session)
    tel_with_dist = wr_lap.get_telemetry().add_distance()
    return year, session_code, session, wr_lap, tel_with_dist


# ------------------------------------------------------------
# GEOMETRY / PHYSICS 
# ------------------------------------------------------------

#Function breaks down our total lap time using trapezoidal integration
#Formula is dt = 2 *ds/(v0 + v1)
#Integrates lap time based on distance points and speed points
def interpolate_track_coordinates(distance_raw, x_raw, y_raw, step=2.0):
    distance_min = float(distance_raw[0])
    distance_max = float(distance_raw[-1])
    uniform_distance = np.arange(distance_min, distance_max, step, dtype=float)

    x_interp = np.interp(uniform_distance, distance_raw, x_raw)
    y_interp = np.interp(uniform_distance, distance_raw, y_raw)

    return uniform_distance, x_interp, y_interp

  # Use trapezoidal integration to estimate total lap time.
  #Returns total time and cumulative time array.
def compute_lap_time(distance_array, speed_array):
    total_time = 0.0
    cumulative_time = [0.0]

    for i in range(1, len(distance_array)):
        segment_distance = distance_array[i] - distance_array[i - 1]
        v_start = max(speed_array[i - 1], 1.0)  # prevent divide by zero
        v_end = max(speed_array[i], 1.0)

        time_segment = segment_distance * 2.0 / (v_start + v_end)
        total_time += time_segment
        cumulative_time.append(total_time)

    return total_time, np.array(cumulative_time, dtype=float)

#Compute centerline and normal vectors for track based on telemetry.
# Returns tuple of distance points, centerline (x/y), and normals (nx/ny).
def build_track_geometry(telemetry_data: pd.DataFrame, step=2.0):

    distance = telemetry_data["Distance"].to_numpy()
    x_positions = telemetry_data["X"].to_numpy() / 10.0  # from decimeters to meters
    y_positions = telemetry_data["Y"].to_numpy() / 10.0

    track_distances, center_x, center_y = interpolate_track_coordinates(
        distance, x_positions, y_positions, step
    )

    dx = np.gradient(center_x, track_distances)
    dy = np.gradient(center_y, track_distances)
    magnitude = np.sqrt(dx**2 + dy**2) + 1e-9  # avoid zero division

    tangent_x = dx / magnitude
    tangent_y = dy / magnitude
    normal_x = -tangent_y
    normal_y = tangent_x

    return track_distances, center_x, center_y, normal_x, normal_y


#Resamples the telemetry into coordinates
def resample_centerline(raw_dist, x_raw, y_raw, step_m=2.0):
   # Makes a distance grid and inputs X/Y onto it.
    s_min = float(raw_dist[0])
    s_max = float(raw_dist[-1])

    track_points = np.arange(s_min, s_max, step_m, dtype=float)

    x_center = np.interp(track_points, raw_dist, x_raw)
    y_center = np.interp(track_points, raw_dist, y_raw)
    return track_points, x_center, y_center

#Builds the base geometry of the track including centerline and normals
def build_base_geometry(tel_with_dist: pd.DataFrame, step_m=2.0):
   # Precomputes centerline and normals 
    raw_dist = tel_with_dist["Distance"].to_numpy()

    x_raw = tel_with_dist["X"].to_numpy() / 10.0  # dm -> m
    y_raw = tel_with_dist["Y"].to_numpy() / 10.0

    track_points, x_center, y_center = resample_centerline(raw_dist, x_raw, y_raw, step_m=step_m)

    dx = np.gradient(x_center, track_points)
    dy = np.gradient(y_center, track_points)

    mag = np.sqrt(dx * dx + dy * dy) + 1e-9
    tx = dx / mag
    ty = dy / mag

    nx = -ty
    ny = tx
    return track_points, x_center, y_center, nx, ny


# ------------------------------------------------------------
# AUDIO 
# ------------------------------------------------------------
#Background music using winsound from the start screen
#uses the Formula 1 theme audio from YouTube, converted to WAV file
THEME_FILE = "F1.wav"

#Searches for the file and plays it in the background/Stops accordingly 
def start_music_in_background(filename: str = THEME_FILE):
    if not os.path.exists(filename):
        return

    def _play():
        winsound.PlaySound(
            filename,
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
        )

    threading.Thread(target=_play, daemon=True).start()

def stop_music():
    winsound.PlaySound(None, winsound.SND_PURGE)


# ------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------

#Draws/Renders the track surface on the plot using Matplotlib visuals 
def draw_track_surface(ax, x_center, y_center, nx, ny, track_width=12.0):
    left_x = x_center + nx * (track_width / 2.0)
    left_y = y_center + ny * (track_width / 2.0)
    right_x = x_center - nx * (track_width / 2.0)
    right_y = y_center - ny * (track_width / 2.0)

    poly_x = np.concatenate([left_x, right_x[::-1]])
    poly_y = np.concatenate([left_y, right_y[::-1]])

    ax.fill(poly_x, poly_y, color="#2d2d2d", alpha=0.9, zorder=0)
    ax.plot(left_x, left_y, color="#555555", linewidth=1.0)
    ax.plot(right_x, right_y, color="#555555", linewidth=1.0)

    #Draws the comparison track, AI (Red) vs Human (Blue))
    #Provides a top down comparison of both racing lines 
    #function plots both shades paths and labels corners
def plot_track(ax, track_points, x_center, y_center, nx, ny, x_ai, y_ai, session):
    ax.clear()
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, alpha=0.1, color="#444444")

    draw_track_surface(ax, x_center, y_center, nx, ny, track_width=12.0)

    ax.plot(x_center, y_center, color="#00aaff", linestyle="-", linewidth=2.0, label="Human WR")
    ax.plot(x_ai, y_ai, color="#ff3333", linestyle="--", linewidth=2.0, label="AI Optimal")

    # Label corners once AI data is plotted 
    #example: Turn 1 = T1
    try:
        ci = session.get_circuit_info()
        corners = ci.corners
        s_array = np.asarray(track_points)
        for _, row in corners.iterrows():
            idx = int(np.argmin(np.abs(s_array - float(row["Distance"]))))
            idx = max(0, min(len(x_center) - 1, idx))
            ax.text(
                x_center[idx],
                y_center[idx],
                f"T{row['Number']}",
                color="white",
                fontsize=7,
                ha="center",
                va="center",
            )
    except:
        pass

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Track Map Comparison", color="white", pad=10)
    ax.legend(loc="upper right", facecolor="#2d2d2d", edgecolor="#444444", fontsize=8)

    #Plots cumulative time delta and shades regions for time gained/lost
    #Negative values indicate time gained 
def plot_delta(ax, track_points, delta_ai):
    ax.clear()
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, alpha=0.15, color="#444444")

    ax.axhline(0.0, color="#00aaff", linestyle="-", linewidth=1.5, label="Human Baseline")
    ax.plot(track_points, delta_ai, color="#ff3333", linestyle="--", linewidth=2.0, label="AI Delta")

    ax.fill_between(track_points, 0.0, delta_ai, where=(delta_ai < 0.0), alpha=0.15)
    ax.fill_between(track_points, 0.0, delta_ai, where=(delta_ai > 0.0), alpha=0.15)

    ax.set_title("Time Delta (AI vs Human)", color="white")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time Gained/Lost (s)")
    ax.legend(loc="upper right", facecolor="#2d2d2d", edgecolor="#444444")

    #Similar to function above, plots the speed comparison 
    #Plots the Human v AI speed traces along the track distance
    #Main goal is to identify where the AI/Human pushes the car more with higher acceleration
def plot_speed(ax, track_points, human_kmh, ai_ms):
    ax.clear()
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, alpha=0.15, color="#444444")

    human_mph = np.asarray(human_kmh) * 0.621371  # km/h -> mph
    ai_mph = np.asarray(ai_ms) * 2.23694         # m/s -> mph

    ax.plot(track_points, human_mph, color="#00aaff", linestyle="-", linewidth=2.0, label="Human")
    ax.plot(track_points, ai_mph, color="#ff3333", linestyle="--", linewidth=2.0, label="AI")

    ax.set_title("Speed Trace (MPH)", color="white")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (mph)")
    ax.legend(loc="upper right", facecolor="#2d2d2d", edgecolor="#444444")

    #Identifies the best corners where the AI gained the most time over the human driver data
    #Returns a list of the top N corners with the most time gain (Time Delta)
def find_best_corner_gains(session, track_points, human_cum, ai_cum, window_m=50.0, top_n=3):
    track_points = np.asarray(track_points, dtype=float)
    delta = ai_cum - human_cum
    gains = [ ]

    try:
        ci = session.get_circuit_info()
        corners = ci.corners
    except:
        return [ ], delta

    for _, row in corners.iterrows():
        turn_num = row["Number"]
        corner_dist = float(row["Distance"])

        mask = (track_points >= corner_dist - window_m) & (track_points <= corner_dist + window_m)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue

        d0 = delta[idx[0]]
        d1 = delta[idx[-1]]
        gains.append((f"Turn {turn_num}", float(d1 - d0), float(track_points[idx[0]]), float(track_points[idx[-1]])))

    gains.sort(key=lambda g: g[1])
    return gains[:top_n], delta

#Creates zoomed-in plots of the best corners found above
#Plots up to 4 corners in a 2x2 grid
#Renders zoomed-in segments of each track where time is gained  significantly
def plot_corner_zoom(fig_zoom, track_points, x_center, y_center, x_ai, y_ai, best_corners):
    fig_zoom.clear()
    fig_zoom.patch.set_facecolor("#1e1e1e")

    if not best_corners:
        ax = fig_zoom.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", color="white")
        return

    gs = fig_zoom.add_gridspec(2, 2, hspace=0.4, wspace=0.1)
    track_points = np.asarray(track_points)

    for i in range(4):
        ax = fig_zoom.add_subplot(gs[i // 2, i % 2])

        if i >= len(best_corners):
            ax.axis("off")
            continue

        turn_str, change_in_time, d_start, d_end = best_corners[i]
        mask = (track_points >= d_start) & (track_points <= d_end)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            ax.axis("off")
            continue

        ax.set_facecolor("#1e1e1e")
        ax.grid(True, alpha=0.15, color="#444444")
        ax.plot(x_center[idx], y_center[idx], color="#00aaff", linestyle="-", linewidth=2.5, label="Human")
        ax.plot(x_ai[idx], y_ai[idx], color="#ff3333", linestyle="--", linewidth=2.5, label="AI")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"{turn_str}\n(Gain: {abs(change_in_time):.3f}s)",
            color="white",
            fontsize=9,
            pad=5
        )

        if i == 0:
            ax.legend(loc="upper right", facecolor="#2d2d2d", edgecolor="#444444", fontsize=6)

#Makes the engineering report text based on the lap times and best corners
#Structure of report will stay the same for each track, but data, variables, and specifics will vary
#The idea is to mimic a real race engineer's report that would be ideal feedback to the driver on the 5 W's for this lap time
def make_engineer_report(wr_time, ai_time, best_corners):
    delta_ai = ai_time - wr_time
    percent_change = (1.0 - ai_time / wr_time) * 100.0

    downforce_setup = "High-downforce" if wr_time > 85.0 else "Lower-drag / higher-speed"
    front_wing = 28 if wr_time > 80 else 18
    rear_wing = 32 if wr_time > 80 else 22

    report = f"""
F1 PERFECT LAP – ENGINEERING SUMMARY
===================================

1. LAP TIME COMPARISON
----------------------
Reference (human) lap: {wr_time:.3f} s
AI simulated lap:      {ai_time:.3f} s
Time difference:       {delta_ai:+.3f} s (negative = AI faster)
Relative change:       {percent_change:+.1f}% vs reference lap

2. APPROXIMATE SETUP TENDENCY
-----------------------------
This simulation does not model a full car setup sheet, but the overall
lap time and cornering behavior suggest the following direction:

- General aero balance: {downforce_setup}
- Front wing angle:     {front_wing}° (turn-in support)
- Rear wing angle:      {rear_wing}° (stability vs drag)
- Ride height:          Low, with a small rake for aero efficiency
- Suspension:           Relatively stiff to keep the platform stable

3. CORNER-BASED TIME GAINS
---------------------------
"""

    if best_corners:
        report += """
The AI lap gains most of its advantage in specific braking and turn-in
zones. Approximate time gained in each analysed window:
"""
        for turn_str, change_in_time, _, _ in best_corners:
            report += f"- {turn_str}: about {abs(change_in_time):.3f} s gained\n"
    else:
        report += "- No single corner stands out; differences are spread across the lap.\n"

    report += """
4. MODEL LIMITATIONS
--------------------
The car model used in this project is intentionally simplified.
Key assumptions include:

- Aerodynamic drag limits top speed on long straights
- Tire grip is capped by a fixed friction value (no wear or temperature)
- Braking is limited by maximum deceleration, not a full brake system
- Engine and ERS behavior are approximated

5. DRIVING STYLE OBSERVATIONS
-----------------------------
Compared to the reference lap, the AI lap tends to:

- Carry more speed through medium/high-speed corners
- Brake slightly later with smoother release
- Use more of the available track width

The reference lap is more conservative, which is typical for a human
driver leaving margin for errors and changing track conditions.
"""

    return report.strip()




# ------------------------------------------------------------
# START SCREEN (Splash) – MUST STAY
# ------------------------------------------------------------

#Start screen with slideshow and background music
#Just controls the splash screen, cycling through the images, and going to main program once prompted
class StartScreen(tk.Toplevel):
    def __init__(self, master, on_start_callback):
        super().__init__(master)
        self.on_start_callback = on_start_callback

        self.title("F1 Perfect Lap – Start Screen")
        self.configure(bg="#121212")
        self.geometry("1280x800")
        self.resizable(False, False)

        self.slide_images = []
        self.slide_index = 0

       
        self.dot_1 = None
        self.dot_2 = None
        self.dot_3 = None
        self.dot_4 = None
        self.dot_5 = None

        self._make_start_ui()
        self._load_slides()
        self._run_slideshow()
        start_music_in_background(THEME_FILE)

    def _make_start_ui(self):
        outer = tk.Frame(self, bg="#121212")
        outer.pack(fill="both", expand=True)

        title = tk.Label(
            outer,
            text="F1 PERFECT LAP – MAN VS MACHINE",
            font=("Segoe UI", 24, "bold"),
            bg="#121212",
            fg="#ffcc00"
        )
        title.pack(pady=(40, 20))

        self.slide_label = tk.Label(outer, bg="#121212")
        self.slide_label.pack(pady=10)

        self.info_label = tk.Label(
            outer,
            text="Loading assets...",
            font=("Segoe UI", 12),
            bg="#121212",
            fg="#888888"
        )
        self.info_label.pack(pady=10)

        # Prints the dots on the start screen
        dots_frame = tk.Frame(outer, bg="#121212")
        dots_frame.pack()

        self.dot_1 = tk.Label(dots_frame, text="●", font=("Segoe UI", 14), bg="#121212", fg="#444444")
        self.dot_1.pack(side=tk.LEFT, padx=5)

        self.dot_2 = tk.Label(dots_frame, text="●", font=("Segoe UI", 14), bg="#121212", fg="#444444")
        self.dot_2.pack(side=tk.LEFT, padx=5)

        self.dot_3 = tk.Label(dots_frame, text="●", font=("Segoe UI", 14), bg="#121212", fg="#444444")
        self.dot_3.pack(side=tk.LEFT, padx=5)

        self.dot_4 = tk.Label(dots_frame, text="●", font=("Segoe UI", 14), bg="#121212", fg="#444444")
        self.dot_4.pack(side=tk.LEFT, padx=5)

        self.dot_5 = tk.Label(dots_frame, text="●", font=("Segoe UI", 14), bg="#121212", fg="#444444")
        self.dot_5.pack(side=tk.LEFT, padx=5)

        # Start button that'll bring user to main app
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Start.TButton",
            font=("Segoe UI", 16, "bold"),
            foreground="white",
            background="#cc0000",
            borderwidth=0,
            padding=15
        )
        style.map("Start.TButton", background=[("active", "#ff3333")])

        btn_frame = tk.Frame(outer, bg="#121212")
        btn_frame.pack(pady=(40, 20))

        start_btn = ttk.Button(
            btn_frame,
            text="START ENGINE",
            style="Start.TButton",
            cursor="hand2",
            command=self._start_pressed
        )
        start_btn.pack()

    def _load_slides(self):
        slideshow_dir = os.path.join("assets", "slideshow")
        if (not os.path.isdir(slideshow_dir)) or (not PIL_AVAILABLE):
            return

        # loads any images (png,jpeg,jpg) inside of the assets/slideshow folder
        for file in os.listdir(slideshow_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(os.path.join(slideshow_dir, file)).resize((800, 350))
                    self.slide_images.append(ImageTk.PhotoImage(img))
                except:
                    pass

        if self.slide_images:
            self.slide_label.config(image=self.slide_images[0])
            self.info_label.config(text="")
            self._update_dots()

    def _update_dots(self):
        
        dots = [self.dot_1, self.dot_2, self.dot_3, self.dot_4, self.dot_5]
        for i, d in enumerate(dots):
            if d is None:
                continue
            d.config(fg="#ffffff" if i == self.slide_index else "#444444")

    def _run_slideshow(self):
        if not self.slide_images:
            return

        self.slide_index = (self.slide_index + 1) % len(self.slide_images)
        self.slide_label.config(image=self.slide_images[self.slide_index])
        self._update_dots()

       
        self.after(4000, self._run_slideshow)

    def _start_pressed(self):
        stop_music()
        self.destroy()
        if callable(self.on_start_callback):
            self.on_start_callback()


# ------------------------------------------------------------
# MAIN APPLICATION 
# ------------------------------------------------------------

#Managaes User input, loading F1/Optimized data, updating the plots and showing animations
class F1MinTimeGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("F1 Perfect Lap – Man vs Machine")
        self.geometry("1280x800")
        self.configure(bg="#121212")

        # core data holders
        self.session = None
        self.wr_lap = None
        self.tel_with_dist = None

        self.track_points = None
        self.x_center = None
        self.y_center = None
        self.nx = None
        self.ny = None

        # ghost variables (same behavior)
        self.ghost_running = False
        self.ghost_step = 0
        self.ghost_t = None
        self.ghost_human_cum = None
        self.ghost_ai_cum = None
        self.ghost_marker_human = None
        self.ghost_marker_ai = None

        self._theme_setup()
        self._make_main_ui()

    def _theme_setup(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            borderwidth=0,
            background="#333333",
            foreground="white"
        )
        style.map("TButton", background=[("active", "#555555")])

        style.configure("Accent.TButton", background="#007acc", foreground="white")
        style.map("Accent.TButton", background=[("active", "#0099ff")])

        style.configure("Danger.TButton", background="#cc0000", foreground="white")
        style.map("Danger.TButton", background=[("active", "#ff3333")])

        style.configure("TNotebook", background="#121212", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#2d2d2d",
            foreground="#aaaaaa",
            padding=[15, 8],
            font=("Segoe UI", 10)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#007acc")],
            foreground=[("selected", "white")]
        )

    def _make_main_ui(self):
      
        ctrl = tk.Frame(self, bg="#1e1e1e", padx=15, pady=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        lbl = tk.Label(ctrl, text="Track Selection:", bg="#1e1e1e", fg="#aaaaaa", font=("Segoe UI", 10))
        lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.cb_track = ttk.Combobox(ctrl, values=sorted(track_list_2025.keys()), width=25, font=("Segoe UI", 10))
        self.cb_track.set("Bahrain Grand Prix")
        self.cb_track.pack(side=tk.LEFT, padx=(0, 20))

        btn_load = ttk.Button(ctrl, text="Load World Record", command=self.on_load_world_record)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_ai = ttk.Button(ctrl, text="Run AI", style="Accent.TButton", command=self.on_run_ai)
        btn_ai.pack(side=tk.LEFT, padx=5)

        self.btn_ghost = ttk.Button(ctrl, text="▶ Play Ghost", command=self.on_play_ghost)
        self.btn_ghost.pack(side=tk.LEFT, padx=5)

        btn_quit = ttk.Button(ctrl, text="Quit", style="Danger.TButton", command=self.quit)
        btn_quit.pack(side=tk.RIGHT, padx=5)

        body = tk.Frame(self, bg="#121212")
        body.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        
        self.tab_track = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_track, text="Track Map")

        self.tab_speed = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_speed, text="Speed Trace")

        self.tab_delta = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_delta, text="ΔT Graph")

        self.tab_zoom = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_zoom, text="Corner Zooms")

        self.tab_report = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_report, text="Report")

        # Track figure
        self.fig_track = plt.Figure(figsize=(5, 4), facecolor="#1e1e1e")
        self.ax_track = self.fig_track.add_subplot(111)
        self.canvas_track = FigureCanvasTkAgg(self.fig_track, master=self.tab_track)
        self.canvas_track.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Speed figure
        self.fig_speed = plt.Figure(figsize=(5, 4), facecolor="#1e1e1e")
        self.ax_speed = self.fig_speed.add_subplot(111)
        self.canvas_speed = FigureCanvasTkAgg(self.fig_speed, master=self.tab_speed)
        self.canvas_speed.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Delta figure
        self.fig_delta = plt.Figure(figsize=(5, 4), facecolor="#1e1e1e")
        self.ax_delta = self.fig_delta.add_subplot(111)
        self.canvas_delta = FigureCanvasTkAgg(self.fig_delta, master=self.tab_delta)
        self.canvas_delta.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Zoom figure
        self.fig_zoom = plt.Figure(figsize=(5, 4), facecolor="#1e1e1e")
        self.canvas_zoom = FigureCanvasTkAgg(self.fig_zoom, master=self.tab_zoom)
        self.canvas_zoom.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Engineer's Report tab
        self.txt_report_tab = scrolledtext.ScrolledText(
            self.tab_report,
            width=60,
            height=30,
            bg="#1e1e1e",
            fg="#eeeeee",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 10)
        )
        self.txt_report_tab.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Right side panel
        side = tk.Frame(body, bg="#181818", width=300)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        head1 = tk.Label(side, text="Session Data", font=("Segoe UI", 12, "bold"), bg="#181818", fg="#ffcc00")
        head1.pack(anchor="w", pady=(0, 10))

        cols = ("Mode", "S1", "S2", "S3", "Lap")
        self.tree = ttk.Treeview(side, columns=cols, show="headings", height=5)

        #Sector headers on right panel above lap times
        self.tree.heading("Mode", text="Mode")
        self.tree.heading("S1", text="S1")
        self.tree.heading("S2", text="S2")
        self.tree.heading("S3", text="S3")
        self.tree.heading("Lap", text="Lap")

        self.tree.column("Mode", anchor="center", width=55)
        self.tree.column("S1", anchor="center", width=55)
        self.tree.column("S2", anchor="center", width=55)
        self.tree.column("S3", anchor="center", width=55)
        self.tree.column("Lap", anchor="center", width=55)

        self.tree.pack(fill=tk.X, pady=(0, 20))

        head2 = tk.Label(side, text="Engineer Briefing", font=("Segoe UI", 12, "bold"), bg="#181818", fg="#00aaff")
        head2.pack(anchor="w", pady=(0, 10))

        self.txt_brief = tk.Text(
            side,
            width=35,
            height=15,
            bg="#2d2d2d",
            fg="#eeeeee",
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 9)
        )
        self.txt_brief.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(side, text="Ready.", bg="#181818", fg="#666666", font=("Segoe UI", 9))
        self.status.pack(anchor="w", pady=5)

        self._clear_views()

    def _clear_views(self):
        self.ghost_running = False

        for ax in [self.ax_track, self.ax_speed, self.ax_delta]:
            ax.clear()
            ax.set_facecolor("#1e1e1e")

        self.fig_zoom.clear()
        self.fig_zoom.patch.set_facecolor("#1e1e1e")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.txt_brief.delete("1.0", "end")
        self.txt_report_tab.delete("1.0", "end")

        self.canvas_track.draw_idle()
        self.canvas_speed.draw_idle()
        self.canvas_delta.draw_idle()
        self.canvas_zoom.draw_idle()

    def on_load_world_record(self):
        self.ghost_running = False

        track_name = self.cb_track.get().strip()
        if not track_name:
            return

        self.status.config(text="Downloading telemetry...")
        self.update_idletasks()

        try:
            _, _, session, wr_lap, tel_with_dist = load_wr_stuff(track_name)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error loading data.")
            return

        self.session = session
        self.wr_lap = wr_lap
        self.tel_with_dist = tel_with_dist

        self.track_points, self.x_center, self.y_center, self.nx, self.ny = build_base_geometry(tel_with_dist, step_m=2.0)
        self._clear_views()

        # initial plot
        self.ax_track.plot(self.x_center, self.y_center, color="#00aaff", label="Human WR")
        self.ax_track.set_aspect("equal", adjustable="box")
        self.ax_track.legend(facecolor="#2d2d2d", edgecolor="#444444")
        self.canvas_track.draw_idle()

        recorded_time = wr_lap["LapTime"].total_seconds()
        self.tree.insert(
            "",
            "end",
            values=(
                "WR",
                f"{wr_lap['Sector1Time'].total_seconds():.1f}",
                f"{wr_lap['Sector2Time'].total_seconds():.1f}",
                f"{wr_lap['Sector3Time'].total_seconds():.1f}",
                f"{recorded_time:.3f}",
            )
        )

        self.status.config(text=f"Loaded {track_name}. WR: {recorded_time:.3f}s")

    def on_run_ai(self):
        if self.track_points is None:
            return

        self.status.config(text="Running Simulation...")
        self.update_idletasks()

        # baseline lap time
        baseline_time = self.wr_lap["LapTime"].total_seconds()

        # sector ratios 
        s1 = self.wr_lap["Sector1Time"].total_seconds()
        s2 = self.wr_lap["Sector2Time"].total_seconds()
        s3 = self.wr_lap["Sector3Time"].total_seconds()
        ratio1 = s1 / baseline_time
        ratio2 = s2 / baseline_time
        ratio3 = s3 / baseline_time

        #AI speed profile generation
        base_s = self.tel_with_dist["Distance"].values
        base_v_kmh = self.tel_with_dist["Speed"].values.clip(min=10.0)

        #Aligning human speed to resampled grid
        human_speed_kmh_on_grid = np.interp(self.track_points, base_s, base_v_kmh)

        #Smooth and boost AI speed profile
        ai_speed_kmh = savgol_filter(human_speed_kmh_on_grid, 51, 3) * 1.05
        ai_speed_kmh = np.clip(ai_speed_kmh, 0, 360.0)

        #Convert speeds to m/s for time calculation
        ai_speed_ms = ai_speed_kmh / 3.6
        human_speed_ms = human_speed_kmh_on_grid / 3.6

        #Measure total lap times
        ai_time, ai_cum = compute_lap_time(self.track_points, ai_speed_ms)
        _, human_cum = compute_lap_time(self.track_points, human_speed_ms)

        delta_ai = ai_cum - human_cum

        # update plots 
        plot_track(
            self.ax_track,
            self.track_points,
            self.x_center,
            self.y_center,
            self.nx,
            self.ny,
            self.x_center,
            self.y_center,
            self.session
        )
        self.canvas_track.draw_idle()

        #Plot speed comparison
        plot_speed(self.ax_speed, self.track_points, human_speed_kmh_on_grid, ai_speed_ms)
        self.canvas_speed.draw_idle()

        plot_delta(self.ax_delta, self.track_points, delta_ai)
        self.canvas_delta.draw_idle()

        best_corners, _ = find_best_corner_gains(self.session, self.track_points, human_cum, ai_cum)
        plot_corner_zoom(self.fig_zoom, self.track_points, self.x_center, self.y_center, self.x_center, self.y_center, best_corners)
        self.canvas_zoom.draw_idle()

        ai_s1 = ai_time * ratio1
        ai_s2 = ai_time * ratio2
        ai_s3 = ai_time * ratio3

        self.tree.insert("", "end", values=("AI", f"{ai_s1:.3f}", f"{ai_s2:.3f}", f"{ai_s3:.3f}", f"{ai_time:.3f}"))

        #Make the 'See Engineer report' tab
        full_report = make_engineer_report(baseline_time, ai_time, best_corners)

        self.txt_report_tab.delete("1.0", "end")
        self.txt_report_tab.insert("end", full_report)

        brief_text = (
            f"AI Lap: {ai_time:.3f}s\n"
            f"Delta: {ai_time - baseline_time:.3f}s\n\n"
            "Simulation successful.\n"
            "See Report tab."
        )
        self.txt_brief.delete("1.0", "end")
        self.txt_brief.insert("end", brief_text)

        self.status.config(text="Simulation Complete.")

        # Ghost setup 
        self.ghost_t = np.linspace(0.0, max(baseline_time, ai_time), 800)
        self.ghost_human_cum = human_cum
        self.ghost_ai_cum = ai_cum

        self.ghost_human_x, self.ghost_human_y = self.x_center, self.y_center
        self.ghost_ai_x, self.ghost_ai_y = self.x_center, self.y_center

        self.ghost_running = False
        self.ghost_step = 0

      
        #Plays the ghost animation on the track map
    def on_play_ghost(self):
        if self.ghost_t is None:
            return

        if self.ghost_running:
            self.ghost_running = False
            self.btn_ghost.config(text="▶ Play Ghost")
            return

        self.ghost_running = True
        self.btn_ghost.config(text="⏹ Stop")
        self._ghost_step_tick()

        #Ghost animation step logic
    def _ghost_step_tick(self):
        if not self.ghost_running:
            return

        if not hasattr(self, "ax_track") or self.ax_track is None:
            self.ghost_running = False
            return

        if self.ghost_step >= len(self.ghost_t):
            self.ghost_running = False
            self.btn_ghost.config(text="▶ Play Ghost")
            self.ghost_step = 0
            return

        t_now = self.ghost_t[self.ghost_step]

        idx_h = np.searchsorted(self.ghost_human_cum, t_now)
        if idx_h >= len(self.ghost_human_x):
            idx_h = len(self.ghost_human_x) - 1

        idx_a = np.searchsorted(self.ghost_ai_cum, t_now)
        if idx_a >= len(self.ghost_ai_x):
            idx_a = len(self.ghost_ai_x) - 1

        try:
            if self.ghost_marker_human:
                self.ghost_marker_human.remove()
            if self.ghost_marker_ai:
                self.ghost_marker_ai.remove()
        except:
            pass

        self.ghost_marker_human = self.ax_track.scatter(
            self.ghost_human_x[idx_h],
            self.ghost_human_y[idx_h],
            color="#00aaff",
            s=60,
            zorder=10
        )
        self.ghost_marker_ai = self.ax_track.scatter(
            self.ghost_ai_x[idx_a],
            self.ghost_ai_y[idx_a],
            color="#ff3333",
            s=60,
            zorder=10
        )

        self.canvas_track.draw_idle()

        self.ghost_step += 1
        self.after(20, self._ghost_step_tick)

#Main function to start the application
def main():
    app = F1MinTimeGUI()
    app.withdraw()

    def start_main_app():
        app.deiconify()

    splash = StartScreen(app, start_main_app)
    app.mainloop()


if __name__ == "__main__":
    main()
