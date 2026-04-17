# =========================================================
# I_EM Explorer v4.3
# Based on user's last working version, with:
# 1. mA as the default working/display unit
# 2. Overlay or stacked multi-cycle views after Analyze All
# 3. Separate pop-up interactive windows for each main graph
# 4. Reverse-search plotted by reversed point index by default
# 5. Equations window showing the actual mathematical criteria
# 6. Clear color distinction between:
#    - Gray  : noise reference region
#    - Green : accepted candidate window
# 7. Improved detailed explanation text
# =========================================================

import os
import glob
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Patch


# =========================================================
# Plot style
# =========================================================
plt.rcParams.update({
    "font.family": "Cambria",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

EM_COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]
EM_MARKERS = ['o', 's', '^', 'D', 'x', '*', 'v', '<', '>', 'p']

NOISE_SHADE_COLOR = "gray"
NOISE_SHADE_ALPHA = 0.18

ACCEPT_SHADE_COLOR = "green"
ACCEPT_SHADE_ALPHA = 0.22

CANDIDATE_SHADE_COLOR = "green"
CANDIDATE_SHADE_ALPHA = 0.16


# =========================================================
# Helpers
# =========================================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def fmt_num(x, fmt=".6g"):
    try:
        if x is None:
            return ""
        x = float(x)
        if not np.isfinite(x):
            return ""
        return format(x, fmt)
    except Exception:
        return ""


def moving_average_reflect(y, win):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y.copy()

    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    if len(y) < win:
        return y.copy()

    pad = win // 2
    ypad = np.r_[y[pad:0:-1], y, y[-2:-pad-2:-1]]
    kernel = np.ones(win) / win
    ys = np.convolve(ypad, kernel, mode='valid')
    return ys


def normalize_window_fraction(npts, frac):
    win = int(round(max(3, frac * npts)))
    if win % 2 == 0:
        win += 1
    return max(3, win)


def detect_csv_columns(df):
    cols = list(df.columns)
    lc = [c.lower().strip() for c in cols]

    current_candidates = []
    rmin_candidates = []
    rmax_candidates = []

    for c, low in zip(cols, lc):
        if low == "pls" or "average current" in low or "current" in low or "pulse" in low:
            current_candidates.append(c)
        if "rmin" in low:
            rmin_candidates.append(c)
        if "rmax" in low:
            rmax_candidates.append(c)

    current_guess = current_candidates[0] if current_candidates else (cols[0] if cols else None)
    rmin_guess = rmin_candidates[0] if rmin_candidates else (cols[1] if len(cols) > 1 else None)
    rmax_guess = rmax_candidates[0] if rmax_candidates else None

    return current_guess, rmin_guess, rmax_guess


def make_display_name(path, idx=None):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if idx is None:
        return stem
    return f"EM{idx}"


def apply_vertical_offset(y, offset, idx):
    y = np.asarray(y, dtype=float)
    return y + idx * offset


def get_color_and_marker(i):
    color = EM_COLORS[i % len(EM_COLORS)]
    marker = EM_MARKERS[i % len(EM_MARKERS)]
    return color, marker


# =========================================================
# Core analysis
# =========================================================
def compute_detection(I_mA, Rmin, smooth_frac=0.05,
                      deriv_thr=0.0008,
                      consec_points=4,
                      r_tol_sigma=2.0,
                      min_points_plateau=10):
    I = np.asarray(I_mA, dtype=float)
    R = np.asarray(Rmin, dtype=float)

    out = {
        "success": False,
        "message": "",
        "I": I,
        "R": R,
        "R_s": None,
        "dR_dI": None,
        "I_rev": None,
        "R_rev": None,
        "dR_dI_rev": None,
        "rev_index": None,
        "sigma_R": np.nan,
        "max_dR_noise": np.nan,
        "slope_thr": np.nan,
        "deltaR_thr": np.nan,
        "idx_rev_found": None,
        "idx_true_found": None,
        "I_em": np.nan,
        "R_at_Iem": np.nan,
        "candidate_windows": [],
        "noise_slice_rev": None,
        "accepted_window_true": None,
        "accepted_window_rev": None,
        "params": {
            "smooth_frac": smooth_frac,
            "consec_points": int(consec_points),
            "r_tol_sigma": r_tol_sigma,
            "min_points_plateau": int(min_points_plateau),
            "user_deriv_thr": deriv_thr,
            "used_window_points": None,
        }
    }

    n = len(I)
    if n < 6:
        out["message"] = "Not enough points."
        return out

    if np.any(np.diff(I) == 0):
        out["message"] = "Current values contain repeated points. Cannot compute derivative reliably."
        return out

    win = normalize_window_fraction(n, smooth_frac)
    out["params"]["used_window_points"] = win

    R_s = moving_average_reflect(R, win)
    dR_dI = np.gradient(R_s, I)

    I_rev = I[::-1]
    R_rev = R_s[::-1]
    dR_dI_rev = dR_dI[::-1]
    rev_index = np.arange(n)

    mpp = int(max(3, min_points_plateau))
    mpp = min(mpp, n)

    noise_region = R_rev[:mpp]
    sigma_R = np.std(noise_region, ddof=1) if len(noise_region) > 1 else 0.0
    max_dR_noise = np.max(np.abs(dR_dI_rev[:mpp])) if len(dR_dI_rev[:mpp]) > 0 else 0.0

    slope_thr = max(float(deriv_thr), 3.0 * float(max_dR_noise))
    deltaR_thr = float(r_tol_sigma) * float(sigma_R)

    idx = None
    accepted_window_true = None
    accepted_window_rev = None
    candidate_windows = []

    upper_k = max(mpp, n - int(consec_points))
    for k in range(mpp, upper_k):
        sl = slice(k, k + int(consec_points))
        if sl.stop > n:
            break

        local_d = dR_dI_rev[sl]
        local_r = R_rev[sl]

        max_abs_d = float(np.max(np.abs(local_d))) if len(local_d) else np.nan
        delta_R = float(np.max(local_r) - np.min(local_r)) if len(local_r) else np.nan

        cond_deriv = np.all(np.abs(local_d) < slope_thr)
        cond_flat = delta_R < deltaR_thr if len(local_r) > 0 else False

        fail_reason = []
        if not cond_deriv:
            fail_reason.append("slope too large")
        if not cond_flat:
            fail_reason.append("resistance spread too large")
        if not fail_reason:
            fail_reason.append("pass")

        true_indices = [n - 1 - kk for kk in range(sl.start, sl.stop)]
        true_indices = sorted(true_indices)

        candidate_windows.append({
            "k_start_rev": k,
            "k_stop_rev": sl.stop - 1,
            "true_start_idx": true_indices[0],
            "true_stop_idx": true_indices[-1],
            "I_start_rev": float(I_rev[k]),
            "I_stop_rev": float(I_rev[sl.stop - 1]),
            "I_start_true": float(I[true_indices[0]]),
            "I_stop_true": float(I[true_indices[-1]]),
            "cond_deriv": bool(cond_deriv),
            "cond_flat": bool(cond_flat),
            "max_abs_d": max_abs_d,
            "delta_R": delta_R,
            "fail_reason": ", ".join(fail_reason)
        })

        if cond_deriv and cond_flat:
            idx = k
            accepted_window_true = (true_indices[0], true_indices[-1])
            accepted_window_rev = (k, sl.stop - 1)
            break

    out["R_s"] = R_s
    out["dR_dI"] = dR_dI
    out["I_rev"] = I_rev
    out["R_rev"] = R_rev
    out["dR_dI_rev"] = dR_dI_rev
    out["rev_index"] = rev_index
    out["sigma_R"] = sigma_R
    out["max_dR_noise"] = max_dR_noise
    out["slope_thr"] = slope_thr
    out["deltaR_thr"] = deltaR_thr
    out["candidate_windows"] = candidate_windows
    out["noise_slice_rev"] = (0, mpp - 1)
    out["accepted_window_true"] = accepted_window_true
    out["accepted_window_rev"] = accepted_window_rev

    if idx is None:
        out["message"] = "No valid plateau region found with current parameters."
        return out

    true_idx = n - 1 - idx
    out["idx_rev_found"] = idx
    out["idx_true_found"] = true_idx
    out["I_em"] = float(I[true_idx])
    out["R_at_Iem"] = float(R[true_idx])
    out["success"] = True
    out["message"] = "I_EM detected successfully."
    return out


# =========================================================
# Explanation text
# =========================================================
def build_general_explanation(ds, result):
    p = result["params"]

    lines = []
    lines.append(f"Dataset: {ds['display_name']}")
    lines.append("")
    lines.append("Current convention")
    lines.append("All current values are displayed in mA, using pulse amplitude as the analysis variable.")
    lines.append("")
    lines.append("What the raw curve means")
    lines.append("Rmin is the low-bias resistance measured between pulses.")
    lines.append("It is used to track irreversible or quasi-irreversible changes after each pulse.")
    lines.append("")
    lines.append("Why smoothing is used")
    lines.append(
        f"A moving average was applied with fraction {p['smooth_frac']:.4f}, corresponding to {p['used_window_points']} points."
    )
    lines.append("This reduces point-to-point noise before computing dR/dI.")
    lines.append("")
    lines.append("Why the derivative matters")
    lines.append("A plateau-like or transition-compatible window should have derivative magnitude close to zero.")
    lines.append("Derivative alone is not enough, so resistance spread is also checked.")
    lines.append("")
    lines.append("Why reverse search is used")
    lines.append("The search begins from the high-current side and moves backward.")
    lines.append("This is intended to locate the first stable plateau-like window when tracing back from high current.")
    lines.append("")
    lines.append("Meaning of the shaded regions in the reverse plot")
    lines.append("Gray shaded region: noise reference region.")
    lines.append("This region is not the detected solution. It is only used to estimate sigma_R and derivative noise.")
    lines.append("Green shaded region: accepted candidate window.")
    lines.append("This is the first window, moving backward from high current, that satisfies both the derivative and flatness criteria.")
    lines.append("")
    lines.append("Noise estimate")
    lines.append(
        f"The first {p['min_points_plateau']} points in reversed order were used as the reference region."
    )
    lines.append(f"sigma_R = {fmt_num(result['sigma_R'])} ohm")
    lines.append(f"max|dR/dI| in noise region = {fmt_num(result['max_dR_noise'])} ohm/mA")
    lines.append("")
    lines.append("Thresholds used")
    lines.append(f"User derivative threshold = {fmt_num(p['user_deriv_thr'])} ohm/mA")
    lines.append(f"Used slope threshold = {fmt_num(result['slope_thr'])} ohm/mA")
    lines.append(f"Flatness threshold = {fmt_num(result['deltaR_thr'])} ohm")
    lines.append("")
    lines.append("Decision rule")
    lines.append(
        f"A candidate window of {p['consec_points']} consecutive points is accepted only if:"
    )
    lines.append("1. all points satisfy |dR/dI| < slope threshold")
    lines.append("2. the resistance spread in the window is below the flatness threshold")
    lines.append("")
    if result["success"]:
        lines.append("Final result")
        lines.append(f"I_EM = {fmt_num(result['I_em'])} mA")
        lines.append(f"Rmin at I_EM = {fmt_num(result['R_at_Iem'])} ohm")
        lines.append("This point is interpreted as the onset of the accepted transition window identified by the criteria above.")
    else:
        lines.append("Final result")
        lines.append("No candidate window satisfied the present criteria.")
        lines.append("Possible reasons: strict thresholds, insufficient smoothing, or no clear plateau under current settings.")

    return "\n".join(lines)


def build_candidate_explanation(result, cand):
    lines = []
    lines.append("Candidate window interpretation")
    lines.append("")
    lines.append(
        f"Window in original order: indices {cand['true_start_idx']} to {cand['true_stop_idx']}"
    )
    lines.append(
        f"Current span: {fmt_num(cand['I_start_true'])} to {fmt_num(cand['I_stop_true'])} mA"
    )
    lines.append("")
    lines.append("Meaning of the shaded regions")
    lines.append("Gray shaded region in the reverse plot: noise reference region used only to estimate thresholds.")
    lines.append("Green shaded region in the reverse plot: accepted candidate window.")
    lines.append("Green shaded region in the candidate plot: the currently selected candidate window in the original current direction.")
    lines.append("")
    lines.append("Numerical test")
    lines.append(
        f"Maximum |dR/dI| in window = {fmt_num(cand['max_abs_d'])} ohm/mA"
    )
    lines.append(
        f"Slope threshold = {fmt_num(result['slope_thr'])} ohm/mA"
    )
    lines.append(
        f"Resistance spread in window = {fmt_num(cand['delta_R'])} ohm"
    )
    lines.append(
        f"Flatness threshold = {fmt_num(result['deltaR_thr'])} ohm"
    )
    lines.append("")
    lines.append("Pass/fail")
    lines.append(f"Slope criterion passed: {'Yes' if cand['cond_deriv'] else 'No'}")
    lines.append(f"Flatness criterion passed: {'Yes' if cand['cond_flat'] else 'No'}")
    lines.append(f"Overall: {cand['fail_reason']}")
    lines.append("")
    if cand["cond_deriv"] and cand["cond_flat"]:
        lines.append("This is the first candidate window that satisfies both criteria.")
        lines.append("Therefore it is accepted and used to define I_EM.")
    else:
        lines.append("This candidate is rejected because at least one criterion failed.")

    return "\n".join(lines)


# =========================================================
# Detached figure utility
# =========================================================
def open_detached_figure(title, draw_func):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    draw_func(ax)
    fig.tight_layout()
    plt.show()


def open_equations_window():
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    try:
        fig.canvas.manager.set_window_title("I_EM Method Equations")
    except Exception:
        pass

    ax.axis("off")

    text = (
        r"$\mathbf{Working\ variable}$" "\n"
        r"$I \equiv I_{\mathrm{Pulse}}\ \mathrm{(mA)}$" "\n\n"
        r"$\mathbf{Smoothing}$" "\n"
        r"$R_s(I)=\mathrm{moving\ average}\!\left(R_{\min}(I)\right)$" "\n\n"
        r"$\mathbf{Derivative}$" "\n"
        r"$\frac{dR}{dI}=\nabla R_s(I)$" "\n\n"
        r"$\mathbf{Reversed\ arrays}$" "\n"
        r"$I_{\mathrm{rev}} = I[::-1]$" "\n"
        r"$R_{\mathrm{rev}} = R_s[::-1]$" "\n\n"
        r"$\mathbf{Noise\ estimate}$" "\n"
        r"$\sigma_R=\mathrm{std}\!\left(R_{\mathrm{rev}}[0:n]\right)$" "\n"
        r"$\max |dR/dI|_{\mathrm{noise}}=\max\!\left(|(dR/dI)_{\mathrm{rev}}[0:n]|\right)$" "\n\n"
        r"$\mathbf{Slope\ threshold}$" "\n"
        r"$\epsilon_d=\max\!\left(\epsilon_{\mathrm{user}},\,3\,\max |dR/dI|_{\mathrm{noise}}\right)$" "\n\n"
        r"$\mathbf{Flatness\ threshold}$" "\n"
        r"$\Delta R_{\mathrm{thr}} = k\,\sigma_R$" "\n\n"
        r"$\mathbf{Window\ spread}$" "\n"
        r"$\Delta R_{\mathrm{window}}=\max(R_{\mathrm{window}})-\min(R_{\mathrm{window}})$" "\n\n"
        r"$\mathbf{Acceptance\ criteria}$" "\n"
        r"$|dR/dI| < \epsilon_d$" "\n"
        r"$\Delta R_{\mathrm{window}} < \Delta R_{\mathrm{thr}}$" "\n\n"
        r"$\mathbf{Definition\ of}\ I_{\mathrm{EM}}$" "\n"
        r"$I_{\mathrm{EM}} = I[k_{\mathrm{first\ valid\ window}}]$"
    )

    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=13)
    fig.tight_layout()
    plt.show()


# =========================================================
# Plot window
# =========================================================
class PlotWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("I_EM Explorer Plots")
        self.geometry("1500x950")
        self.minsize(1000, 700)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._make_plot_panel(0, 0, "Raw + Smoothed", "raw")
        self._make_plot_panel(0, 1, "Derivative", "deriv")
        self._make_plot_panel(1, 0, "Reverse Search", "reverse")
        self._make_plot_panel(1, 1, "Candidate Window Focus", "candidate")
        self._make_plot_panel(2, 0, "Summary across EM cycles", "summary", colspan=2)

    def _make_plot_panel(self, row, col, title, key, colspan=1):
        frame = ttk.LabelFrame(self, text=title)
        frame.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=6, pady=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        fig = plt.Figure(figsize=(6.2, 4.2), dpi=100)
        ax = fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky="ew")

        setattr(self, f"fig_{key}", fig)
        setattr(self, f"ax_{key}", ax)
        setattr(self, f"canvas_{key}", canvas)

    def clear_all(self):
        for key in ["raw", "deriv", "reverse", "candidate", "summary"]:
            ax = getattr(self, f"ax_{key}")
            canvas = getattr(self, f"canvas_{key}")
            ax.clear()
            ax.set_title("No data loaded")
            canvas.draw()


# =========================================================
# Main app
# =========================================================
class IEMExplorerAppV43(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("I_EM Explorer Controls")
        self.geometry("900x1040")
        self.minsize(760, 800)

        self.file_paths = []
        self.datasets = []
        self.current_index = None
        self.current_result = None
        self.current_candidate_index = None

        self.plot_window = PlotWindow(self)

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self.close_all)
        self.plot_window.protocol("WM_DELETE_WINDOW", self.close_all)

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        file_box = ttk.LabelFrame(self, text="1. Files")
        file_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        btn_row = ttk.Frame(file_box)
        btn_row.pack(fill="x", padx=6, pady=6)

        ttk.Button(btn_row, text="Add CSV files", command=self.add_files).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Add folder", command=self.add_folder).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Clear files", command=self.clear_files).pack(side="left", padx=2)

        self.file_listbox = tk.Listbox(file_box, height=8, exportselection=False)
        self.file_listbox.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.file_listbox.bind("<<ListboxSelect>>", self.on_select_dataset)

        col_box = ttk.LabelFrame(self, text="2. Columns and current unit")
        col_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        col_box.columnconfigure(1, weight=1)

        ttk.Label(col_box, text="Current column").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.current_col_var = tk.StringVar()
        self.current_col_combo = ttk.Combobox(col_box, textvariable=self.current_col_var, state="readonly")
        self.current_col_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(col_box, text="Rmin column").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.rmin_col_var = tk.StringVar()
        self.rmin_col_combo = ttk.Combobox(col_box, textvariable=self.rmin_col_var, state="readonly")
        self.rmin_col_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(col_box, text="Display name").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.display_name_var = tk.StringVar()
        ttk.Entry(col_box, textvariable=self.display_name_var).grid(row=2, column=1, sticky="ew", padx=6, pady=4)

        self.current_in_mA_var = tk.BooleanVar(value=False)
        self.remove_zero_rmin_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            col_box,
            text="Current column is already in mA, do not convert",
            variable=self.current_in_mA_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=2)

        ttk.Checkbutton(
            col_box,
            text="Exclude rows where Rmin = 0",
            variable=self.remove_zero_rmin_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=2)

        ttk.Button(
            col_box,
            text="Apply choices to selected file",
            command=self.apply_column_selection
        ).grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

        param_box = ttk.LabelFrame(self, text="3. Detection parameters")
        param_box.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        param_box.columnconfigure(1, weight=1)

        self.smooth_frac_var = tk.DoubleVar(value=0.05)
        self.deriv_thr_var = tk.DoubleVar(value=0.0008)
        self.consec_var = tk.IntVar(value=4)
        self.rtol_sigma_var = tk.DoubleVar(value=2.0)
        self.min_plateau_pts_var = tk.IntVar(value=10)

        self._add_labeled_entry(param_box, "Smoothing fraction", self.smooth_frac_var, 0)
        self._add_labeled_entry(param_box, "Derivative threshold (Ohm/mA)", self.deriv_thr_var, 1)
        self._add_labeled_entry(param_box, "Consecutive points", self.consec_var, 2)
        self._add_labeled_entry(param_box, "Sigma multiplier", self.rtol_sigma_var, 3)
        self._add_labeled_entry(param_box, "Points for plateau noise", self.min_plateau_pts_var, 4)

        btns = ttk.Frame(param_box)
        btns.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        btns.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(btns, text="Analyze selected", command=self.analyze_selected).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(btns, text="Analyze all", command=self.analyze_all).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(btns, text="Export summary CSV", command=self.export_summary_csv).grid(row=0, column=2, sticky="ew", padx=2)

        view_box = ttk.LabelFrame(self, text="4. Plot mode and detached figures")
        view_box.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        view_box.columnconfigure(1, weight=1)

        ttk.Label(view_box, text="Main plot mode").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.plot_mode_var = tk.StringVar(value="selected")
        plot_mode_combo = ttk.Combobox(
            view_box,
            textvariable=self.plot_mode_var,
            state="readonly",
            values=["selected", "overlay_all", "stack_all"]
        )
        plot_mode_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        plot_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_all_plots())

        ttk.Label(view_box, text="Reverse x-axis").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.reverse_x_mode_var = tk.StringVar(value="point_index")
        reverse_mode_combo = ttk.Combobox(
            view_box,
            textvariable=self.reverse_x_mode_var,
            state="readonly",
            values=["point_index", "current_mA"]
        )
        reverse_mode_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        reverse_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_all_plots())

        ttk.Label(view_box, text="Stack offset, raw/smoothed (ohm)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.stack_offset_raw_var = tk.DoubleVar(value=0.15)
        ttk.Entry(view_box, textvariable=self.stack_offset_raw_var).grid(row=2, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(view_box, text="Stack offset, derivative (ohm/mA)").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.stack_offset_deriv_var = tk.DoubleVar(value=0.20)
        ttk.Entry(view_box, textvariable=self.stack_offset_deriv_var).grid(row=3, column=1, sticky="ew", padx=6, pady=4)

        pop_frame = ttk.Frame(view_box)
        pop_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        for i in range(5):
            pop_frame.columnconfigure(i, weight=1)

        ttk.Button(pop_frame, text="Open Raw", command=self.open_raw_popup).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(pop_frame, text="Open Derivative", command=self.open_deriv_popup).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(pop_frame, text="Open Reverse", command=self.open_reverse_popup).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(pop_frame, text="Open Candidate", command=self.open_candidate_popup).grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(pop_frame, text="Open Summary", command=self.open_summary_popup).grid(row=0, column=4, sticky="ew", padx=2)

        ttk.Button(view_box, text="Refresh plots", command=self.refresh_all_plots).grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6)
        )

        ttk.Button(view_box, text="Open Equations", command=open_equations_window).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6)
        )

        bottom = ttk.Notebook(self)
        bottom.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))

        tab_candidates = ttk.Frame(bottom)
        tab_explain = ttk.Frame(bottom)
        tab_decision = ttk.Frame(bottom)
        tab_help = ttk.Frame(bottom)

        bottom.add(tab_candidates, text="Candidate windows")
        bottom.add(tab_explain, text="Detailed explanation")
        bottom.add(tab_decision, text="Decision summary")
        bottom.add(tab_help, text="Parameter help")

        self._build_candidates_tab(tab_candidates)
        self._build_explain_tab(tab_explain)
        self._build_decision_tab(tab_decision)
        self._build_help_tab(tab_help)

    def _add_labeled_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=4)

    def _build_candidates_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        cand_cols = ("ID", "I_start", "I_stop", "max|dR/dI|", "dR", "SlopeOK", "FlatOK", "Status")
        self.candidate_tree = ttk.Treeview(parent, columns=cand_cols, show="headings")
        for c in cand_cols:
            self.candidate_tree.heading(c, text=c)
            if c == "Status":
                self.candidate_tree.column(c, width=170, anchor="center")
            else:
                self.candidate_tree.column(c, width=90, anchor="center")

        self.candidate_tree.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sc = ttk.Scrollbar(parent, orient="vertical", command=self.candidate_tree.yview)
        sc.grid(row=0, column=1, sticky="ns", pady=6)
        self.candidate_tree.config(yscrollcommand=sc.set)
        self.candidate_tree.bind("<<TreeviewSelect>>", self.on_select_candidate)

    def _build_explain_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.explain_text = tk.Text(parent, wrap="word")
        self.explain_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sc = ttk.Scrollbar(parent, orient="vertical", command=self.explain_text.yview)
        sc.grid(row=0, column=1, sticky="ns", pady=6)
        self.explain_text.config(yscrollcommand=sc.set)

    def _build_decision_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.decision_text = tk.Text(parent, wrap="word")
        self.decision_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sc = ttk.Scrollbar(parent, orient="vertical", command=self.decision_text.yview)
        sc.grid(row=0, column=1, sticky="ns", pady=6)
        self.decision_text.config(yscrollcommand=sc.set)

    def _build_help_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.help_text = tk.Text(parent, wrap="word")
        self.help_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sc = ttk.Scrollbar(parent, orient="vertical", command=self.help_text.yview)
        sc.grid(row=0, column=1, sticky="ns", pady=6)
        self.help_text.config(yscrollcommand=sc.set)

        self.help_text.insert(
            "1.0",
            "Current unit\n"
            "The analysis uses pulse amplitude in mA as the default display and working unit.\n"
            "Leave the checkbox unchecked when the file stores current in A.\n"
            "Check it only when the file is already in mA.\n\n"
            "Plot mode\n"
            "selected: only the selected dataset is shown in the main plots.\n"
            "overlay_all: all analyzed datasets are shown on the same axes.\n"
            "stack_all: all analyzed datasets are shown with vertical offsets.\n\n"
            "Reverse x-axis\n"
            "point_index is usually easier to explain.\n"
            "current_mA shows the reversed current values instead.\n\n"
            "Shaded regions\n"
            "Gray shading is the noise reference region.\n"
            "Green shading is the accepted candidate window.\n\n"
            "Smoothing fraction\n"
            "Defines the moving-average window as a fraction of the total number of points.\n\n"
            "Derivative threshold\n"
            "Sets the minimum strictness for local flatness in dR/dI.\n"
            "The code then compares it with derivative noise in the reversed reference region and uses the larger one.\n\n"
            "Consecutive points\n"
            "Requires stability over several neighboring points, not just one point.\n\n"
            "Sigma multiplier\n"
            "Controls how much resistance spread is allowed inside a candidate plateau window.\n\n"
            "Points for plateau noise\n"
            "Defines how many reversed points are used to estimate sigma_R and derivative noise."
        )
        self.help_text.config(state="disabled")

    # -----------------------------------------------------
    # File loading
    # -----------------------------------------------------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not paths:
            return
        self._append_files(list(paths))

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing CSV files")
        if not folder:
            return
        paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not paths:
            messagebox.showwarning("No CSV files", "No CSV files found in the selected folder.")
            return
        self._append_files(paths)

    def _append_files(self, paths):
        new_paths = []
        existing = set(self.file_paths)
        for p in paths:
            if p not in existing:
                new_paths.append(p)

        if not new_paths:
            return

        start_idx = len(self.datasets) + 1
        for i, path in enumerate(new_paths, start=start_idx):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                messagebox.showwarning("Read error", f"Could not read:\n{path}\n\n{e}")
                continue

            current_guess, rmin_guess, _ = detect_csv_columns(df)
            entry = {
                "path": path,
                "df": df,
                "columns": list(df.columns),
                "current_col": current_guess,
                "rmin_col": rmin_guess,
                "display_name": make_display_name(path, i),
                "result": None,
            }
            self.datasets.append(entry)
            self.file_paths.append(path)

        self.refresh_file_list()

        if self.datasets:
            idx = len(self.datasets) - len(new_paths)
            idx = max(0, idx)
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(idx)
            self.file_listbox.event_generate("<<ListboxSelect>>")

    def clear_files(self):
        self.file_paths.clear()
        self.datasets.clear()
        self.current_index = None
        self.current_result = None
        self.current_candidate_index = None

        self.file_listbox.delete(0, tk.END)
        self.current_col_combo["values"] = []
        self.rmin_col_combo["values"] = []

        self.clear_candidate_table()
        self.set_text(self.explain_text, "")
        self.set_text(self.decision_text, "")
        self.plot_window.clear_all()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for i, ds in enumerate(self.datasets, start=1):
            self.file_listbox.insert(tk.END, f"{i:02d} | {ds['display_name']}")

    # -----------------------------------------------------
    # Selection and preparation
    # -----------------------------------------------------
    def on_select_dataset(self, event=None):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.current_index = idx
        ds = self.datasets[idx]

        cols = ds["columns"]
        self.current_col_combo["values"] = cols
        self.rmin_col_combo["values"] = cols

        self.current_col_var.set(ds["current_col"] if ds["current_col"] in cols else (cols[0] if cols else ""))
        if ds["rmin_col"] in cols:
            self.rmin_col_var.set(ds["rmin_col"])
        elif len(cols) > 1:
            self.rmin_col_var.set(cols[1])
        else:
            self.rmin_col_var.set(cols[0] if cols else "")

        self.display_name_var.set(ds["display_name"])
        self.current_candidate_index = None

        if ds["result"] is not None:
            self.current_result = ds["result"]
            self.update_all_views_for_dataset(ds, ds["result"])
        else:
            self.preview_current_dataset()

    def apply_column_selection(self):
        if self.current_index is None:
            messagebox.showinfo("No file selected", "Please select a file first.")
            return

        ds = self.datasets[self.current_index]
        ds["current_col"] = self.current_col_var.get()
        ds["rmin_col"] = self.rmin_col_var.get()
        ds["display_name"] = self.display_name_var.get().strip() or ds["display_name"]
        self.refresh_file_list()
        self.preview_current_dataset()

    def get_prepared_xy(self, ds):
        df = ds["df"].copy()
        current_col = ds["current_col"]
        rmin_col = ds["rmin_col"]

        if current_col not in df.columns or rmin_col not in df.columns:
            raise ValueError("Selected columns were not found in the dataframe.")

        I = pd.to_numeric(df[current_col], errors="coerce")
        R = pd.to_numeric(df[rmin_col], errors="coerce")

        mask = np.isfinite(I) & np.isfinite(R)
        if self.remove_zero_rmin_var.get():
            mask &= (R != 0)

        I = I[mask].to_numpy(dtype=float)
        R = R[mask].to_numpy(dtype=float)

        if not self.current_in_mA_var.get():
            I = I * 1e3

        if len(I) == 0:
            raise ValueError("No valid data points remain after filtering.")

        order = np.argsort(I)
        I = I[order]
        R = R[order]

        return I, R

    def get_analyzed_datasets(self):
        items = []
        for ds in self.datasets:
            r = ds.get("result", None)
            if r is not None and isinstance(r, dict) and "I" in r and r["I"] is not None:
                items.append(ds)
        return items

    # -----------------------------------------------------
    # Analysis
    # -----------------------------------------------------
    def analyze_selected(self):
        if self.current_index is None:
            messagebox.showinfo("No file selected", "Please select a file first.")
            return

        try:
            ds = self.datasets[self.current_index]
            I, R = self.get_prepared_xy(ds)

            result = compute_detection(
                I_mA=I,
                Rmin=R,
                smooth_frac=safe_float(self.smooth_frac_var.get(), 0.05),
                deriv_thr=safe_float(self.deriv_thr_var.get(), 0.0008),
                consec_points=max(2, safe_int(self.consec_var.get(), 4)),
                r_tol_sigma=safe_float(self.rtol_sigma_var.get(), 2.0),
                min_points_plateau=max(3, safe_int(self.min_plateau_pts_var.get(), 10)),
            )

            ds["result"] = result
            ds["display_name"] = self.display_name_var.get().strip() or ds["display_name"]
            self.current_result = result
            self.current_candidate_index = None

            self.refresh_file_list()
            self.update_all_views_for_dataset(ds, result)
            self.update_summary_plot()

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Analysis error", str(e))

    def analyze_all(self):
        if not self.datasets:
            messagebox.showinfo("No files", "Please load at least one CSV file.")
            return

        errors = []
        for ds in self.datasets:
            try:
                I, R = self.get_prepared_xy(ds)
                ds["result"] = compute_detection(
                    I_mA=I,
                    Rmin=R,
                    smooth_frac=safe_float(self.smooth_frac_var.get(), 0.05),
                    deriv_thr=safe_float(self.deriv_thr_var.get(), 0.0008),
                    consec_points=max(2, safe_int(self.consec_var.get(), 4)),
                    r_tol_sigma=safe_float(self.rtol_sigma_var.get(), 2.0),
                    min_points_plateau=max(3, safe_int(self.min_plateau_pts_var.get(), 10)),
                )
            except Exception as e:
                ds["result"] = {
                    "success": False,
                    "message": str(e),
                    "I_em": np.nan,
                    "R_at_Iem": np.nan,
                    "slope_thr": np.nan,
                    "deltaR_thr": np.nan,
                    "sigma_R": np.nan,
                    "candidate_windows": [],
                    "params": {}
                }
                errors.append(f"{ds['display_name']}: {e}")

        if self.current_index is not None:
            ds = self.datasets[self.current_index]
            if ds["result"] is not None:
                self.current_result = ds["result"]
                self.update_all_views_for_dataset(ds, ds["result"])
        else:
            self.refresh_all_plots()

        self.update_summary_plot()

        if errors:
            messagebox.showwarning("Finished with warnings", "\n".join(errors[:10]))

    # -----------------------------------------------------
    # Drawing logic
    # -----------------------------------------------------
    def get_plot_mode(self):
        return self.plot_mode_var.get().strip()

    def get_reverse_x_mode(self):
        return self.reverse_x_mode_var.get().strip()

    def draw_raw_plot(self, ax):
        ax.clear()
        mode = self.get_plot_mode()

        if mode == "selected":
            if self.current_index is None:
                ax.set_title("No selected dataset")
                return
            ds = self.datasets[self.current_index]
            result = ds.get("result", None)
            if result is None or result.get("I") is None:
                ax.set_title("No analyzed data for selected dataset")
                return
            self._draw_single_raw(ax, ds, result)
            return

        analyzed = self.get_analyzed_datasets()
        if not analyzed:
            ax.set_title("No analyzed datasets")
            return

        offset = safe_float(self.stack_offset_raw_var.get(), 0.15)

        for i, ds in enumerate(analyzed):
            r = ds["result"]
            I = r["I"]
            R = r["R"]
            R_s = r["R_s"]
            color, marker = get_color_and_marker(i)

            y_raw = R.copy()
            y_s = R_s.copy()

            if mode == "stack_all":
                y_raw = apply_vertical_offset(y_raw, offset, i)
                y_s = apply_vertical_offset(y_s, offset, i)

            ax.plot(
                I, y_raw,
                marker=marker, markersize=3.8, lw=0.9,
                markerfacecolor='none', color=color, alpha=0.95,
                label=f"{ds['display_name']} raw"
            )
            ax.plot(
                I, y_s,
                lw=1.6, color=color, alpha=0.95,
                label=f"{ds['display_name']} smooth"
            )

            if r.get("success", False):
                k = r["idx_true_found"]
                ax.scatter(I[k], y_raw[k], s=40, color=color, zorder=5)

        ax.set_xlabel(r"$I_{Pulse}$ (mA)")
        ax.set_ylabel(r"$R_{min}$ ($\Omega$)")
        if mode == "overlay_all":
            ax.set_title("Raw and smoothed, overlay of all analyzed cycles")
        else:
            ax.set_title("Raw and smoothed, stacked view of all analyzed cycles")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(False)

    def _draw_single_raw(self, ax, ds, result):
        I = result["I"]
        R = result["R"]
        R_s = result["R_s"]

        line_raw, = ax.plot(I, R, marker='o', markersize=4.5, lw=1.1, markerfacecolor='none', label="Raw Rmin")
        line_s = None
        if R_s is not None:
            line_s, = ax.plot(I, R_s, lw=1.8, label="Smoothed Rmin")

        handles = [line_raw]
        if line_s is not None:
            handles.append(line_s)

        if result["success"]:
            k = result["idx_true_found"]
            scat = ax.scatter(I[k], R[k], s=55, zorder=5, color=ACCEPT_SHADE_COLOR, label=r"$I_{\mathrm{EM}}$")
            ax.axvline(I[k], ls='--', lw=1.0, color='black')

            if result["accepted_window_true"] is not None:
                i0, i1 = result["accepted_window_true"]
                ax.axvspan(
                    min(I[i0], I[i1]),
                    max(I[i0], I[i1]),
                    color=ACCEPT_SHADE_COLOR,
                    alpha=ACCEPT_SHADE_ALPHA
                )
                handles.append(
                    Patch(facecolor=ACCEPT_SHADE_COLOR, alpha=ACCEPT_SHADE_ALPHA, edgecolor='none', label="Accepted window")
                )

            ax.text(I[k], R[k], f"  I_EM = {fmt_num(result['I_em'])} mA", va='bottom', ha='left', fontsize=9)
            handles.append(scat)

        ax.set_xlabel(r"$I_{Pulse}$ (mA)")
        ax.set_ylabel(r"$R_{min}$ ($\Omega$)")
        ax.set_title(f"Raw and smoothed data: {ds['display_name']}")
        ax.legend(handles=handles, frameon=False, fontsize=9)
        ax.grid(False)

    def draw_derivative_plot(self, ax):
        ax.clear()
        mode = self.get_plot_mode()

        if mode == "selected":
            if self.current_index is None:
                ax.set_title("No selected dataset")
                return
            ds = self.datasets[self.current_index]
            result = ds.get("result", None)
            if result is None or result.get("dR_dI") is None:
                ax.set_title("No analyzed data for selected dataset")
                return
            self._draw_single_derivative(ax, ds, result)
            return

        analyzed = self.get_analyzed_datasets()
        if not analyzed:
            ax.set_title("No analyzed datasets")
            return

        offset = safe_float(self.stack_offset_deriv_var.get(), 0.20)

        for i, ds in enumerate(analyzed):
            r = ds["result"]
            I = r["I"]
            d = r["dR_dI"]
            color, marker = get_color_and_marker(i)

            y = d.copy()
            if mode == "stack_all":
                y = apply_vertical_offset(y, offset, i)

            ax.plot(
                I, y,
                marker=marker, markersize=3.8, lw=1.0,
                markerfacecolor='none', color=color,
                label=ds["display_name"]
            )

            if r.get("success", False):
                k = r["idx_true_found"]
                ax.scatter(I[k], y[k], s=35, color=color, zorder=5)

        if mode == "overlay_all":
            ax.set_title("Derivative overlay of all analyzed cycles")
        else:
            ax.set_title("Derivative stacked view of all analyzed cycles")

        ax.set_xlabel(r"$I_{Pulse}$ (mA)")
        ax.set_ylabel(r"$dR/dI$ ($\Omega$/mA)")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(False)

    def _draw_single_derivative(self, ax, ds, result):
        I = result["I"]
        d = result["dR_dI"]

        line_d, = ax.plot(I, d, marker='o', markersize=4, lw=1.1, markerfacecolor='none', label=r"$dR/dI$")
        line_thr_pos = ax.axhline(result["slope_thr"], ls='--', lw=1.0, label="Used slope threshold")
        ax.axhline(-result["slope_thr"], ls='--', lw=1.0)

        handles = [line_d, line_thr_pos]

        if result["success"]:
            k = result["idx_true_found"]
            scat = ax.scatter(I[k], d[k], s=50, zorder=4, color=ACCEPT_SHADE_COLOR)
            ax.axvline(I[k], ls='--', lw=1.0)
            handles.append(scat)

        ax.set_xlabel(r"$I_{Pulse}$ (mA)")
        ax.set_ylabel(r"$dR/dI$ ($\Omega$/mA)")
        ax.set_title(f"Derivative and slope threshold: {ds['display_name']}")
        ax.legend(handles=handles, frameon=False, fontsize=9)
        ax.grid(False)

    def draw_reverse_plot(self, ax):
        ax.clear()
        mode = self.get_plot_mode()

        if mode == "selected":
            if self.current_index is None:
                ax.set_title("No selected dataset")
                return
            ds = self.datasets[self.current_index]
            result = ds.get("result", None)
            if result is None or result.get("I_rev") is None:
                ax.set_title("No analyzed data for selected dataset")
                return
            self._draw_single_reverse(ax, ds, result)
            return

        analyzed = self.get_analyzed_datasets()
        if not analyzed:
            ax.set_title("No analyzed datasets")
            return

        x_mode = self.get_reverse_x_mode()
        offset = safe_float(self.stack_offset_raw_var.get(), 0.15)

        for i, ds in enumerate(analyzed):
            r = ds["result"]
            color, marker = get_color_and_marker(i)

            if x_mode == "point_index":
                x = r["rev_index"]
                xlabel = "Reversed point index, high current to low current"
            else:
                x = r["I_rev"]
                xlabel = r"Reversed current ($mA$)"

            y = r["R_rev"].copy()
            if mode == "stack_all":
                y = apply_vertical_offset(y, offset, i)

            ax.plot(
                x, y,
                marker=marker, markersize=3.5, lw=1.0,
                markerfacecolor='none', color=color,
                label=ds["display_name"]
            )

            if r.get("success", False):
                idx = r["idx_rev_found"]
                ax.scatter(x[idx], y[idx], s=35, color=color, zorder=5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Smoothed reversed $R_{min}$ ($\Omega$)")
        if mode == "overlay_all":
            ax.set_title("Reverse-search overlay of all analyzed cycles")
        else:
            ax.set_title("Reverse-search stacked view of all analyzed cycles")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(False)

    def _draw_single_reverse(self, ax, ds, result):
        x_mode = self.get_reverse_x_mode()

        if x_mode == "point_index":
            x = result["rev_index"]
            xlabel = "Reversed point index, high current to low current"
        else:
            x = result["I_rev"]
            xlabel = r"Reversed current ($mA$)"

        R_rev = result["R_rev"]
        d_rev = result["dR_dI_rev"]

        ax2 = ax.twinx()

        line_rrev, = ax.plot(x, R_rev, marker='o', markersize=4, lw=1.1, markerfacecolor='none', label="Smoothed R reversed")
        line_drev, = ax2.plot(x, d_rev, lw=1.0, label="dR/dI reversed")

        n0, n1 = result["noise_slice_rev"]
        x0_noise = min(x[n0], x[n1])
        x1_noise = max(x[n0], x[n1])

        ax.axvspan(
            x0_noise, x1_noise,
            color=NOISE_SHADE_COLOR,
            alpha=NOISE_SHADE_ALPHA
        )

        handles = [
            line_rrev,
            Patch(facecolor=NOISE_SHADE_COLOR, alpha=NOISE_SHADE_ALPHA, edgecolor='none', label="Noise reference region"),
            line_drev
        ]

        if result["success"] and result["accepted_window_rev"] is not None:
            idx0, idx1 = result["accepted_window_rev"]
            x0_acc = min(x[idx0], x[idx1])
            x1_acc = max(x[idx0], x[idx1])

            ax.axvspan(
                x0_acc, x1_acc,
                color=ACCEPT_SHADE_COLOR,
                alpha=ACCEPT_SHADE_ALPHA
            )

            scat1 = ax.scatter(x[idx0], R_rev[idx0], s=50, zorder=5, color=ACCEPT_SHADE_COLOR)
            ax2.scatter(x[idx0], d_rev[idx0], s=40, zorder=5, color=ACCEPT_SHADE_COLOR)

            handles.append(
                Patch(facecolor=ACCEPT_SHADE_COLOR, alpha=ACCEPT_SHADE_ALPHA, edgecolor='none', label="Accepted candidate window")
            )
            handles.append(scat1)

            ax.text(
                x[idx0], R_rev[idx0],
                "  Accepted",
                va='bottom', ha='left', fontsize=9, color=ACCEPT_SHADE_COLOR
            )

        ax2.axhline(result["slope_thr"], ls='--', lw=1.0)
        ax2.axhline(-result["slope_thr"], ls='--', lw=1.0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Smoothed reversed $R_{min}$ ($\Omega$)")
        ax2.set_ylabel(r"Reversed $dR/dI$ ($\Omega$/mA)")
        ax.set_title(f"Reverse-search logic: {ds['display_name']}")
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
        ax.grid(False)

    def draw_candidate_plot(self, ax):
        ax.clear()

        if self.current_index is None:
            ax.set_title("No selected dataset")
            return

        ds = self.datasets[self.current_index]
        result = ds.get("result", None)
        if result is None or result.get("I") is None:
            ax.set_title("No analyzed data for selected dataset")
            return

        sel = self.candidate_tree.selection()
        if not sel:
            ax.set_title("No candidate selected")
            return

        idx = int(sel[0])
        cands = result.get("candidate_windows", [])
        if idx < 0 or idx >= len(cands):
            ax.set_title("Invalid candidate selection")
            return

        cand = cands[idx]

        I = result["I"]
        R = result["R"]
        R_s = result["R_s"]
        d = result["dR_dI"]

        ax2 = ax.twinx()

        line_raw, = ax.plot(I, R, marker='o', markersize=3.8, lw=0.9, markerfacecolor='none', label="Raw Rmin")
        line_s, = ax.plot(I, R_s, lw=1.5, label="Smoothed Rmin")
        line_d, = ax2.plot(I, d, lw=0.9, label="dR/dI")

        i0 = cand["true_start_idx"]
        i1 = cand["true_stop_idx"]

        ax.axvspan(
            min(I[i0], I[i1]),
            max(I[i0], I[i1]),
            color=CANDIDATE_SHADE_COLOR,
            alpha=CANDIDATE_SHADE_ALPHA
        )
        ax.plot(I[i0:i1+1], R_s[i0:i1+1], lw=2.8, color=CANDIDATE_SHADE_COLOR)

        thr_line = ax2.axhline(result["slope_thr"], ls='--', lw=1.0, label="Slope threshold")
        ax2.axhline(-result["slope_thr"], ls='--', lw=1.0)

        handles = [
            line_raw,
            line_s,
            line_d,
            thr_line,
            Patch(facecolor=CANDIDATE_SHADE_COLOR, alpha=CANDIDATE_SHADE_ALPHA, edgecolor='none', label="Candidate window")
        ]

        if result["success"]:
            k = result["idx_true_found"]
            ax.axvline(I[k], ls='--', lw=1.0, color='black')
            scat = ax.scatter(I[k], R[k], s=40, color=ACCEPT_SHADE_COLOR, zorder=5, label=r"$I_{\mathrm{EM}}$")
            handles.append(scat)

        ax.set_xlabel(r"$I_{Pulse}$ (mA)")
        ax.set_ylabel(r"$R_{min}$ ($\Omega$)")
        ax2.set_ylabel(r"$dR/dI$ ($\Omega$/mA)")
        ax.set_title("Candidate window in original-order data")
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
        ax.grid(False)

    def draw_summary_plot(self, ax):
        ax.clear()

        valid_names = []
        valid_iem = []
        valid_idx = []

        for i, ds in enumerate(self.datasets, start=1):
            r = ds.get("result", None)
            if r is not None and np.isfinite(r.get("I_em", np.nan)):
                valid_names.append(ds["display_name"])
                valid_iem.append(r["I_em"])
                valid_idx.append(i)

        if valid_iem:
            ax.plot(valid_idx, valid_iem, '-', color='black', lw=1.2, zorder=1)
            for j, (xv, yv, name) in enumerate(zip(valid_idx, valid_iem, valid_names)):
                color, marker = get_color_and_marker(j)
                ax.plot(
                    xv, yv, marker=marker, color=color, lw=0,
                    markersize=7, markerfacecolor='none', markeredgewidth=1.5
                )
            ax.set_xticks(valid_idx)
            ax.set_xticklabels(valid_names, rotation=0)
            ax.set_xlabel("EM cycle")
            ax.set_ylabel(r"$I_{\mathrm{EM}}$ (mA)")
            ax.set_title(r"$I_{\mathrm{EM}}$ across analyzed cycles")
            ax.grid(False)
        else:
            ax.set_title("No valid I_EM values to plot")

    # -----------------------------------------------------
    # Analysis and plot refresh
    # -----------------------------------------------------
    def preview_current_dataset(self):
        if self.current_index is None:
            return
        try:
            ds = self.datasets[self.current_index]
            I, R = self.get_prepared_xy(ds)

            ax = self.plot_window.ax_raw
            ax.clear()
            ax.plot(I, R, marker='o', markersize=4, lw=1.2, markerfacecolor='none')
            ax.set_xlabel(r"$I_{Pulse}$ (mA)")
            ax.set_ylabel(r"$R_{min}$ ($\Omega$)")
            ax.set_title(f"Preview: {ds['display_name']}")
            self.plot_window.fig_raw.tight_layout()
            self.plot_window.canvas_raw.draw()

            for key, title in [
                ("deriv", "Run analysis to compute derivative"),
                ("reverse", "Run analysis to inspect reverse search"),
                ("candidate", "Select a candidate after analysis"),
            ]:
                ax2 = getattr(self.plot_window, f"ax_{key}")
                canvas2 = getattr(self.plot_window, f"canvas_{key}")
                ax2.clear()
                ax2.set_title(title)
                canvas2.draw()

            self.clear_candidate_table()
            self.set_text(self.explain_text, "Run analysis to generate detailed explanations.")
            self.set_text(self.decision_text, "No decision summary yet.")
            self.update_summary_plot()

        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def refresh_all_plots(self):
        self.update_raw_plot()
        self.update_derivative_plot()
        self.update_reverse_plot()
        self.update_candidate_plot_main()
        self.update_summary_plot()

    def update_all_views_for_dataset(self, ds, result):
        self.update_raw_plot()
        self.update_derivative_plot()
        self.update_reverse_plot()
        self.populate_candidate_table(result)
        self.update_decision_text(ds, result)

        if result["candidate_windows"]:
            accepted_idx = None
            for i, cand in enumerate(result["candidate_windows"]):
                if cand["cond_deriv"] and cand["cond_flat"]:
                    accepted_idx = i
                    break
            if accepted_idx is not None:
                self.select_candidate_row(accepted_idx)
            else:
                self.set_text(self.explain_text, build_general_explanation(ds, result))
                self.clear_candidate_plot()
        else:
            self.set_text(self.explain_text, build_general_explanation(ds, result))
            self.clear_candidate_plot()

    def update_raw_plot(self):
        ax = self.plot_window.ax_raw
        self.draw_raw_plot(ax)
        self.plot_window.fig_raw.tight_layout()
        self.plot_window.canvas_raw.draw()

    def update_derivative_plot(self):
        ax = self.plot_window.ax_deriv
        self.draw_derivative_plot(ax)
        self.plot_window.fig_deriv.tight_layout()
        self.plot_window.canvas_deriv.draw()

    def update_reverse_plot(self):
        ax = self.plot_window.ax_reverse
        self.draw_reverse_plot(ax)
        self.plot_window.fig_reverse.tight_layout()
        self.plot_window.canvas_reverse.draw()

    def update_candidate_plot_main(self):
        ax = self.plot_window.ax_candidate
        self.draw_candidate_plot(ax)
        self.plot_window.fig_candidate.tight_layout()
        self.plot_window.canvas_candidate.draw()

    def clear_candidate_plot(self):
        ax = self.plot_window.ax_candidate
        ax.clear()
        ax.set_title("No candidate selected")
        self.plot_window.canvas_candidate.draw()

    def update_summary_plot(self):
        ax = self.plot_window.ax_summary
        self.draw_summary_plot(ax)
        self.plot_window.fig_summary.tight_layout()
        self.plot_window.canvas_summary.draw()

    # -----------------------------------------------------
    # Candidate table
    # -----------------------------------------------------
    def clear_candidate_table(self):
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)

    def populate_candidate_table(self, result):
        self.clear_candidate_table()
        cands = result.get("candidate_windows", [])
        for i, cand in enumerate(cands, start=1):
            self.candidate_tree.insert(
                "", "end", iid=str(i - 1),
                values=(
                    i,
                    fmt_num(cand["I_start_true"]),
                    fmt_num(cand["I_stop_true"]),
                    fmt_num(cand["max_abs_d"]),
                    fmt_num(cand["delta_R"]),
                    "Yes" if cand["cond_deriv"] else "No",
                    "Yes" if cand["cond_flat"] else "No",
                    cand["fail_reason"],
                )
            )

    def select_candidate_row(self, idx):
        iid = str(idx)
        if iid in self.candidate_tree.get_children():
            self.candidate_tree.selection_set(iid)
            self.candidate_tree.focus(iid)
            self.candidate_tree.see(iid)
            self.on_select_candidate()

    def on_select_candidate(self, event=None):
        if self.current_index is None:
            return
        ds = self.datasets[self.current_index]
        result = ds.get("result", None)
        if result is None:
            return

        sel = self.candidate_tree.selection()
        if not sel:
            self.set_text(self.explain_text, build_general_explanation(ds, result))
            return

        idx = int(sel[0])
        cands = result.get("candidate_windows", [])
        if idx < 0 or idx >= len(cands):
            return

        cand = cands[idx]
        self.current_candidate_index = idx

        self.update_candidate_plot_main()
        self.set_text(self.explain_text, build_candidate_explanation(result, cand))

    # -----------------------------------------------------
    # Decision summary
    # -----------------------------------------------------
    def update_decision_text(self, ds, result):
        self.set_text(self.decision_text, build_general_explanation(ds, result))

    # -----------------------------------------------------
    # Detached pop-up plots
    # -----------------------------------------------------
    def open_raw_popup(self):
        open_detached_figure("I_EM Explorer - Raw", self.draw_raw_plot)

    def open_deriv_popup(self):
        open_detached_figure("I_EM Explorer - Derivative", self.draw_derivative_plot)

    def open_reverse_popup(self):
        open_detached_figure("I_EM Explorer - Reverse", self.draw_reverse_plot)

    def open_candidate_popup(self):
        open_detached_figure("I_EM Explorer - Candidate", self.draw_candidate_plot)

    def open_summary_popup(self):
        open_detached_figure("I_EM Explorer - Summary", self.draw_summary_plot)

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------
    def export_summary_csv(self):
        if not self.datasets:
            messagebox.showinfo("No data", "No datasets are loaded.")
            return

        rows = []
        for i, ds in enumerate(self.datasets, start=1):
            r = ds.get("result", None)
            rows.append({
                "Index": i,
                "Display": ds["display_name"],
                "Path": ds["path"],
                "CurrentColumn": ds["current_col"],
                "RminColumn": ds["rmin_col"],
                "CurrentAlreadyInmA": self.current_in_mA_var.get(),
                "I_EM_mA": r.get("I_em", np.nan) if r else np.nan,
                "R_at_IEM_ohm": r.get("R_at_Iem", np.nan) if r else np.nan,
                "sigma_R": r.get("sigma_R", np.nan) if r else np.nan,
                "max_dR_noise": r.get("max_dR_noise", np.nan) if r else np.nan,
                "slope_thr": r.get("slope_thr", np.nan) if r else np.nan,
                "deltaR_thr": r.get("deltaR_thr", np.nan) if r else np.nan,
                "Success": r.get("success", False) if r else False,
                "Message": r.get("message", "Not analyzed") if r else "Not analyzed",
                "smooth_frac": r.get("params", {}).get("smooth_frac", np.nan) if r else np.nan,
                "used_window_points": r.get("params", {}).get("used_window_points", np.nan) if r else np.nan,
                "user_deriv_thr": r.get("params", {}).get("user_deriv_thr", np.nan) if r else np.nan,
                "consec_points": r.get("params", {}).get("consec_points", np.nan) if r else np.nan,
                "r_tol_sigma": r.get("params", {}).get("r_tol_sigma", np.nan) if r else np.nan,
                "min_points_plateau": r.get("params", {}).get("min_points_plateau", np.nan) if r else np.nan,
            })

        out = pd.DataFrame(rows)
        path = filedialog.asksaveasfilename(
            title="Save summary CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        out.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Summary saved to:\n{path}")

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------
    def set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def close_all(self):
        try:
            if self.plot_window.winfo_exists():
                self.plot_window.destroy()
        except Exception:
            pass
        self.destroy()


# =========================================================
# Run app
# =========================================================
if __name__ == "__main__":
    app = IEMExplorerAppV43()
    app.mainloop()
