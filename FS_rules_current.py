!/usr/bin/env python3
"""
FS_rules_Hzband_textreport_final.py
────────────────────────────────────
Analyze impedance sweeps and select worst cases.

Highlights
    • Optional absolute-rule analysis controlled by configuration.
    • Each relative rule stores up to ``MAX_REL_CASES`` cases.
    • Generates plain-text reports and sequence plots.
    • Absolute winners are visualized separately when enabled.

Outputs
    – ``absolute_worst_cases.txt``
    – ``relative_worst_cases.txt``
    – ``absolute_cases.png`` (when absolute rules are enabled)
    – ``positive_sequence.png``
    – ``zero_sequence.png``
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ───── CONFIGURATION PARAMETERS ──────────────────────────────────────────
INCLUDE_NEGATIVE_PEAKS = True         # If True, also detect negative peaks (dips) in X/Z curves
ENABLE_ABSOLUTE = False           # True = run R1–R4, R1_0–R4_0, C3; False = skip them
FUND = 60.0                      # Fundamental frequency [Hz]
MAX_HARMONIC = 4                 # Highest harmonic to check (2..MAX_HARMONIC)
HARMONIC_BAND_HZ = 5.0  # ±Hz band around each harmonic (e.g. ±5 Hz of n·FUND)
Z_REF, X_REF = 100.0, 50.0       # Reference impedances [Ω]
CLUSTER_BAND = 0.03              # ±3% clustering for envelope rule (C3)
ENV_Z_SHIFT = float(os.getenv("ENV_Z_SHIFT", "0.05"))  # Fractional |Z| difference for C3
MIN_REL_LIST = 5                 # Minimum number of relative worst cases
MAX_REL_CASES = 5                # Max cases to keep per relative rule
PEAK_PROMINENCE = None           # Prominence for find_peaks (None = no filtering)
BOOK = Path("fs_seet.xlsx")     # Input workbook
ABS_OUT = Path("absolute_worst_cases.txt")
REL_OUT = Path("relative_worst_cases.txt")
FIG_POS = Path("positive_sequence.png")
FIG_ZERO = Path("zero_sequence.png")
FIG_ABS = Path("absolute_cases.png")

# Comments on configuration:
#  - ENABLE_ABSOLUTE: set to False to skip all absolute‐rule logic (R1–R4, R1_0–R4_0, C3).
#  - FUND: if your system is 50 Hz, change to 50.0.
#  - MAX_HARMONIC: number of harmonics to check (e.g. 4 = 2nd, 3rd, 4th).
#  - HARMONIC_BAND_HZ: window size around each harmonic. With FUND=60 and n=2,
#      the band spans 120±5 Hz (115–125 Hz).
#  - PEAK_PROMINENCE: threshold for scipy.find_peaks; if None, all local maxima count.
#  - Z_REF, X_REF, CLUSTER_BAND, ENV_Z_SHIFT: absolute‐rule thresholds. ENV_Z_SHIFT
#    can also be set via the environment variable ``ENV_Z_SHIFT``.

# ─────────────────────────────────────────────────────────────────────────
# Build LABELS dictionary including new exact/peak tags
LABELS = {
    # Absolute rules (positive)
    "R1":   "Largest |Z1| at 2H/3H (bin peak, |X1|≤0.05·|Z1|)",
    "R2":   "Highest Q1·|Z1| peak (Z1_pk>5·Z_ref & Q1>3)",
    "R3":   "Deep capacitive well1 (X1_min<−4·X_ref, R1_min<10)",
    "R4":   "Low-freq Z1 (<0.8·FUND) with Q1>2",
    # Absolute rules (zero)
    "R1_0": "Largest |Z0| at 2H/3H (bin peak, |X0|≤0.05·|Z0|)",
    "R2_0": "Highest Q0·|Z0| peak (Z0_pk>5·Z_ref & Q0>3)",
    "R3_0": "Deep capacitive well0 (X0_min<−4·X_ref, R0_min<10)",
    "R4_0": "Low-freq Z0 (<0.8·FUND) with Q0>2",
    "C3":   "Envelope: same f_pk, |Z1_pk| differs >10%",

    # Peer metrics (perfect harmonic exactly)
    # Positive exact
    **{f"H-Z_exact({n}H)": f"Max |Z1| exactly at {n}·FUND" for n in range(2, MAX_HARMONIC+1)},
    **{f"H-X_exact({n}H)": f"Max |X1| exactly at {n}·FUND" for n in range(2, MAX_HARMONIC+1)},
    **{f"H-Q_exact({n}H)": f"Max Q1 exactly at {n}·FUND" for n in range(2, MAX_HARMONIC+1)},
    # Positive bin‐peak
    **{f"H-Z_peak({n}H)": f"Max |Z1| peak in ±{HARMONIC_BAND_HZ}Hz of {n}·FUND"
      for n in range(2, MAX_HARMONIC+1)},
    **{f"H-X_peak({n}H)": f"Max |X1| peak in ±{HARMONIC_BAND_HZ}Hz of {n}·FUND"
@@ -287,51 +289,51 @@ def main():
            if rule in ("R1", "R1_0"):
                key_peak = "Z1_peak_2" if rule == "R1" else "Z0_peak_2"
                Zvals2 = meta.loc[candidates, key_peak]
                Zvals3 = meta.loc[candidates, key_peak.replace("2", "3")]
                Z_combined = pd.DataFrame({"2nd": Zvals2, "3rd": Zvals3}).max(axis=1)
                winner = Z_combined.idxmax()
            else:
                key_map = {
                    "R2":  "Z1_pk",
                    "R3":  "X1_min",
                    "R4":  "f_pk",
                    "R2_0":"Z0_pk",
                    "R3_0":"X0_min",
                    "R4_0":"f_pk"
                }
                key = key_map[rule]
                if "min" in key:
                    winner = meta.loc[candidates, key].idxmin()
                else:
                    winner = meta.loc[candidates, key].idxmax()
    
            sel_abs[winner] = rule
            print(f"    {rule} → {winner}")
    
        # 5. Envelope rule C3
        print(f"▶ 5. Applying envelope rule C3 (ENV_Z_SHIFT={ENV_Z_SHIFT}) …")
        for c in cases:
            if c in sel_abs:
                continue
            for w in sel_abs:
                same = abs(meta.at[c, "f_pk"] - meta.at[w, "f_pk"]) / meta.at[w, "f_pk"] < CLUSTER_BAND
                big = abs(meta.at[c, "Z1_pk"] - meta.at[w, "Z1_pk"]) / meta.at[w, "Z1_pk"] > ENV_Z_SHIFT
                if same and big:
                    sel_abs[c] = "C3"
                    print(f"    C3 → {c}")
                    break
        print(f"   Absolute count = {len(sel_abs)}")
    else:
        print("▶ Skipping absolute‐rule analysis (ENABLE_ABSOLUTE=False)")
    
    # ───── 6. Build Relative Peer‐Metric List ─────────────────────────────────
    print("▶ 6. Building relative peer‐metric list …")
    peer_pool = [c for c in cases if c not in sel_abs]
    peer_rule_cases = {}
    peer_first_tag = {}
    
    # Build REL_ORDER dynamically: exact then peak for each harmonic, positive then zero
    for n in range(2, MAX_HARMONIC + 1):
        REL_ORDER.extend([
            f"H-Z_exact({n}H)", f"H-X_exact({n}H)", f"H-Q_exact({n}H)",
            f"H-Z_peak({n}H)", f"H-X_peak({n}H)", f"H-Q_peak({n}H)"
@@ -487,123 +489,142 @@ def main():
    
    abs_map = {}
    for case, tag in sel_abs.items():
        abs_map.setdefault(tag, []).append(case)
    emit(ABS_OUT, abs_map, "ABSOLUTE RULE LIST", ABS_ORDER)
    emit(REL_OUT, peer_rule_cases, "RELATIVE TOP-5 LIST", REL_ORDER)
    
    # ───── 9. Plotting Selected Cases with Harmonic Lines ────────────────────
    print("▶ 9. Plotting selected cases …")
    all_cases = list(peer_first_tag.keys())
    case_expl = {c: LABELS.get(peer_first_tag[c], "") for c in all_cases}
    
    pos_cases = [
        c for c in all_cases
        if not peer_first_tag[c].startswith(("H-X0", "H-Z0", "H-Q0", "Q0-R", "R0-Min", "E-Zero0", "ΣZ0"))
    ]
    zero_cases = [c for c in all_cases if c not in pos_cases]
    
    harmonic = freqs / FUND
    harmonics = np.arange(1, MAX_HARMONIC + 1)
    bin_halfwidth = HARMONIC_BAND_HZ / FUND
    
    def reserve_and_legend(fig, axs, handles, raw_labels, tag_map, expl_map):
        labels = [f"{c}: {tag_map[c]} – {expl_map[c]}" for c in raw_labels]
        n = len(labels)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        line_height = 0.03
        legend_height = nrows * line_height
        bottom_margin = 0.05 + legend_height + 0.02
        width, height = fig.get_size_inches()
        if bottom_margin > 0.5:
            scale = bottom_margin / 0.5
            fig.set_size_inches(width, height * scale, forward=True)
            bottom_margin = 0.5
        fig.subplots_adjust(top=0.95, bottom=bottom_margin, hspace=0.3)
        y_anchor = bottom_margin / 2
        fig.legend(handles, labels, loc="lower center", ncol=ncols,
                   frameon=False, fontsize="small", bbox_to_anchor=(0.5, y_anchor))
    
    def style_for_case(case, tag_map):
        tag = tag_map.get(case, "")
        label = LABELS.get(tag, "")
        style = {}
        if "Highest" in label:
            style["linewidth"] = 2.5
        if "peak" in tag:
            style["linestyle"] = ":"
        return style

    def plot_sequence(axs, metrics, cases, label_func, tag_map):
        """Plot each metric for the given cases."""
        for ax, (_, df, ylabel) in zip(axs, metrics):
            for n in harmonics:
                ax.axvline(n, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axvline(n - bin_halfwidth, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.axvline(n + bin_halfwidth, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            for c in cases:
                style = style_for_case(c, tag_map)
                ax.plot(harmonic, df[c], label=label_func(c), **style)
            ax.set_ylabel(ylabel)
            ax.set_xticks(harmonics)
            ax.set_xticklabels([f"{n}H" for n in harmonics])
        axs[-1].set_xlabel("Harmonic Number (n)")
        return axs[0].get_legend_handles_labels()
    
    # 9a. Positive‐sequence plots (X1, R1, X1/R1, Z1)
    print("▶ 9a. Positive‐sequence plots …")
    metrics_pos = [
        ("X1", X1, "X1 (Ω)"),
        ("R1", R1, "R1 (Ω)"),
        ("X1/R1", X1.div(R1.replace(0, np.nan)), "X1/R1"),
        ("Z1", Z1, "Z1 (Ω)"),
    ]
    fig1, axs1 = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    h1, labs1 = plot_sequence(axs1, metrics_pos, pos_cases, lambda c: c, peer_first_tag)

    reserve_and_legend(fig1, axs1, h1, labs1, peer_first_tag, case_expl)

    fig1.savefig(FIG_POS, dpi=300)
    print(f"   ↳ saved {FIG_POS.name}")
    
    # 9b. Zero‐sequence plots (X0, R0, X0/R0, Z0)
    print("▶ 9b. Zero‐sequence plots …")
    metrics_zero = [
        ("X0", X0, "X0 (Ω)"),
        ("R0", R0, "R0 (Ω)"),
        ("X0/R0", X0.div(R0.replace(0, np.nan)), "X0/R0"),
        ("Z0", Z0, "Z0 (Ω)"),
    ]
    fig2, axs2 = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
    h2, labs2 = plot_sequence(axs2, metrics_zero, zero_cases, lambda c: c, peer_first_tag)
    reserve_and_legend(fig2, axs2, h2, labs2, peer_first_tag, case_expl)
    fig2.savefig(FIG_ZERO, dpi=300)
    print(f"   ↳ saved {FIG_ZERO.name}")

    # 9c. Absolute-rule plots
    if ENABLE_ABSOLUTE and sel_abs:
        print("▶ 9c. Absolute-rule plots …")
        abs_cases = list(sel_abs.keys())
        abs_first_tag = {c: sel_abs[c] for c in abs_cases}
        abs_case_expl = {c: LABELS.get(sel_abs[c], "") for c in abs_cases}
        pos_abs = [c for c in abs_cases if not sel_abs[c].endswith("_0")]
        zero_abs = [c for c in abs_cases if sel_abs[c].endswith("_0")]

        fig3, axs3 = plt.subplots(4, 2, figsize=(14, 15), sharex="col")
        handles, labels = [], []
        if pos_abs:
            h3p, lab3p = plot_sequence(axs3[:, 0], metrics_pos, pos_abs, lambda c: c, abs_first_tag)
            handles.extend(h3p)
            labels.extend(lab3p)
        else:
            for ax in axs3[:, 0]:
                ax.axis("off")
        if zero_abs:
            h3z, lab3z = plot_sequence(axs3[:, 1], metrics_zero, zero_abs, lambda c: c, abs_first_tag)
            handles.extend(h3z)
            labels.extend(lab3z)
        else:
            for ax in axs3[:, 1]:
                ax.axis("off")
        reserve_and_legend(fig3, axs3.ravel(), handles, labels, abs_first_tag, abs_case_expl)
        fig3.savefig(FIG_ABS, dpi=300)
        print(f"   ↳ saved {FIG_ABS.name}")

    plt.show()
    
    
    
    
    # ───── Append Plain-Text Report at End ─────
    def write_text_report(path):
        try:
            from collections import defaultdict
            lines = []
            lines.append("Frequency Sweep – Worst-Case Harmonic Selection Report")
            lines.append("=" * 60)
            lines.append(f"Fundamental Frequency: {FUND} Hz")
            lines.append(f"Harmonic Range: 2H to {MAX_HARMONIC}H")
            lines.append(f"Bin Width: ±{HARMONIC_BAND_HZ} Hz")
            lines.append(f"Include Negative Peaks: {'Yes' if INCLUDE_NEGATIVE_PEAKS else 'No'}")
README.md
+1
-1

# Frequency Sweep Worst-Case Selection

This repository contains the Python scripts for analyzing impedance sweeps stored in an Excel workbook and identifying worst cases according to absolute and relative rules.

`FS_rules_current.py` is the most up-to-date version of the tool. The original file `FS_rules_Hzband_textreport_final.py` is preserved alongside it for reference.

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, `scipy`

## Input Workbook

Provide `fs_seet.xlsx` with sheets `R1`, `X1`, `R0`, and `X0`. Each sheet should use frequency as the index and cases as columns.

## Running

```bash
python FS_rules_current.py
```

`FS_rules_current.py` represents the latest code, while the previous `FS_rules_Hzband_textreport_final.py` is kept in the repository for historical reference.

Adjust configuration constants at the top of the script to change behaviour. In particular, `MAX_REL_CASES` controls how many cases are kept for each relative rule. `ENV_Z_SHIFT` sets the envelope-rule difference threshold used when absolute rules are enabled. You may override it by defining the environment variable before running, e.g. `ENV_Z_SHIFT=0.1 python FS_rules_current.py`. (The parameter has no effect unless `ENABLE_ABSOLUTE` is `True`.)

## Outputs

- `absolute_worst_cases.txt`
- `relative_worst_cases.txt`
- `worst_case_report.txt`
- `absolute_cases.png` – plots of absolute-rule winners (only when absolute rules run)
- `positive_sequence.png`
- `zero_sequence.png`