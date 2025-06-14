#!/usr/bin/env python3
"""
FS_rules_ref.py
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
import warnings

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
BOOK = Path("FS_sweep.xlsx")     # Input workbook
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
      for n in range(2, MAX_HARMONIC+1)},
    **{f"H-Q_peak({n}H)": f"Max Q1 peak in ±{HARMONIC_BAND_HZ}Hz of {n}·FUND"
      for n in range(2, MAX_HARMONIC+1)},

    # Zero exact
    **{f"H-Z0_exact({n}H)": f"Max |Z0| exactly at {n}·FUND" for n in range(2, MAX_HARMONIC+1)},
    **{f"H-X0_exact({n}H)": f"Max |X0| exactly at {n}·FUND" for n in range(2, MAX_HARMONIC+1)},
    **{f"H-Q0_exact({n}H)": f"Max Q0 exactly at {n}·FUND" for n in range(2, MAX_HARMONIC+1)},
    # Zero bin‐peak
    **{f"H-Z0_peak({n}H)": f"Max |Z0| peak in ±{HARMONIC_BAND_HZ}Hz of {n}·FUND"
      for n in range(2, MAX_HARMONIC+1)},
    **{f"H-X0_peak({n}H)": f"Max |X0| peak in ±{HARMONIC_BAND_HZ}Hz of {n}·FUND"
      for n in range(2, MAX_HARMONIC+1)},
    **{f"H-Q0_peak({n}H)": f"Max Q0 peak in ±{HARMONIC_BAND_HZ}Hz of {n}·FUND"
      for n in range(2, MAX_HARMONIC+1)},

    # Global peer metrics
    "Q-R":   "Highest Q1 at any peak",
    "R-Min": "Minimum R1 over all freq",
    "E-Zero":"Earliest X1 zero crossing / min freq",
    "ΣZ":    "Largest Σ|Z1| (energy)",

    "Q0-R":   "Highest Q0 at any peak",
    "R0-Min": "Minimum R0 over all freq",
    "E-Zero0":"Earliest X0 zero crossing / min freq",
    "ΣZ0":    "Largest Σ|Z0| (energy)",

    # Fallback
    "TopUp": "Peer-top-up by ladder score"
}

ABS_ORDER = ["R1","R2","R3","R4","R1_0","R2_0","R3_0","R4_0","C3"]


def load_data():
    """Load the workbook and return frequency index and all dataframes."""
    print(f"▶ 1. Loading {BOOK.name} …")
    try:
        R1 = pd.read_excel(BOOK, "R1", index_col=0)
        X1 = pd.read_excel(BOOK, "X1", index_col=0)
        R0 = pd.read_excel(BOOK, "R0", index_col=0)
        X0 = pd.read_excel(BOOK, "X0", index_col=0)
    except Exception as e:
        sys.exit(f"❌ Cannot open {BOOK}: {e}")

    freqs = R1.index.to_numpy()
    cases = R1.columns
    return freqs, cases, R1, X1, R0, X0


def compute_metrics(freqs, cases, R1, X1, R0, X0):
    """Compute base metrics and return them along with meta dataframe."""
    print("▶ 2. Computing base metrics (Z1, Z0, Q1, Q0, etc.) …")
    Z1 = np.hypot(R1, X1)
    Q1 = (X1.abs() / R1.replace(0, np.nan)).fillna(np.inf)
    Z0 = np.hypot(R0, X0)
    Q0 = (X0.abs() / R0.replace(0, np.nan)).fillna(np.inf)

    meta = pd.DataFrame(index=cases)

    # 2.1 Global peaks (positive-sequence)
    meta["Z1_pk"] = Z1.max()
    meta["f_pk"] = Z1.idxmax()
    meta["Q1_pk"] = [Q1.at[meta.at[c, "f_pk"], c] for c in cases]

    # 2.2 Global peaks (zero-sequence)
    meta["Z0_pk"] = Z0.max()
    meta["Q0_pk"] = [Q0.at[meta.at[c, "f_pk"], c] for c in cases]

    # 2.3 Harmonic-exact and harmonic-bin-peak metrics
    for seq_label, Zdf, Xdf, Rdf in [
        ("1", Z1, X1, R1),  # positive-sequence
        ("0", Z0, X0, R0)   # zero-sequence
    ]:
        for n in range(2, MAX_HARMONIC + 1):
            f_target = n * FUND
            idx_nearest = np.abs(freqs - f_target).argmin()
            col_exact_Z = f"Z{seq_label}_exact_{n}"
            col_exact_X = f"X{seq_label}_exact_{n}"
            col_exact_R = f"R{seq_label}_exact_{n}"
            col_exact_Q = f"Q{seq_label}_exact_{n}"
            meta[col_exact_Z] = Zdf.iloc[idx_nearest]
            meta[col_exact_X] = Xdf.iloc[idx_nearest]
            meta[col_exact_R] = Rdf.iloc[idx_nearest]
            meta[col_exact_Q] = (
                meta[col_exact_X].abs() / meta[col_exact_R].replace(0, np.nan)
            ).fillna(np.inf)

            mask = (freqs >= f_target - HARMONIC_BAND_HZ) & (
                freqs <= f_target + HARMONIC_BAND_HZ
            )
            if mask.sum() == 0:
                meta[f"Z{seq_label}_peak_{n}"] = 0.0
                meta[f"X{seq_label}_peak_{n}"] = 0.0
                meta[f"Q{seq_label}_peak_{n}"] = 0.0
                continue

            Z_band = Zdf.loc[mask, :]
            X_band = Xdf.loc[mask, :]
            R_band = Rdf.loc[mask, :]

            Z_peak = pd.Series(index=cases, dtype=float)
            X_at_peak = pd.Series(index=cases, dtype=float)
            R_at_peak = pd.Series(index=cases, dtype=float)
            Q_peak = pd.Series(index=cases, dtype=float)

            for c in cases:
                values = Z_band[c].to_numpy()
                if INCLUDE_NEGATIVE_PEAKS:
                    peaks_pos, _ = find_peaks(values, prominence=PEAK_PROMINENCE)
                    peaks_neg, _ = find_peaks(-values, prominence=PEAK_PROMINENCE)
                    peaks = np.concatenate([peaks_pos, peaks_neg])
                else:
                    peaks, _ = find_peaks(values, prominence=PEAK_PROMINENCE)

                if peaks.size > 0:
                    best_idx = peaks[np.argmax(values[peaks])]
                    freq_of_peak = freqs[mask][best_idx]
                    Z_peak[c] = values[best_idx]
                    X_at_peak[c] = X_band.at[freq_of_peak, c]
                    R_at_peak[c] = R_band.at[freq_of_peak, c]
                    Q_peak[c] = abs(X_at_peak[c]) / (
                        R_at_peak[c] if R_at_peak[c] != 0 else np.nan
                    )
                else:
                    Z_peak[c] = 0.0
                    X_at_peak[c] = 0.0
                    R_at_peak[c] = np.nan
                    Q_peak[c] = 0.0

            meta[f"Z{seq_label}_peak_{n}"] = Z_peak
            meta[f"X{seq_label}_peak_{n}"] = X_at_peak
            meta[f"Q{seq_label}_peak_{n}"] = Q_peak.fillna(np.inf)

    # 2.4 Min values and energy (area under |Z| curve)
    meta["X1_min"], meta["R1_min"] = X1.min(), R1.min()
    meta["X0_min"], meta["R0_min"] = X0.min(), R0.min()
    meta["ΣZ1"] = pd.Series(
        np.trapezoid(Z1.to_numpy(), x=freqs, axis=0), index=cases
    )
    meta["ΣZ0"] = pd.Series(
        np.trapezoid(Z0.to_numpy(), x=freqs, axis=0), index=cases
    )

    return meta, Z1, Z0, Q1, Q0


def apply_absolute_rules(meta, cases):
    """Return a mapping of case -> absolute rule tag."""
    tags_abs = {c: [] for c in cases}
    sel_abs = {}

    if ENABLE_ABSOLUTE:
        print("▶ 3. Applying absolute rules …")
        for c in cases:
            fpk = meta.at[c, "f_pk"]

            Z2p = meta.at[c, "Z1_peak_2"]
            Z3p = meta.at[c, "Z1_peak_3"]
            X2p = abs(meta.at[c, "X1_peak_2"])
            X3p = abs(meta.at[c, "X1_peak_3"])
            Zpeak = max(Z2p, Z3p)
            X_at_Zpeak = X2p if Z2p >= Z3p else X3p
            if (Zpeak > 5 * Z_REF) and (X_at_Zpeak <= 0.05 * Zpeak):
                tags_abs[c].append("R1")

            Zpk1, Qpk1 = meta.at[c, "Z1_pk"], meta.at[c, "Q1_pk"]
            if (Zpk1 > 5 * Z_REF) and (Qpk1 > 3):
                tags_abs[c].append("R2")

            Xmin1, Rmin1 = meta.at[c, "X1_min"], meta.at[c, "R1_min"]
            if (Xmin1 < -4 * X_REF) and (Rmin1 < 10):
                tags_abs[c].append("R3")

            if (fpk < 0.8 * FUND) and (Qpk1 > 2):
                tags_abs[c].append("R4")

            Z2p0 = meta.at[c, "Z0_peak_2"]
            Z3p0 = meta.at[c, "Z0_peak_3"]
            X2p0 = abs(meta.at[c, "X0_peak_2"])
            X3p0 = abs(meta.at[c, "X0_peak_3"])
            Zpeak0 = max(Z2p0, Z3p0)
            X_at_Zpeak0 = X2p0 if Z2p0 >= Z3p0 else X3p0
            if (Zpeak0 > 5 * Z_REF) and (X_at_Zpeak0 <= 0.05 * Zpeak0):
                tags_abs[c].append("R1_0")

            Zpk0, Qpk0 = meta.at[c, "Z0_pk"], meta.at[c, "Q0_pk"]
            if (Zpk0 > 5 * Z_REF) and (Qpk0 > 3):
                tags_abs[c].append("R2_0")

            Xmin0, Rmin0 = meta.at[c, "X0_min"], meta.at[c, "R0_min"]
            if (Xmin0 < -4 * X_REF) and (Rmin0 < 10):
                tags_abs[c].append("R3_0")

            if (fpk < 0.8 * FUND) and (Qpk0 > 2):
                tags_abs[c].append("R4_0")

        print("   Hit counts:")
        for rule in ABS_ORDER[:-1]:
            count = sum(rule in tags_abs[c] for c in cases)
            print(f"    {rule}: {count}")

        print("▶ 4. Selecting winners for each absolute rule …")
        for rule in ABS_ORDER[:-1]:
            candidates = [c for c in cases if rule in tags_abs[c]]
            if not candidates:
                continue

            if rule in ("R1", "R1_0"):
                key_peak = "Z1_peak_2" if rule == "R1" else "Z0_peak_2"
                Zvals2 = meta.loc[candidates, key_peak]
                Zvals3 = meta.loc[candidates, key_peak.replace("2", "3")]
                Z_combined = pd.DataFrame({"2nd": Zvals2, "3rd": Zvals3}).max(axis=1)
                winner = Z_combined.idxmax()
            else:
                key_map = {
                    "R2": "Z1_pk",
                    "R3": "X1_min",
                    "R4": "f_pk",
                    "R2_0": "Z0_pk",
                    "R3_0": "X0_min",
                    "R4_0": "f_pk",
                }
                key = key_map[rule]
                if "min" in key:
                    winner = meta.loc[candidates, key].idxmin()
                else:
                    winner = meta.loc[candidates, key].idxmax()

            sel_abs[winner] = rule
            print(f"    {rule} → {winner}")

        print(f"▶ 5. Applying envelope rule C3 (ENV_Z_SHIFT={ENV_Z_SHIFT}) …")
        for c in cases:
            if c in sel_abs:
                continue
            for w in sel_abs:
                same = (
                    abs(meta.at[c, "f_pk"] - meta.at[w, "f_pk"]) / meta.at[w, "f_pk"]
                    < CLUSTER_BAND
                )
                big = (
                    abs(meta.at[c, "Z1_pk"] - meta.at[w, "Z1_pk"]) / meta.at[w, "Z1_pk"]
                    > ENV_Z_SHIFT
                )
                if same and big:
                    sel_abs[c] = "C3"
                    print(f"    C3 → {c}")
                    break
        print(f"   Absolute count = {len(sel_abs)}")
    else:
        print("▶ Skipping absolute-rule analysis (ENABLE_ABSOLUTE=False)")

    return sel_abs


def select_relative_cases(meta, cases, sel_abs):
    """Build relative peer-metric mapping and return rule cases and order."""
    print("▶ 6. Building relative peer-metric list …")
    peer_pool = [c for c in cases if c not in sel_abs]
    peer_rule_cases = {}
    peer_first_tag = {}

    rel_order = []
    for n in range(2, MAX_HARMONIC + 1):
        rel_order.extend(
            [
                f"H-Z_exact({n}H)",
                f"H-X_exact({n}H)",
                f"H-Q_exact({n}H)",
                f"H-Z_peak({n}H)",
                f"H-X_peak({n}H)",
                f"H-Q_peak({n}H)",
            ]
        )
    for n in range(2, MAX_HARMONIC + 1):
        rel_order.extend(
            [
                f"H-Z0_exact({n}H)",
                f"H-X0_exact({n}H)",
                f"H-Q0_exact({n}H)",
                f"H-Z0_peak({n}H)",
                f"H-X0_peak({n}H)",
                f"H-Q0_peak({n}H)",
            ]
        )
    rel_order.extend([
        "Q-R",
        "R-Min",
        "E-Zero",
        "ΣZ",
        "Q0-R",
        "R0-Min",
        "E-Zero0",
        "ΣZ0",
        "TopUp",
    ])

    for n in range(2, MAX_HARMONIC + 1):
        Z_exact = meta[f"Z1_exact_{n}"].loc[peer_pool]
        X_exact = meta[f"X1_exact_{n}"].abs().loc[peer_pool]
        Q_exact = meta[f"Q1_exact_{n}"].loc[peer_pool]
        if not Z_exact.empty:
            for c in Z_exact.nlargest(MAX_REL_CASES).index:
                peer_rule_cases.setdefault(f"H-Z_exact({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Z_exact({n}H)")
        if not X_exact.empty:
            for c in X_exact.nlargest(MAX_REL_CASES).index:
                peer_rule_cases.setdefault(f"H-X_exact({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-X_exact({n}H)")
        if not Q_exact.empty:
            for c in Q_exact.nlargest(MAX_REL_CASES).index:
                peer_rule_cases.setdefault(f"H-Q_exact({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Q_exact({n}H)")

        Z_peak = meta[f"Z1_peak_{n}"].loc[peer_pool]
        X_peak = meta[f"X1_peak_{n}"].abs().loc[peer_pool]
        Q_peak = meta[f"Q1_peak_{n}"].loc[peer_pool]
        if not Z_peak.empty:
            top_cases = Z_peak[Z_peak > 0].nlargest(MAX_REL_CASES).index
            for c in top_cases:
                peer_rule_cases.setdefault(f"H-Z_peak({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Z_peak({n}H)")
        if not X_peak.empty:
            top_cases = X_peak[X_peak > 0].nlargest(MAX_REL_CASES).index
            for c in top_cases:
                peer_rule_cases.setdefault(f"H-X_peak({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-X_peak({n}H)")
        if not Q_peak.empty:
            top_cases = Q_peak[Q_peak > 0].nlargest(MAX_REL_CASES).index
            for c in top_cases:
                peer_rule_cases.setdefault(f"H-Q_peak({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Q_peak({n}H)")

    for n in range(2, MAX_HARMONIC + 1):
        Z0_exact = meta[f"Z0_exact_{n}"].loc[peer_pool]
        X0_exact = meta[f"X0_exact_{n}"].abs().loc[peer_pool]
        Q0_exact = meta[f"Q0_exact_{n}"].loc[peer_pool]
        if not Z0_exact.empty:
            for c in Z0_exact.nlargest(MAX_REL_CASES).index:
                peer_rule_cases.setdefault(f"H-Z0_exact({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Z0_exact({n}H)")
        if not X0_exact.empty:
            for c in X0_exact.nlargest(MAX_REL_CASES).index:
                peer_rule_cases.setdefault(f"H-X0_exact({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-X0_exact({n}H)")
        if not Q0_exact.empty:
            for c in Q0_exact.nlargest(MAX_REL_CASES).index:
                peer_rule_cases.setdefault(f"H-Q0_exact({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Q0_exact({n}H)")

        Z0_peak = meta[f"Z0_peak_{n}"].loc[peer_pool]
        X0_peak = meta[f"X0_peak_{n}"].abs().loc[peer_pool]
        Q0_peak = meta[f"Q0_peak_{n}"].loc[peer_pool]
        if not Z0_peak.empty:
            top_cases = Z0_peak[Z0_peak > 0].nlargest(MAX_REL_CASES).index
            for c in top_cases:
                peer_rule_cases.setdefault(f"H-Z0_peak({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Z0_peak({n}H)")
        if not X0_peak.empty:
            top_cases = X0_peak[X0_peak > 0].nlargest(MAX_REL_CASES).index
            for c in top_cases:
                peer_rule_cases.setdefault(f"H-X0_peak({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-X0_peak({n}H)")
        if not Q0_peak.empty:
            top_cases = Q0_peak[Q0_peak > 0].nlargest(MAX_REL_CASES).index
            for c in top_cases:
                peer_rule_cases.setdefault(f"H-Q0_peak({n}H)", []).append(c)
                peer_first_tag.setdefault(c, f"H-Q0_peak({n}H)")

    if peer_pool:
        for tag, series, smallest in [
            ("Q-R", meta.loc[peer_pool, "Q1_pk"], False),
            ("R-Min", meta.loc[peer_pool, "R1_min"], True),
            ("E-Zero", meta.loc[peer_pool, "f_pk"], True),
            ("ΣZ", meta.loc[peer_pool, "ΣZ1"], False),
            ("Q0-R", meta.loc[peer_pool, "Q0_pk"], False),
            ("R0-Min", meta.loc[peer_pool, "R0_min"], True),
            ("ΣZ0", meta.loc[peer_pool, "ΣZ0"], False),
        ]:
            cases_sorted = (
                series.nsmallest(MAX_REL_CASES)
                if smallest
                else series.nlargest(MAX_REL_CASES)
            ).index
            for c in cases_sorted:
                peer_rule_cases.setdefault(tag, []).append(c)
                peer_first_tag.setdefault(c, tag)

    total_selected = sum(len(v) for v in peer_rule_cases.values())
    print(f"   Initial peer count = {total_selected}")
    for tag in rel_order:
        for c in peer_rule_cases.get(tag, []):
            print(f"    {tag} → {c}")

    ladder = (
        meta["Z1_pk"] / Z_REF
        + meta[[f"Z1_peak_{n}" for n in range(2, MAX_HARMONIC + 1)]].max(axis=1) / Z_REF
        + meta["Q1_pk"]
        + ((0.8 * FUND - meta["f_pk"]).clip(lower=0) / (0.8 * FUND))
        + meta["X1_min"].abs() / X_REF
    )
    print("▶ 7. Top-up if needed …")
    while len(peer_first_tag) < MIN_REL_LIST:
        remaining = [c for c in ladder.index if c not in peer_first_tag]
        nxt = pd.Series(ladder.loc[remaining]).idxmax()
        peer_rule_cases.setdefault("TopUp", []).append(nxt)
        peer_first_tag.setdefault(nxt, "TopUp")
        print(f"    TopUp → {nxt} (score={ladder[nxt]:.2f})")
    print(f"   Final peer count = {len(peer_first_tag)}")

    return peer_rule_cases, peer_first_tag, rel_order


def plot_results(
    freqs,
    meta,
    R1,
    X1,
    R0,
    X0,
    Z1,
    Z0,
    peer_rule_cases,
    peer_first_tag,
    sel_abs,
):
    """Plot all figures and save them to disk."""
    print("▶ 9. Plotting selected cases …")
    all_cases = list(peer_first_tag.keys())
    case_expl = {c: LABELS.get(peer_first_tag[c], "") for c in all_cases}

    pos_cases = [
        c
        for c in all_cases
        if not peer_first_tag[c].startswith(
            ("H-X0", "H-Z0", "H-Q0", "Q0-R", "R0-Min", "E-Zero0", "ΣZ0")
        )
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
        bottom_margin = legend_height + 0.05

        top = 0.95
        default_bottom = 0.05
        base_frac = top - default_bottom
        new_frac = top - bottom_margin
        if new_frac <= 0:
            warnings.warn("Legend too tall – results may be clipped")
            new_frac = 0.1
        if new_frac < base_frac:
            scale = base_frac / new_frac
            fig.set_figheight(fig.get_figheight() * scale)
        fig.subplots_adjust(top=top, bottom=bottom_margin, hspace=0.3)
        y_anchor = bottom_margin / 2
        fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, fontsize="small", bbox_to_anchor=(0.5, y_anchor))

    def line_kwargs(case):
        tag = peer_first_tag.get(case, sel_abs.get(case, "")).lower()
        if "peak" in tag:
            return {"linestyle": "--", "linewidth": 1.0}
        return {"linestyle": "-", "linewidth": 1.5}

    def plot_sequence(axs, metrics, cases, label_func):
        for ax, (_, df, ylabel) in zip(axs, metrics):
            for n in harmonics:
                ax.axvline(n, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axvline(n - bin_halfwidth, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.axvline(n + bin_halfwidth, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            for c in cases:
                ax.plot(harmonic, df[c], label=label_func(c), **line_kwargs(c))
            ax.set_ylabel(ylabel)
            ax.set_xticks(harmonics)
            ax.set_xticklabels([f"{n}H" for n in harmonics])
        axs[-1].set_xlabel("Harmonic Number (n)")
        return axs[0].get_legend_handles_labels()

    print("▶ 9a. Positive-sequence plots …")
    metrics_pos = [
        ("X1", X1, "X1 (Ω)"),
        ("R1", R1, "R1 (Ω)"),
        ("X1/R1", X1.div(R1.replace(0, np.nan)), "X1/R1"),
        ("Z1", Z1, "Z1 (Ω)"),
    ]
    fig1, axs1 = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    h1, labs1 = plot_sequence(axs1, metrics_pos, pos_cases, lambda c: c)
    reserve_and_legend(fig1, axs1, h1, labs1, peer_first_tag, case_expl)
    fig1.savefig(FIG_POS, dpi=300)
    print(f"   ↳ saved {FIG_POS.name}")

    print("▶ 9b. Zero-sequence plots …")
    metrics_zero = [
        ("X0", X0, "X0 (Ω)"),
        ("R0", R0, "R0 (Ω)"),
        ("X0/R0", X0.div(R0.replace(0, np.nan)), "X0/R0"),
        ("Z0", Z0, "Z0 (Ω)"),
    ]
    fig2, axs2 = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
    h2, labs2 = plot_sequence(axs2, metrics_zero, zero_cases, lambda c: c)
    reserve_and_legend(fig2, axs2, h2, labs2, peer_first_tag, case_expl)
    fig2.savefig(FIG_ZERO, dpi=300)
    print(f"   ↳ saved {FIG_ZERO.name}")

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
            h3p, lab3p = plot_sequence(axs3[:, 0], metrics_pos, pos_abs, lambda c: c)
            handles.extend(h3p)
            labels.extend(lab3p)
        else:
            for ax in axs3[:, 0]:
                ax.axis("off")
        if zero_abs:
            h3z, lab3z = plot_sequence(axs3[:, 1], metrics_zero, zero_abs, lambda c: c)
            handles.extend(h3z)
            labels.extend(lab3z)
        else:
            for ax in axs3[:, 1]:
                ax.axis("off")
        reserve_and_legend(fig3, axs3.ravel(), handles, labels, abs_first_tag, abs_case_expl)
        fig3.savefig(FIG_ABS, dpi=300)
        print(f"   ↳ saved {FIG_ABS.name}")

    plt.show()


def write_report(meta, peer_rule_cases, rel_order):
    """Write a detailed plain-text report."""
    try:
        from collections import defaultdict

        lines = [
            "Frequency Sweep – Worst-Case Harmonic Selection Report",
            "=" * 60,
            f"Fundamental Frequency: {FUND} Hz",
            f"Harmonic Range: 2H to {MAX_HARMONIC}H",
            f"Bin Width: ±{HARMONIC_BAND_HZ} Hz",
            f"Include Negative Peaks: {'Yes' if INCLUDE_NEGATIVE_PEAKS else 'No'}",
            f"Absolute Rule Enabled: {'Yes' if ENABLE_ABSOLUTE else 'No'}",
            "",
        ]

        tag_map = defaultdict(list)
        for tag, cases_list in peer_rule_cases.items():
            for case in cases_list:
                tag_map[case].append(tag)

        for tag in rel_order:
            lines.append(tag)
            lines.append("-" * len(tag))
            for case in peer_rule_cases.get(tag, []):
                use_Z0 = tag.startswith(("H-X0", "H-Z0", "H-Q0", "Q0-R", "R0-Min", "ΣZ0"))
                fpk = meta.at[case, "f_pk"]
                zpk = meta.at[case, "Z0_pk" if use_Z0 else "Z1_pk"]
                qpk = meta.at[case, "Q0_pk" if use_Z0 else "Q1_pk"]
                lines.append(f"{case:<40} | {fpk:6.1f} Hz | Z={zpk:7.1f} Ω | Q={qpk:4.2f}")
            lines.append("")

        dup_cases = {c: t for c, t in tag_map.items() if len(t) > 1}
        if dup_cases:
            lines.append("Duplicate Cases Triggered by Multiple Rules")
            lines.append("-" * 44)
            for case, tags in dup_cases.items():
                fpk = meta.at[case, "f_pk"]
                zpk = meta.at[case, "Z1_pk"]
                qpk = meta.at[case, "Q1_pk"]
                tag_str = ", ".join(tags)
                lines.append(f"{case:<40} | {tag_str:<24} | {fpk:6.1f} Hz | Z={zpk:7.1f} Ω | Q={qpk:4.2f}")
            lines.append("")

        with open("worst_case_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("   ↳ Plain-text report saved as: worst_case_report.txt")
    except Exception as e:
        print(f"   [Text Report Error] {e}")


def main():
    freqs, cases, R1, X1, R0, X0 = load_data()
    meta, Z1, Z0, Q1, Q0 = compute_metrics(freqs, cases, R1, X1, R0, X0)
    sel_abs = apply_absolute_rules(meta, cases)
    peer_rule_cases, peer_first_tag, rel_order = select_relative_cases(meta, cases, sel_abs)

    def emit(path, mapping, title, order):
        print(f"\n{title}")
        print("Case".ljust(38), "Tag".ljust(14), "f_pk   Z1_pk   Q1_pk   Explanation")
        print("─" * 130)
        lines = []
        for tag in order:
            for case in mapping.get(tag, []):
                explanation = LABELS.get(tag, "")
                line = (
                    f"{case} | {tag} | {explanation} | "
                    f"f_pk={meta.at[case,'f_pk']:.1f}Hz | "
                    f"Z1_pk={meta.at[case,'Z1_pk']:.1f}Ω | "
                    f"Q1_pk={meta.at[case,'Q1_pk']:.2f}"
                )
                lines.append(line)
                print(
                    f"{case:<38} {tag:<14} {meta.at[case,'f_pk']:>5.1f}  "
                    f"{meta.at[case,'Z1_pk']:>7.1f}  {meta.at[case,'Q1_pk']:>6.2f}   {explanation}"
                )
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"   ↳ saved {len(lines)} rows to {path.name}")

    abs_map = {}
    for case, tag in sel_abs.items():
        abs_map.setdefault(tag, []).append(case)
    emit(ABS_OUT, abs_map, "ABSOLUTE RULE LIST", ABS_ORDER)
    emit(REL_OUT, peer_rule_cases, "RELATIVE TOP-5 LIST", rel_order)

    plot_results(freqs, meta, R1, X1, R0, X0, Z1, Z0, peer_rule_cases, peer_first_tag, sel_abs)
    write_report(meta, peer_rule_cases, rel_order)


if __name__ == "__main__":
    main()

