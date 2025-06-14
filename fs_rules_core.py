import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
import logging

logger = logging.getLogger(__name__)


def load_data(book: Path):
    """Load the Excel workbook and return frequency index and dataframes."""
    logger.info(f"\u25B6 1. Loading {book.name} \u2026")
    try:
        R1 = pd.read_excel(book, "R1", index_col=0)
        X1 = pd.read_excel(book, "X1", index_col=0)
        R0 = pd.read_excel(book, "R0", index_col=0)
        X0 = pd.read_excel(book, "X0", index_col=0)
    except Exception as e:
        sys.exit(f"\u274C Cannot open {book}: {e}")

    freqs = R1.index.to_numpy()
    cases = R1.columns
    return freqs, cases, R1, X1, R0, X0


def compute_metrics(
    freqs,
    cases,
    R1,
    X1,
    R0,
    X0,
    FUND: float,
    MAX_HARMONIC: int,
    HARMONIC_BAND_HZ: float,
    INCLUDE_NEGATIVE_PEAKS: bool,
    PEAK_PROMINENCE,
):
    """Compute base metrics and return them along with a meta dataframe."""
    logger.info("\u25B6 2. Computing base metrics (Z1, Z0, Q1, Q0, etc.) …")
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
        ("0", Z0, X0, R0),  # zero-sequence
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
    trapz = getattr(np, "trapezoid", np.trapz)
    meta["ΣZ1"] = pd.Series(trapz(Z1.to_numpy(), x=freqs, axis=0), index=cases)
    meta["ΣZ0"] = pd.Series(trapz(Z0.to_numpy(), x=freqs, axis=0), index=cases)

    return meta, Z1, Z0, Q1, Q0


def reserve_and_legend(fig, axs, handles, raw_labels, tag_map, expl_map, *, max_height=0.5):
    """Adjust bottom margin and place a figure legend."""
    labels = [f"{c}: {tag_map[c]} – {expl_map[c]}" for c in raw_labels]
    n = len(labels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    line_height = 0.03
    legend_height = nrows * line_height
    bottom_margin = legend_height + 0.05
    if max_height is not None:
        bottom_margin = min(bottom_margin, max_height)
    top = 0.95
    default_bottom = 0.05
    base_frac = top - default_bottom
    new_frac = top - bottom_margin
    clipped = bottom_margin < legend_height + 0.05
    if new_frac <= 0:
        warnings.warn("Legend too tall – results may be clipped")
        new_frac = 0.1
        clipped = True
    if new_frac < base_frac:
        scale = base_frac / new_frac
        fig.set_figheight(fig.get_figheight() * scale)
    fig.subplots_adjust(top=top, bottom=bottom_margin, hspace=0.3)
    if clipped:
        warnings.warn("Legend truncated; labels may be clipped")
    y_anchor = bottom_margin / 2
    fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, fontsize="small", bbox_to_anchor=(0.5, y_anchor))


def plot_sequence(axs, metrics, cases, label_func, harmonic, harmonics, bin_halfwidth, line_kwargs=None):
    """Plot each metric for the given cases."""
    for ax, (_, df, ylabel) in zip(axs, metrics):
        for n in harmonics:
            ax.axvline(n, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(n - bin_halfwidth, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.axvline(n + bin_halfwidth, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        for c in cases:
            kw = line_kwargs(c) if line_kwargs else {}
            ax.plot(harmonic, df[c], label=label_func(c), **kw)
        ax.set_ylabel(ylabel)
        ax.set_xticks(harmonics)
        ax.set_xticklabels([f"{n}H" for n in harmonics])
    axs[-1].set_xlabel("Harmonic Number (n)")
    return axs[0].get_legend_handles_labels()
