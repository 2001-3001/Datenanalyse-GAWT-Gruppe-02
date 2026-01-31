"""
Projekt: Wahrscheinlichkeitstheorie / Deskriptive Statistik (Lastenheft)

Dieser Code:
- liest eine CSV (automatisch ',' oder ';')
- erwartet: eine Datums-Spalte + eine numerische Spalte (z.B. Ausfuhrwert)
- erzeugt Urliste (R1.4) und Rangliste (R1.5)
- berechnet Kennzahlen (R1.7–R1.16)
- erstellt Histogramm + Boxplot (R1.10/R1.12)
- berechnet Spearman zwischen Zeitindex t=1..n und Wert (R1.21)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def read_csv_auto_sep(path: str) -> pd.DataFrame:
    """Liest CSV mit automatischer Trennzeichenerkennung (',' oder ';')."""
    try:
        return pd.read_csv(path, sep=",")
    except Exception:
        return pd.read_csv(path, sep=";")


def prepare_two_columns(df: pd.DataFrame,
                        date_col: str | None = None,
                        value_col: str | None = None) -> pd.DataFrame:
    """
    Gibt ein DataFrame mit genau 2 Spalten zurück:
    - 'Datum' (String)
    - 'Wert' (float)
    Wenn keine Spaltennamen angegeben werden, nimmt die Funktion die ersten beiden Spalten.
    """
    if date_col is None or value_col is None:
        if df.shape[1] < 2:
            raise ValueError("CSV muss mindestens 2 Spalten haben (Datum + Wert).")
        date_col = df.columns[0]
        value_col = df.columns[1]

    out = df[[date_col, value_col]].copy()
    out.columns = ["Datum", "Wert"]

    # Wert numerisch machen (Komma/Leerzeichen robust behandeln)
    out["Wert"] = (
        out["Wert"]
        .astype(str)
        .str.replace(".", "", regex=False)      # Tausenderpunkt entfernen (falls vorhanden)
        .str.replace(",", ".", regex=False)     # Dezimalkomma -> Punkt
    )
    out["Wert"] = pd.to_numeric(out["Wert"], errors="coerce")

    out = out.dropna(subset=["Wert"]).reset_index(drop=True)
    return out


def compute_descriptive_stats(x: pd.Series) -> dict:
    """Berechnet die wichtigsten Kennzahlen für das Lastenheft."""
    n = int(x.shape[0])

    mean = float(x.mean())
    median = float(x.median())

    # Modus: bei stetigen Daten oft nicht eindeutig -> nur dann ausgeben, wenn wirklich doppelte Werte existieren
    mode_series = x.mode(dropna=True)
    mode_list = mode_series.tolist()
    if x.value_counts().max() == 1:
        mode_value = None  # kein eindeutiger Modus
    else:
        mode_value = mode_list  # kann mehrere Modi geben

    sample_variance = float(x.var(ddof=1))
    std = float(x.std(ddof=1))

    q1 = float(x.quantile(0.25))
    q2 = float(x.quantile(0.50))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1  # Quartilsabstand R_Q0.5

    deciles = {f"D{i}": float(x.quantile(i / 10)) for i in range(1, 10)}

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "mode": mode_value,
        "sample_variance": sample_variance,
        "std": std,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "IQR_R_Q0.5": iqr,
        "deciles": deciles,
    }


def export_r14_urliste(df2: pd.DataFrame, out_csv: str) -> None:
    """R1.4: Urliste (Datum + Wert) in Originalreihenfolge."""
    df2.to_csv(out_csv, index=False)


def export_r15_rangliste(df2: pd.DataFrame, out_csv: str) -> None:
    """R1.5: Rangliste (nur Werte) aufsteigend sortiert."""
    df2[["Wert"]].sort_values(by="Wert", ascending=True).to_csv(out_csv, index=False)


def plot_histogram(x: pd.Series, out_png: str | None = None) -> None:
    """R1.12: Histogramm."""
    plt.figure()
    plt.hist(x.dropna().values, bins="auto")
    plt.xlabel("Wert")
    plt.ylabel("Häufigkeit")
    plt.title("Histogramm der Werte")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_boxplot(x: pd.Series, out_png: str | None = None) -> None:
    """R1.10: Box-Whisker-Plot."""
    plt.figure()
    plt.boxplot(x.dropna().values, vert=True)
    plt.ylabel("Wert")
    plt.title("Boxplot der Werte")
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def spearman_time_vs_value(df2: pd.DataFrame) -> tuple[float, float]:
    """
    R1.21: Spearman zwischen Zeitindex t=1..n und Wert.
    Rückgabe: (rho, p_value)
    """
    x = df2["Wert"].astype(float)
    t = np.arange(1, len(x) + 1)
    rho, p = spearmanr(t, x)
    return float(rho), float(p)


def main():
    # ======== ANPASSEN: Dateiname + (optional) Spaltennamen ========
    data_path = "data-1.csv"
    # Falls deine Spalten NICHT die ersten zwei sind, setze:
    # date_col = "Monat und Jahr"
    # value_col = "Ausfuhr: Wert in Tsd. EUR"
    date_col = None
    value_col = None

    df = read_csv_auto_sep(data_path)
    df2 = prepare_two_columns(df, date_col=date_col, value_col=value_col)

    # R1.4 / R1.5 Dateien
    export_r14_urliste(df2, "R1_4_Urliste_Datensatz_1.csv")
    export_r15_rangliste(df2, "R1_5_Rangliste_Ausfuhrwert.csv")

    # Kennzahlen
    stats = compute_descriptive_stats(df2["Wert"])

    print("=== Kennzahlen (Datensatz 1) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Plots (für Bericht/Anhang)
    plot_histogram(df2["Wert"], out_png="R1_12_Histogramm.png")
    plot_boxplot(df2["Wert"], out_png="R1_10_Boxplot.png")

    # Spearman
    rho, p = spearman_time_vs_value(df2)
    print(f"\nSpearman (Zeitindex vs. Wert): rho={rho:.4f}, p={p:.4g}")


if __name__ == "__main__":
    main()

#   Ende code.py