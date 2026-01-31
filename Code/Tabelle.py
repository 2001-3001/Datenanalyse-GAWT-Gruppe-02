import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plots(filename, box=False, scatter=False):
    df = pd.read_csv(filename, sep=";")
    df = df.iloc[:, :2]
    df.columns = ["Datum", "Ausfuhr"]

    # Ausfuhr numerisch
    df["Ausfuhr"] = (
        df["Ausfuhr"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Ausfuhr"] = pd.to_numeric(df["Ausfuhr"], errors="coerce")

    # Zeitindex
    df["t"] = np.arange(1, len(df) + 1)

    if box:
        plt.boxplot(df["Ausfuhr"].dropna())
        plt.title("Boxplot der Ausfuhrwerte")
        plt.ylabel("Ausfuhr (Tsd. EUR)")
        plt.show()

    if scatter:
        plt.scatter(df["t"], df["Ausfuhr"], s=8)
        plt.xlabel("Zeitindex (Monate)")
        plt.ylabel("Ausfuhr (Tsd. EUR)")
        plt.title("Scatterplot: Zeitindex vs. Ausfuhr")
        plt.show()

# Aufruf
plots("data-1.csv", box=True, scatter=True)
