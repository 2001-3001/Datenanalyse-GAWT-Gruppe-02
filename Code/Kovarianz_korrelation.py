import pandas as pd
import numpy as np

# Datensatz einlesen (automatisch ; oder , anpassen falls nötig)
df = pd.read_csv("data-1.csv", sep=";")

# Spalten auswählen
df = df.iloc[:, :2]
df.columns = ["Datum", "Ausfuhr"]

# Ausfuhr numerisch machen
df["Ausfuhr"] = (
    df["Ausfuhr"].astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)
df["Ausfuhr"] = pd.to_numeric(df["Ausfuhr"], errors="coerce")

# Zeitindex erzeugen
df["t"] = np.arange(1, len(df) + 1)

# Kovarianz
kovarianz = df["t"].cov(df["Ausfuhr"])
print("Kovarianz (Zeit, Ausfuhr):", kovarianz)

# Korrelationskoeffizient (Pearson)
korrelation = df["t"].corr(df["Ausfuhr"])
print("Korrelationskoeffizient (Pearson):", korrelation)
