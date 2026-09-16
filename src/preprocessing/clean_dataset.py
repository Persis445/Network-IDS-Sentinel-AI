from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "MachineLearningCVE"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Constant features identified during dataset exploration.
CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


# Fix malformed Web Attack labels found in CICIDS2017.
LABEL_MAPPING = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """Remove whitespace and BOM characters from column names."""

    df.columns = (
        df.columns
        .str.strip()
        .str.lstrip("\ufeff")
    )

    return df


def clean_labels(df):
    """Normalize the CICIDS2017 class labels."""

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .replace(LABEL_MAPPING)
    )

    return df


def clean_numeric_values(df):
    """
    Replace infinite values with NaN and remove
    rows containing missing numerical values.
    """

    numeric_columns = df.select_dtypes(include=np.number).columns

    # Replace positive and negative infinity with NaN.
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    rows_before = len(df)

    # Remove rows containing missing numerical values.
    df = df.dropna(subset=numeric_columns)

    rows_removed = rows_before - len(df)

    return df, rows_removed


def remove_constant_features(df):
    """Remove features identified as globally constant."""

    columns_to_remove = [
        column
        for column in CONSTANT_FEATURES
        if column in df.columns
    ]

    df = df.drop(columns=columns_to_remove)

    return df


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print("=" * 65)
    print("SentinelAI - CICIDS2017 Dataset Preprocessing")
    print("=" * 65)

    # Create processed directory if it doesn't exist.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find raw CSV files.
    csv_files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{RAW_DATA_DIR}"
        )

    print(f"\nFound {len(csv_files)} CSV files.")

    # Store cleaned files temporarily.
    cleaned_files = []

    total_original_rows = 0
    total_invalid_rows = 0
    total_file_duplicates = 0

    # ========================================================
    # PROCESS FILES ONE AT A TIME
    # ========================================================

    for file in csv_files:

        print("\n" + "-" * 65)
        print(f"Processing: {file.name}")
        print("-" * 65)

        # Load only ONE file into memory.
        df = pd.read_csv(file)

        original_rows = len(df)
        total_original_rows += original_rows

        print(f"Original rows: {original_rows:,}")

        # ----------------------------------------------------
        # 1. Clean column names
        # ----------------------------------------------------

        df = clean_column_names(df)

        # ----------------------------------------------------
        # 2. Clean labels
        # ----------------------------------------------------

        df = clean_labels(df)

        # ----------------------------------------------------
        # 3. Handle missing/infinite values
        # ----------------------------------------------------

        df, invalid_rows = clean_numeric_values(df)

        total_invalid_rows += invalid_rows

        print(
            f"Invalid rows removed: {invalid_rows:,}"
        )

        # ----------------------------------------------------
        # 4. Remove constant features
        # ----------------------------------------------------

        df = remove_constant_features(df)

        # ----------------------------------------------------
        # 5. Remove duplicate rows within this file
        # ----------------------------------------------------

        rows_before_duplicates = len(df)

        df = df.drop_duplicates()

        duplicates_removed = (
            rows_before_duplicates - len(df)
        )

        total_file_duplicates += duplicates_removed

        print(
            f"Duplicate rows removed: {duplicates_removed:,}"
        )

        print(
            f"Rows remaining: {len(df):,}"
        )

        print(
            f"Columns remaining: {len(df.columns)}"
        )

        # ----------------------------------------------------
        # Save this cleaned file
        # ----------------------------------------------------

        output_file = (
            PROCESSED_DATA_DIR
            / f"cleaned_{file.name}"
        )

        df.to_csv(
            output_file,
            index=False
        )

        cleaned_files.append(output_file)

        print(
            f"Saved: {output_file.name}"
        )

        # Release memory before processing next file.
        del df

    # ========================================================
    # COMBINE CLEANED FILES
    # ========================================================

    print("\n" + "=" * 65)
    print("Combining cleaned files...")
    print("=" * 65)

    # Load cleaned files one at a time.
    cleaned_datasets = []

    for file in cleaned_files:

        print(f"Loading: {file.name}")

        df = pd.read_csv(file)

        cleaned_datasets.append(df)

    combined_df = pd.concat(
        cleaned_datasets,
        ignore_index=True
    )

    # Release individual DataFrames.
    del cleaned_datasets

    # ========================================================
    # GLOBAL DUPLICATE REMOVAL
    # ========================================================

    rows_before_global_duplicates = len(combined_df)

    combined_df = combined_df.drop_duplicates()

    global_duplicates_removed = (
        rows_before_global_duplicates
        - len(combined_df)
    )

    # ========================================================
    # SAVE FINAL DATASET
    # ========================================================

    final_output = (
        PROCESSED_DATA_DIR
        / "cicids2017_cleaned.csv"
    )

    combined_df.to_csv(
        final_output,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE")
    print("=" * 65)

    print(
        f"Original rows:             "
        f"{total_original_rows:,}"
    )

    print(
        f"Invalid rows removed:      "
        f"{total_invalid_rows:,}"
    )

    print(
        f"File-level duplicates:     "
        f"{total_file_duplicates:,}"
    )

    print(
        f"Global duplicates removed: "
        f"{global_duplicates_removed:,}"
    )

    print(
        f"Final rows:                "
        f"{len(combined_df):,}"
    )

    print(
        f"Final columns:             "
        f"{len(combined_df.columns)}"
    )

    print(
        f"\nFinal dataset saved to:\n"
        f"{final_output}"
    )

    # ========================================================
    # FINAL CLASS DISTRIBUTION
    # ========================================================

    print("\nFinal class distribution:")

    print(
        combined_df["Label"]
        .value_counts()
    )


if __name__ == "__main__":
    main()