import pandas as pd

def load_data(filepath):
    """Load and clean the dataset."""
    df = pd.read_csv(filepath)

    # Drop irrelevant columns
    if 'Serial No.' in df.columns:
        df.drop('Serial No.', axis=1, inplace=True)

    return df

def check_missing_duplicates(df):
    """Check for missing values and duplicates."""
    print("Missing values:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())

def handle_outliers(df):
    """Optionally cap/floor outliers or return cleaned df as-is."""
    return df
