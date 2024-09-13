from pathlib import Path
import pandas as pd

# Assuming project_root is already defined
project_root = Path("..").resolve()

def read_csv(relative_path, **kwargs):
    """
    Utility function to read a CSV file with the project root prepended.
    Args:
        relative_path (str or Path): The relative path from the project root.
        **kwargs: Additional arguments to pass to pd.read_csv.
    Returns:
        DataFrame: Loaded DataFrame from the CSV file.
    """
    # Construct the full path
    full_path = project_root / relative_path
    return pd.read_csv(full_path, **kwargs)

def read_pickle(relative_path):
    full_path = project_root / relative_path
    return pd.read_pickle(full_path)
