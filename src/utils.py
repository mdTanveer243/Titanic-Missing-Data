from pathlib import Path
import pandas as pd

def load_csv(path_str: str):
    p = Path(path_str).expanduser().resolve()
    df = pd.read_csv(p)
    return df, p

def save_csv(df: pd.DataFrame, original_path, out_dir='outputs', suffix='_cleaned'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(original_path).stem + f"{suffix}.csv")
    df.to_csv(out_path, index=False)
    return out_path

def missing_counts(df: pd.DataFrame):
    return df.isna().sum().sort_values(ascending=False)
