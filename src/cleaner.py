import pandas as pd

def fill_missing(df: pd.DataFrame, method: str, fill_categorical: bool = True) -> pd.DataFrame:
    method = method.lower().strip()
    num_cols = df.select_dtypes(include=['number']).columns

    if method == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif method == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif method == 'mode':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mode().iloc[0])
    else:
        raise ValueError("Method must be one of: mean, median, mode")

    # To fully clean the dataset for later analysis,
    # fill categorical columns with their mode.
    if fill_categorical:
        cat_cols = df.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    return df