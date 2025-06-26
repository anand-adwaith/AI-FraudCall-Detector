import pandas as pd
import re
import unicodedata


def drop_empty_rows(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Remove rows where the text column is null or empty.
    """
    # Drop NaN
    df = df.dropna(subset=[text_col])
    # Drop empty strings
    df = df[df[text_col].str.strip().astype(bool)]
    return df

def trim_whitespace_and_quotes(text: str) -> str:
    """
    Strip leading/trailing whitespace and surrounding quotes.
    """
    # Remove surrounding double or single quotes
    text = re.sub(r'^[\'"]+|[\'"]+$', '', text)
    # Trim whitespace
    return text.strip()

def unicode_normalize(text: str) -> str:
    """
    Normalize unicode characters to their canonical form (NFC).
    """
    return unicodedata.normalize('NFC', text)

def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    """
    return text.lower()

def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces, tabs, or newlines into a single space.
    """
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_dataframe(csv_path: str, text_col: str = 'Call_Transcript') -> pd.DataFrame:
    """
    Load CSV, apply preprocessing steps, and return cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)
    # 1. Drop empty or null text rows
    df = drop_empty_rows(df, text_col)  # :contentReference[oaicite:0]{index=0}

    # Apply text-level cleaning
    def clean_row(text: str) -> str:
        text = trim_whitespace_and_quotes(text)
        text = unicode_normalize(text)     # :contentReference[oaicite:1]{index=1}
        text = lowercase_text(text)        # :contentReference[oaicite:2]{index=2}
        text = normalize_whitespace(text)  # :contentReference[oaicite:3]{index=3}
        return text

    # 2–5. Vectorize cleaning over the text column
    df[text_col] = df[text_col].apply(clean_row)

    return df