import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with missing critical info
    df.dropna(subset=['Business Title', 'Work Location', 'Agency', 'Salary Range From', 'Salary Range To', 'Job Category'], inplace=True)

    # Convert salary to float and create Average Salary
    df['Salary Range From'] = df['Salary Range From'].astype(float)
    df['Salary Range To'] = df['Salary Range To'].astype(float)
    df['Average Salary'] = (df['Salary Range From'] + df['Salary Range To']) / 2

    # Fill NaN for strings and clean text
    df['Preferred Skills'] = df['Preferred Skills'].fillna("").astype(str)
    df['Job Description'] = df['Job Description'].fillna("").astype(str)
    df['Work Location'] = df['Work Location'].str.strip().str.title()
    df['Business Title'] = df['Business Title'].str.strip().str.title()
    df['Agency'] = df['Agency'].str.strip().str.title()
    df['Job Category'] = df['Job Category'].str.strip().str.title()

    # If Years of Experience exists and is numeric, convert it, else create dummy zero
    if 'Years of Experience' in df.columns:
        df['Years of Experience'] = pd.to_numeric(df['Years of Experience'], errors='coerce').fillna(0).astype(int)
    else:
        df['Years of Experience'] = 0

    return df
