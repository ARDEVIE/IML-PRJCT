import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def parse_comma_separated_col(x):
    if pd.isna(x) or x == "":
        return []
    return [item.strip() for item in x.split(',')]

def clean_dataset(df):
    
    cols_to_drop = [
    'poster_path', 'backdrop_path', 'homepage', 
    'imdb_id', 'original_title', 'overview', 'tagline',]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df = df.dropna(subset=['budget']) 
    df = df[df['vote_count'] > 100] 


    
    if 'status' in df.columns:
        df = df[df['status'] == 'Released']
    
    df = df.dropna(subset=['vote_average'])

    
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'])

    
    return df

def process_budget_revenue(df):
    cols = ['budget', 'revenue']
    
    for col in cols:
        if col in df.columns:
        
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df[col] = df[col].replace(0, np.nan)

    df = df.dropna(subset=['budget', 'revenue'])

    df = df[df['budget'] > 0]
    df = df[df['revenue'] > 0]

    return df


def feature_engineering_pipeline(df):
    list_cols = ['genres', 'production_companies', 'production_countries', 
                 'spoken_languages']
    
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_comma_separated_col)

    if 'production_companies' in df.columns:
        df['companies_count'] = df['production_companies'].apply(len)


    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day_of_week'] = df['release_date'].dt.dayofweek
    
    return df


def month_to_season(m):
    if m in [12, 1, 2]:
        return 'winter'
    elif m in [3, 4, 5]:
        return 'spring'
    elif m in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'