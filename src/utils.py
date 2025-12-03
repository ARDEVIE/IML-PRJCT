import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def parse_comma_separated_col(x):
    if pd.isna(x) or x == "":
        return []
    return [item.strip() for item in x.split(',')]

def clean_dataset(df):
    cols_to_drop = ['poster_path','imdb_id','original_title','overview','tagline','backdrop_path']    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    if 'status' in df.columns:
        df = df[df['status'] == 'Released']
    
    df = df.dropna(subset=['imdb_rating'])
    df = df[(df['vote_count'] > 500) | (df['imdb_votes'] > 1000)]
    
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'])
        
    return df

def process_budget_revenue(df):
    cols = ['budget', 'revenue']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(0.0, np.nan)
    return df

def feature_engineering_pipeline(df):

    list_cols = ['genres', 'production_companies', 'production_countries', 
                 'spoken_languages', 'cast', 'writers']
    
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_comma_separated_col)

    if 'cast' in df.columns:
        df['cast_size'] = df['cast'].apply(len)
        df['top_cast'] = df['cast'].apply(lambda x: x[:3])

    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
    
    return df