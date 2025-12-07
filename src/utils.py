import pandas as pd
import numpy as np
import ast

def load_data(path):
    return pd.read_csv(path)

def parse_list_col(x):

    if pd.isna(x) or x == "":
        return []
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(item).strip() for item in val]
    except:
        pass
    
    return [item.strip() for item in x.split(',')]

def clean_dataset(df):

    cols_to_drop = [
        'poster_path', 'backdrop_path', 'homepage', 
        'imdb_id', 'original_title', 'overview', 'tagline'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df = df.dropna(subset=['budget']) 
    
    if 'vote_count' in df.columns:
        df = df[df['vote_count'] >= 50] 

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
    
    df = df[df['budget'] > 10000] 
    df = df[df['revenue'] > 0]

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

def feature_engineering_pipeline(df):
    # Parse lists
    list_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
    for col in list_cols:
        if col in df.columns:
            df[f'{col}_list'] = df[col].apply(parse_list_col)

    if 'production_companies' in df.columns:
        df['companies_count'] = df['production_companies'].apply(lambda x: len(str(x).split(',')))

    # Dates
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day_of_week'] = df['release_date'].dt.dayofweek # 0=Mon, 6=Sun
        
        # Season
        df['season'] = df['release_month'].apply(month_to_season)
        
        # Weekend (1 if Sat or Sun)
        df['is_weekend'] = df['release_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Log transform 
    if 'budget' in df.columns:
        df['log_budget'] = np.log1p(df['budget'])
    if 'revenue' in df.columns:
        df['log_revenue'] = np.log1p(df['revenue'])
    if 'popularity' in df.columns:
        df['log_popularity'] = np.log1p(df['popularity'])

    return df

def prepare_features_for_ml(df):
    df_ml = df.copy()
    
    # One-Hot Genres
    all_genres = set([g for sublist in df_ml['genres_list'] for g in sublist])
    for genre in all_genres:
        col_name = f"genre_{genre.replace(' ', '')}"
        df_ml[col_name] = df_ml['genres_list'].apply(lambda x: 1 if genre in x else 0)
        
    # One-Hot Studios
    comp_col = 'production_companies_list'
    
    if comp_col in df_ml.columns:
        all_companies = [c for sublist in df_ml[comp_col] for c in sublist]
        
        top_10_companies = pd.Series(all_companies).value_counts().head(10).index.tolist()
        
        df_ml['has_top_studio'] = df_ml[comp_col].apply(lambda x: 1 if any(c in top_10_companies for c in x) else 0)
        
        for company in top_10_companies:
            col_name = f"studio_{company.replace(' ', '').replace('.', '')}"
            df_ml[col_name] = df_ml[comp_col].apply(lambda x: 1 if company in x else 0)
    else:
        print(f"Not found")
        
    # One-Hot Seasons 
    if 'season' in df_ml.columns:
        seasons = pd.get_dummies(df_ml['season'], prefix='season')
        df_ml = pd.concat([df_ml, seasons], axis=1)
    
    return df_ml