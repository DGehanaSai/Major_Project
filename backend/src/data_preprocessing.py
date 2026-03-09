# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# # Assuming the file names in config.py are correct
# from .config import YIELD_CSV, WEATHER_CSV, SOIL_CSV, NDVI_CSV

# def load_raw_data():
#     yield_df = pd.read_csv(YIELD_CSV)
#     weather_df = pd.read_csv(WEATHER_CSV)
#     soil_df = pd.read_csv(SOIL_CSV)
#     ndvi_df = pd.read_csv(NDVI_CSV)
#     return yield_df, weather_df, soil_df, ndvi_df

# def merge_data():
#     yield_df, weather_df, soil_df, ndvi_df = load_raw_data()
    
#     # 1. Standardize ID column name: 'plot_id' -> 'field_id'
#     # This resolves the KeyError 'field_id'
#     yield_df.rename(columns={'plot_id': 'field_id', 'crop_type': 'crop'}, inplace=True)
#     soil_df.rename(columns={'plot_id': 'field_id'}, inplace=True)
#     ndvi_df.rename(columns={'plot_id': 'field_id'}, inplace=True)
    
#     # 2. Rename columns in yield_df and convert units (kg to tons)
#     yield_df.rename(columns={'harvest_yield_kg': 'production_tons'}, inplace=True)
#     yield_df['production_tons'] = yield_df['production_tons'] / 1000.0 # kg -> tons
    
#     # Placeholder for 'area_ha' to enable 'yield_t_per_ha' calculation
#     # Assuming 1 hectare, since area data is missing from snippets
#     yield_df['area_ha'] = 1.0

#     # 3. Feature renaming for weather, soil, and ndvi before merging
#     weather_df.rename(columns={
#         'cumulative_rainfall_mm': 'avg_rain_mm', 
#         'days_above_30c': 'heatwave_days'
#     }, inplace=True)
    
#     soil_df.rename(columns={
#         'soil_type': 'soil_texture',
#         'ph_level': 'soil_ph',
#         'organic_carbon_percent': 'soil_oc',
#         'nitrogen_ppm': 'soil_n',
#         'phosphorus_ppm': 'soil_p',
#         'potassium_ppm': 'soil_k',
#     }, inplace=True)
    
#     ndvi_df.rename(columns={
#         'peak_ndvi': 'ndvi_max',
#         'avg_ndvi_season': 'ndvi_mean',
#         'ndvi_variability': 'ndvi_std',
#     }, inplace=True)
    
#     # 4. Merge data (Merging only on 'field_id' as 'season'/'year' are missing in weather/ndvi)
#     df = (
#         yield_df
#         .merge(
#             weather_df.drop(columns=['sunshine_hours'], errors='ignore'),
#             on="field_id", 
#             how="left"
#         )
#         .merge(soil_df, on="field_id", how="left")
#         "avg_rain_mm", # Renamed
#         # "avg_humidity", # Dropped
#         "heatwave_days", # Renamed
#         "ndvi_mean", # Renamed
#         "ndvi_max", # Renamed
#         "ndvi_std", # Renamed
#         "area_ha", # Placeholder (value 1.0)
#     ]
    
#     # Ensure only existing columns are selected
#     final_feature_cols = [col for col in feature_cols if col in df.columns]
    
#     X = df[final_feature_cols].copy()
#     y = df[target_col].copy()
#     return X, y, final_feature_cols
import pandas as pd
import numpy as np
# Import paths from config file
from .config import YIELD_CSV, WEATHER_CSV, SOIL_CSV, NDVI_CSV

def load_raw_data():
    """Loads raw data using paths defined in config.py."""
    # NOTE: These paths assume a standard project directory structure.
    yield_df = pd.read_csv(YIELD_CSV)
    weather_df = pd.read_csv(WEATHER_CSV)
    soil_df = pd.read_csv(SOIL_CSV)
    ndvi_df = pd.read_csv(NDVI_CSV)
    return yield_df, weather_df, soil_df, ndvi_df
    
def merge_data():
    """Merges all data files based on plot_id."""
    yield_df, weather_df, soil_df, ndvi_df = load_raw_data()
    
    # CRITICAL FIX: Rename 'field_id' in weather.csv to 'plot_id' for consistent merging
    weather_df.rename(columns={'field_id': 'plot_id'}, inplace=True)
    
    df = yield_df.merge(ndvi_df, on='plot_id', how='inner')
    df = df.merge(soil_df, on='plot_id', how='inner')
    df = df.merge(weather_df, on='plot_id', how='inner')
    
    # No encoders used since we will use pd.get_dummies (One-Hot Encoding)
    return df, None 

def build_feature_target(df):
    """
    Creates features, applies CRITICAL Log-Transformation to the target, 
    and returns features (X) and log-transformed target (y_log).
    """
    
    # --- CRITICAL FEATURE ENGINEERING ---
    df['planting_date'] = pd.to_datetime(df['planting_date'])
    # Extract month and day as new features
    df['planting_month'] = df['planting_date'].dt.month
    df['planting_day'] = df['planting_date'].dt.day
    
    # Handle Categorical Features using One-Hot Encoding
    df = pd.get_dummies(df, columns=['crop_type', 'soil_type'], drop_first=True)
    
    # Define features and target. Exclude ID, temporal columns, and original target.
    feature_cols = [col for col in df.columns if col not in ['plot_id', 'year', 'planting_date', 'harvest_yield_kg']]
    
    X = df[feature_cols]
    y = df['harvest_yield_kg']
    
    # *** CRITICAL FIX: Log-Transform the Target ***
    y_log = np.log(y)
    
    return X, y_log, feature_cols