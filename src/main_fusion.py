# src/main_fusion.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- 1. CONFIGURATION AND FILE PATHS ---
SENSOR_PATH = 'data/DS1_Sensor_Data.xlsx'
METEO_PATH = 'data/DS2_Meteorology_Data.xlsx'

def load_data_safely():
    if not os.path.exists(SENSOR_PATH) or not os.path.exists(METEO_PATH):
        raise FileNotFoundError(
            "Data files not found. Please ensure 'DS1_Sensor_Data.xlsx' and "
            "'DS2_Meteorology_Data.xlsx' are located in the './data/' directory. "
        )
    
    # Load dataframes
    df_sensor = pd.read_excel(SENSOR_PATH)
    df_meteo = pd.read_excel(METEO_PATH)
    
    return df_sensor, df_meteo

def perform_data_fusion(df_sensor, df_meteo):
    
    # --- ETL Step 1: Cleaning/Feature Engineering on Meteorology Data (DS2) ---
    df_meteo.drop('Weather_Station_ID', axis=1, inplace=True) 
    
    # Ensure Timestamp is the key for joining (Temporal Fusion)
    df_sensor['Timestamp'] = pd.to_datetime(df_sensor['Timestamp'])
    df_meteo['Timestamp'] = pd.to_datetime(df_meteo['Timestamp'])

    fusion_df = pd.merge(df_sensor, df_meteo, on='Timestamp', how='inner')
    
    # --- ETL Step 2: Extract temporal features from the fused data ---
    fusion_df['Hour'] = fusion_df['Timestamp'].dt.hour
    fusion_df['DayOfWeek'] = fusion_df['Timestamp'].dt.dayofweek
    fusion_df.drop('Timestamp', axis=1, inplace=True)
    
    return fusion_df

def build_and_evaluate_model(fusion_df):
    
    X = fusion_df.drop('AQI_Target', axis=1)
    y = fusion_df['AQI_Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction and Evaluation
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation (Linear Regression) ---")
    print(f"Features Used: {list(X.columns)}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    
    return model

if __name__ == '__main__':
    print("--- Python AQI Data Fusion and ML Prediction Module ---")
    
    try:
        df_sensor, df_meteo = load_data_safely()
        
        # Perform Fusion and ETL
        fusion_data = perform_data_fusion(df_sensor, df_meteo)
        print(f"✅ Data Fusion Successful. Final DataFrame Shape: {fusion_data.shape}")
        
        model = build_and_evaluate_model(fusion_data)
        
        print("\n✅ Module execution complete.")
        
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
    except Exception as e:
        print("\n❌ CRITICAL ERROR: An internal processing error occurred during ML pipeline.")
