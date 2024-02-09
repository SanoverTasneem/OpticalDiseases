# Import necessary modules
from data_preprocessing import preprocess_data
from model_pipeline import train_and_evaluate, plot_correlation_matrix
import pandas as pd
import numpy as np

def main():
    df_path = 'D:\\Northeastern\\Semester 2\\Python\\Optical Diseases\\full_df.csv'
    df_cleaned = preprocess_data(df_path)
    print(df_cleaned.head())
    
    # Define X (features) and y (target) for models
    X = df_cleaned[['Patient Age', 'Patient Sex', 'D', 'H']]  # Adjust features as needed
    y = df_cleaned['MONR']  # Adjust target variable as needed
    

    # Train and evaluate your models
    train_and_evaluate(X, y)

    print("Completed")

if __name__ == "__main__":
    main()
