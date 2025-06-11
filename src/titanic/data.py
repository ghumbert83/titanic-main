"""
Load, preprocess, prepare, and save the Titanic dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

DATA_DIR = "data/"

def load_data():
    """
    Load the Titanic dataset from a CSV file.
    
    Returns:
        DataFrame: The loaded Titanic dataset.
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR,"train.csv"),index_col=0)
    test_df_missing_target  = pd.read_csv(os.path.join(DATA_DIR,"test.csv"),index_col=0)
    target_test_df          = pd.read_csv(os.path.join(DATA_DIR,"gender_submission.csv"),index_col=0)
    test_df = pd.merge(  test_df_missing_target
                   , target_test_df
                   , left_index=True
                   , right_index=True
                   , how='left'
                   , suffixes=('', '_target')).copy()
    
    return train_df, test_df

       
def clean_data(df):
    """
    clean the Titanic dataset.
    
    Args:
        df (DataFrame): The Titanic dataset.
        
    Returns:
        DataFrame: The preprocessed Titanic dataset.
    """
    df.drop(columns['Name', 'Ticket', 'Cabin'], inplace=True)
    imputer = SimpleImputer().set_output(transform="pandas")
    imputer.fit(df[['Age']])
    df['Age'] = imputer.transform(df[['Age']])

    return df

def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: 
    """
    Prepare the Titanic dataset for training.
    
    Args:
        df (DataFrame): The preprocessed Titanic dataset.
        
    Returns:
        tuple: A tuple containing [X,y] the features DataFrame and the target Series.
    """
    from sklearn.preprocessing import StandardScaler

    numeric_features = ['Age', 'Fare']

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

    from sklearn.preprocessing import OneHotEncoder
    categorical_features = ["Sex", "Embarked"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first').set_output(transform="pandas")
    df_encoded = encoder.fit_transform(df[categorical_features])
    df_final = pd.concat([df_scaled, df_encoded], axis=1).drop(columns=["Sex", "Embarked"])

    X = df_final.drop(columns=["Survived"])
    y = df_final["Survived"]

    return X, y
