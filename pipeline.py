from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline

DATA_PATH = Path('data/')

PROCESSED_FILE_PATH_PARQUET = DATA_PATH / 'heart_2022_no_nans.parquet'

VAR_LIST_PATH = Path('data/vars_list_with_descriptions.txt')


"""Importing data"""
def preprocess(data):
    df = pd.read_parquet(data)
    #Selecionar apenas as colunas necessárias para alimentar o modelo
    selected_features = [
        'Sex','AgeCategory','PhysicalHealthDays', 'PhysicalActivities', 'SleepHours', 'HadHeartAttack', 
        'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD','HadDepressiveDisorder', 'HadKidneyDisease', 
        'HadArthritis','HadDiabetes', 'SmokerStatus','ECigaretteUsage', 'AlcoholDrinkers', 
        'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver','CovidPos','HeightInMeters','WeightInKilograms'
    ]
    #Criar novo dataframe apenas com colunas desejadas
    df = df[selected_features]
    df['BMI'] = df['WeightInKilograms']/(df['HeightInMeters']**2)
    df['BMI'] = round(df['BMI'],2)
    df = df.drop(columns=['HeightInMeters','WeightInKilograms'])
    df.to_parquet('teste.parquet')
    # Selecionar todas as colunas do tipo 'object' para o LabelEncoder
    colunas_labelencoder = df.select_dtypes(include=['object']).columns.tolist()

    # Selecionar todas as colunas do tipo 'float64' para o StandardScaler
    colunas_standardscaler = df.select_dtypes(include=['float64']).columns.tolist()

    # Inicializar o LabelEncoder
    labelencoder = LabelEncoder()

    # Inicializar o StandardScaler
    #scaler = StandardScaler()

    # Aplicar o LabelEncoder nas colunas 'object'
    for coluna in colunas_labelencoder:
        df[coluna] = labelencoder.fit_transform(df[coluna])

    # Aplicar o StandardScaler nas colunas 'float64'
    #for coluna in colunas_standardscaler:
        #df[coluna] = scaler.fit_transform(df[[coluna]])

    df.to_parquet('data/dataset_heart_preprocessed.parquet',index=False)

def preprocess_user_input(data):
    df = data
    #Selecionar apenas as colunas necessárias para alimentar o modelo
    selected_features = [
        'Sex','AgeCategory','PhysicalHealthDays', 'PhysicalActivities', 'SleepHours', 
        'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD','HadDepressiveDisorder', 'HadKidneyDisease', 
        'HadArthritis','HadDiabetes', 'SmokerStatus','ECigaretteUsage', 'AlcoholDrinkers', 
        'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver','CovidPos','HeightInMeters','WeightInKilograms'
    ]
    #Criar novo dataframe apenas com colunas desejadas
    df = df[selected_features]

    df['BMI'] = df['WeightInKilograms']/df['HeightInMeters']**2
    df['BMI'] = round(df['BMI'],2)
    df = df.drop(columns = ['HeightInMeters','WeightInKilograms'])
    df.to_parquet('teste.parquet')

    # Selecionar todas as colunas do tipo 'object' para o LabelEncoder
    colunas_labelencoder = df.select_dtypes(include=['object']).columns.tolist()

    # Selecionar todas as colunas do tipo 'float64' para o StandardScaler
    colunas_standardscaler = df.select_dtypes(include=['float64']).columns.tolist()

    # Inicializar o LabelEncoder
    labelencoder = LabelEncoder()

    # Inicializar o StandardScaler
    scaler = StandardScaler()

    # Aplicar o LabelEncoder nas colunas 'object'
    for coluna in colunas_labelencoder:
        df[coluna] = labelencoder.fit_transform(df[coluna])

    # Aplicar o StandardScaler nas colunas 'float64'
    for coluna in colunas_standardscaler:
        df[coluna] = scaler.fit_transform(df[[coluna]])
    
    return df

preprocess(PROCESSED_FILE_PATH_PARQUET)


