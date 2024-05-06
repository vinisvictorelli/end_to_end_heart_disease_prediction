import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pandas as pd

data = 'data/dataset_heart_preprocessed.parquet'

def model_evaluation(data):
    heart_df = pd.read_parquet(data)
    description = heart_df.drop(columns=['HadHeartAttack'])
    target = heart_df['HadHeartAttack']
    # Contagem de valores únicos na coluna 'target'
    contagem_valores = target.value_counts()

    print(contagem_valores)
    # Criando o objeto SMOTE
    smote = SMOTE(random_state=42)

    # Aplicando o SMOTE
    X_resampled, y_resampled = smote.fit_resample(description, target)
    # Criando o objeto de undersampling
    #undersampler = RandomUnderSampler(random_state=None)

    # Aplicando o undersampling
    #X_resampled, y_resampled = undersampler.fit_resample(description, target)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, 
        test_size=0.2, 
        random_state=42)
    # Instantiate the logistic regression model
    model = LogisticRegression()
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy*100,2))

    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Display predicted probabilities
    print("Predicted Probabilities:\n", round(y_pred_proba[0][1] * 100, 2))

def model_train_save(data):
    # Carregar os dados do arquivo Parquet
    heart_df = pd.read_parquet(data)

    # Separar as features (X) e o target (y)
    description = heart_df.drop(columns=['HadHeartAttack'])
    target = heart_df['HadHeartAttack']
    # Criando o objeto SMOTE
    smote = SMOTE(random_state=42)

    # Aplicando o SMOTE
    X_resampled, y_resampled = smote.fit_resample(description, target)
    # Criando o objeto de undersampling
    #undersampler = RandomUnderSampler(random_state=None)

    # Aplicando o undersampling
    #X_resampled, y_resampled = undersampler.fit_resample(description, target)
    # Instanciar o modelo de regressão logística
    model = LogisticRegression()

    # Treinar o modelo com todos os dados disponíveis
    model.fit(X_resampled, y_resampled)

    # Salvar o modelo treinado em formato pickle
    with open('modelo_logistico.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Modelo treinado e salvo com sucesso!")

def undersample_techniques(data):
    # Suponha que você tenha um DataFrame chamado df com uma coluna 'classe' que indica a classe de cada amostra
    # Exemplo de DataFrame
    df = pd.read_parquet(data)

    # Separando features e labels
    X = df.drop(columns=['HadHeartAttack'])
    y = df['HadHeartAttack']

    # Criando o objeto de undersampling
    undersampler = RandomUnderSampler(random_state=42)

    # Aplicando o undersampling
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # Criando o objeto SMOTE
    #smote = SMOTE(random_state=42)

    # Aplicando o SMOTE
    #X_resampled, y_resampled = smote.fit_resample(X, y)

    # Visualizando a distribuição das classes após o undersampling
    print(pd.Series(y_resampled).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, 
        test_size=0.2, 
        random_state=42)
    # Instantiate the logistic regression model
    model = LogisticRegression()
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy*100,2))

    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Display predicted probabilities
    print("Predicted Probabilities:\n", round(y_pred_proba[0][1] * 100, 2))

model_train_save(data)

