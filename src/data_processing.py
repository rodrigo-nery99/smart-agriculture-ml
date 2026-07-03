import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV."""
    return pd.read_csv(filepath)

def prepare_knn_data(df: pd.DataFrame):
    """Separa as features e aplica a padronização para o KNN."""
    X = df.drop('label', axis=1)
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, X, y

def prepare_apriori_data(df: pd.DataFrame) -> pd.DataFrame:
    """Discretiza variáveis contínuas e aplica One-Hot Encoding para o Apriori."""
    X = df.drop('label', axis=1)
    y = df['label']
    
    df_apriori = pd.DataFrame()
    for col in X.columns:
        df_apriori[col] = pd.qcut(df[col], q=3, labels=['Baixo', 'Medio', 'Alto'])

    df_apriori['label'] = y
    
    # Transforma em variáveis dummy
    df_apriori_dummies = pd.get_dummies(df_apriori)
    return df_apriori_dummies