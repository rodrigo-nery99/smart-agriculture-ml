from src.data_processing import load_data, prepare_knn_data, prepare_apriori_data
from src.models import train_knn, recommend_crop, generate_association_rules

def main():
    # 1. Carregamento dos dados
    filepath = 'data/Crop_recommendation.csv'
    df = load_data(filepath)
    print("Dataset carregado com sucesso!\n")

    # 2. Processamento e Treinamento: KNN
    X_scaled, scaler, X, y = prepare_knn_data(df)
    knn_model = train_knn(X_scaled)

    # 3. Teste de Recomendação (KNN)
    condicoes_teste = {
        'N': 90, 'P': 42, 'K': 43,
        'temperature': 20.8, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9
    }
    recommend_crop(knn_model, scaler, df, condicoes_teste)

    # 4. Processamento e Regras de Associação: Apriori
    df_apriori_dummies = prepare_apriori_data(df)
    generate_association_rules(df_apriori_dummies)

if __name__ == "__main__":
    main()