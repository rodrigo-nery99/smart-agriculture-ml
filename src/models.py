import pandas as pd
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules

def train_knn(X_scaled, n_neighbors=5):
    """Treina e retorna o modelo KNN."""
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_scaled)
    return knn

def recommend_crop(knn_model, scaler, df, nova_amostra_dict):
    """Recebe novos dados ambientais, padroniza e retorna a recomendação."""
    amostra_df = pd.DataFrame([nova_amostra_dict])
    amostra_scaled = scaler.transform(amostra_df)

    # Calcular similaridade
    distancias, indices = knn_model.kneighbors(amostra_scaled)

    # Gerar recomendações baseadas nos índices vizinhos
    recomendacoes = df.iloc[indices[0]]['label'].value_counts()

    print("Condições de Entrada:")
    for k, v in nova_amostra_dict.items():
        print(f" - {k}: {v}")

    print("\nCulturas Recomendadas (baseado em similaridade):")
    for cultura, contagem in recomendacoes.items():
        print(f" - {cultura.capitalize()} (Apareceu em {contagem}/5 vizinhos mais próximos)")

def generate_association_rules(df_apriori_dummies, min_support=0.02, min_threshold=1.5):
    """Gera e exibe as regras de associação usando Apriori."""
    frequent_itemsets = apriori(df_apriori_dummies, min_support=min_support, use_colnames=True)
    regras = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)

    # Filtrar regras interessantes
    regras_finais = regras[regras['consequents'].apply(lambda x: any('label' in item for item in x))]
    regras_finais = regras_finais.sort_values(by=['lift', 'confidence'], ascending=[False, False])

    print("\nInterpretação da Principal Regra Encontrada:")
    if not regras_finais.empty:
        principal_regra = regras_finais.iloc[0]
        ant = list(principal_regra['antecedents'])
        cons = list(principal_regra['consequents'])[0].replace("label_", "")

        print(f"Se o solo/clima apresentar {ant},")
        print(f"Há uma confiança de {principal_regra['confidence']*100:.2f}% de que a cultura adequada é '{cons}'.")
        print(f"O Lift de {principal_regra['lift']:.2f} indica que essa associação é {principal_regra['lift']:.2f} vezes mais forte do que o acaso.")
    else:
        print("O modelo não encontrou regras de associação fortes o suficiente.")