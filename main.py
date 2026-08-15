import pandas as pd
from sklearn.model_selection import train_test_split

from src.eda import check_df
from src.feature_engineering import data_prep
from src.modeling import fit_models

def main():
    data_path = "data/diabetes.csv"
    print(f"=== 1. Chargement des données : {data_path} ===")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {data_path} est introuvable.")
        return

    # Visualisation EDA rapide
    check_df(df, head=3)

    print("\n=== 2. Séparation Train / Test ===")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n=== 3. Application du Pipeline de Prétraitement ===")
    X_train_prep, y_train_prep = data_prep(X_train, y_train)
    X_test_prep, y_test_prep = data_prep(X_test, y_test)

    print(f"Dimensions Train : {X_train_prep.shape}")
    print(f"Dimensions Test  : {X_test_prep.shape}")

    print("\n=== 4. Entraînement et Évaluation des Modèles ===")
    # Cette fonction exécute : Base Models -> GridSearch -> Voting Classifier
    voting_clf, best_models = fit_models(X_train_prep, y_train_prep)

    print("\n=== 5. Évaluation finale du Voting Classifier sur le jeu de TEST ===")
    test_acc = voting_clf.score(X_test_prep, y_test_prep)
    print(f"Accuracy sur le jeu de test : {test_acc:.4f}")

if __name__ == "__main__":
    main()