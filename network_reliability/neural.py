import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pyAgrum as gum
import pyAgrum.lib.bn2graph as bn2graph
from graphviz import Source
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 1. Charger et préparer les données
def load_and_preprocess_data(data_file):
    df = pd.read_csv(data_file)
    print("Columns in df:", df.columns)

    # Vérifier que les colonnes attendues sont bien présentes
    if 'Nodes' not in df.columns or 'P_dispo' not in df.columns:
        raise ValueError("Le fichier CSV doit contenir les colonnes 'Nodes' et 'P_dispo'.")

    # La variable explicative est "Availability", la cible est "Nodes"
    X = df[['P_dispo']].values  # Feature unique
    y = df['Nodes'].astype(str)  # Labels sous forme de texte

    # Division en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# 📌 2. Construire et entraîner le modèle de classification MLP
def build_and_train_model(X_train, y_train):
    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

# 📌 3. Évaluer le modèle
def evaluate_model(mlp, X_test, y_test):
    y_pred = mlp.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"\n✅ Précision du modèle: {mlp.score(X_test, y_test):.2f}")
    print("\n📊 Matrice de Confusion:\n", cm)
    print("\n📜 Rapport de Classification:\n", cr)

    return {
        'accuracy': mlp.score(X_test, y_test),
        'classification_report': cr,
        'confusion_matrix': cm
    }

# 📌 4. Construire un réseau bayésien basé sur les données
def build_bayesian_network():
    bn = gum.BayesNet('Bayesian_Model')

    # Ajouter la variable "Availability" et les nœuds du réseau
    availability_var = gum.LabelizedVariable('P_dispo', 'P_dispo', 2)
    bn.add(availability_var)

    for i in range(1, 10):
        bn.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))
        bn.addArc('P_dispo', f'N{i}')

    return bn

# 📌 5. Apprendre le réseau bayésien à partir des données
def train_bayesian_network(X_train, y_train):
    bn = build_bayesian_network()

    # Conversion en DataFrame pour pyAgrum
    df_train = pd.DataFrame(X_train, columns=['P_dispo'])
    df_train['Target'] = y_train.values

    learner = gum.BNLearner(df_train)
    learner.useSmoothingPrior(1)
    bn_trained = learner.learnParameters(bn.dag())

    return bn_trained

# 📌 6. Afficher la matrice de confusion
def plot_confusion_matrix(cm, filename=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=True, yticklabels=True)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# 📌 7. Exécuter le pipeline
if __name__ == "__main__":
    data_file = 'C:/Users/PC/Mon Drive/electra_project - Copie/data/ieee_9_nodes_reliability.csv'

    # Chargement et prétraitement des données
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file)

    # Entraînement du modèle de classification
    mlp = build_and_train_model(X_train, y_train)

    # Évaluation du modèle
    eval_results = evaluate_model(mlp, X_test, y_test)

    # Sauvegarde de la matrice de confusion
    cm_filename = 'static/images/confusion_matrix.png'
    plot_confusion_matrix(eval_results['confusion_matrix'], cm_filename)

    # Construction et apprentissage du réseau bayésien
    bn_trained = train_bayesian_network(X_train, y_train)

    print("\n✅ Réseau bayésien entraîné avec succès.")
