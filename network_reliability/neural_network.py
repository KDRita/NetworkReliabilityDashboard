import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pyAgrum as gum
import pyAgrum.lib.image as gimg

def build_and_visualize_model(data_file, network):
    # Charger les données
    data = pd.read_csv(data_file)
    data['Nodes'] = data['Nodes'].map(node_mapping)
    node_mapping = {f"N{i}": i-1 for i in range(1, 10)}
    # Séparer les caractéristiques et la cible
    X = data([0,0.98],[1,0.941],[2,0.9116],[3,0.8764],[4,0.93286],[5,0.901104],[6,0.876852],[7,0.919787],[8,0.86945264])
    y = data['Nodes']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardiser les caractéristiques
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Construire le modèle de réseau de neurones
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)

    # Prédire les résultats
    y_pred = mlp.predict(X_test)

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Générer et sauvegarder la matrice de confusion
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matrice de Confusion pour {network}')
    plt.colorbar()
    tick_marks = np.arange(len(set(y)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie Valeur')
    plt.savefig('static/images/confusion_matrix.png')
    plt.close()

    # Générer le rapport de classification
    cr = classification_report(y_test, y_pred, output_dict=True)

    # Créer le modèle Bayésien
    if network == 'ieee_9_nodes':
        bn = gum.BayesNet('IEEE_9_nodes_model')
        for i in range(1, 10):
            bn.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))
        edges = [("N1", "N2"), ("N1", "N3"), ("N2", "N4"), ("N2", "N5"), ("N3", "N6"), ("N4", "N7"), ("N5", "N8"), ("N6", "N9")]
    else:
        bn = gum.BayesNet('IEEE_14_nodes_model')
        for i in range(1, 15):
            bn.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))
        edges = [
            ("N1", "N2"), ("N1", "N5"), ("N2", "N3"), ("N2", "N4"), ("N3", "N4"),
            ("N4", "N5"), ("N4", "N7"), ("N4", "N9"), ("N5", "N6"), ("N6", "N11"),
            ("N6", "N12"), ("N7", "N8"), ("N7", "N9"), ("N9", "N10"), ("N9", "N14"),
            ("N10", "N11"), ("N12", "N13")
        ]

    for edge in edges:
        bn.addArc(edge[0], edge[1])

    # Définir les CPTs (Tables de Probabilités Conditionnelles)
    if network == 'ieee_9_nodes':
        bn.cpt("N1").fillWith([0.98, 0.02])
        bn.cpt("N2")[{'N1': 0}] = [0.95, 0.05]
        bn.cpt("N2")[{'N1': 1}] = [0.5, 0.5]
        bn.cpt("N3")[{'N1': 0}] = [0.92, 0.08]
        bn.cpt("N3")[{'N1': 1}] = [0.5, 0.5]
        bn.cpt("N4")[{'N2': 0}] = [0.90, 0.10]
        bn.cpt("N4")[{'N2': 1}] = [0.5, 0.5]
        bn.cpt("N5")[{'N2': 0}] = [0.96, 0.04]
        bn.cpt("N5")[{'N2': 1}] = [0.5, 0.5]
        bn.cpt("N6")[{'N3': 0}] = [0.94, 0.06]
        bn.cpt("N6")[{'N3': 1}] = [0.5, 0.5]
        bn.cpt("N7")[{'N4': 0}] = [0.93, 0.07]
        bn.cpt("N7")[{'N4': 1}] = [0.5, 0.5]
        bn.cpt("N8")[{'N5': 0}] = [0.95, 0.05]
        bn.cpt("N8")[{'N5': 1}] = [0.5, 0.5]
        bn.cpt("N9")[{'N6': 0}] = [0.91, 0.09]
        bn.cpt("N9")[{'N6': 1}] = [0.5, 0.5]
    else:
        bn.cpt("N1").fillWith([0.98, 0.02])
        bn.cpt("N2")[{'N1': 0}] = [0.95, 0.05]
        bn.cpt("N2")[{'N1': 1}] = [0.5, 0.5]
        bn.cpt("N3")[{'N2': 0}] = [0.92, 0.08]
        bn.cpt("N3")[{'N2': 1}] = [0.5, 0.5]
        bn.cpt("N4")[{'N2': 0, 'N3': 0}] = [0.90, 0.10]
        bn.cpt("N4")[{'N2': 0, 'N3': 1}] = [0.50, 0.50]
        bn.cpt("N4")[{'N2': 1, 'N3': 0}] = [0.50, 0.50]
        bn.cpt("N4")[{'N2': 1, 'N3': 1}] = [0.50, 0.50]
        bn.cpt("N5")[{'N1': 0, 'N4': 0}] = [0.96, 0.04]
        bn.cpt("N5")[{'N1': 0, 'N4': 1}] = [0.50, 0.50]
        bn.cpt("N5")[{'N1': 1, 'N4': 0}] = [0.50, 0.50]
        bn.cpt("N5")[{'N1': 1, 'N4': 1}] = [0.50, 0.50]
        bn.cpt("N6")[{'N5': 0}] = [0.94, 0.06]
        bn.cpt("N6")[{'N5': 1}] = [0.5, 0.5]
        bn.cpt("N7")[{'N4': 0}] = [0.93, 0.07]
        bn.cpt("N7")[{'N4': 1}] = [0.5, 0.5]
        bn.cpt("N8")[{'N7': 0}] = [0.95, 0.05]
        bn.cpt("N8")[{'N7': 1}] = [0.5, 0.5]
        bn.cpt("N9")[{'N4': 0, 'N7': 0}] = [0.91, 0.09]
        bn.cpt("N9")[{'N4': 0, 'N7': 1}] = [0.50, 0.50]
        bn.cpt("N9")[{'N4': 1, 'N7': 0}] = [0.50, 0.50]
        bn.cpt("N9")[{'N4': 1, 'N7': 1}] = [0.50, 0.50]
        bn.cpt("N10")[{'N9': 0}] = [0.92, 0.08]
        bn.cpt("N10")[{'N9': 1}] = [0.5, 0.5]
        bn.cpt("N11")[{'N6': 0, 'N10': 0}] = [0.93, 0.07]
        bn.cpt("N11")[{'N6': 0, 'N10': 1}] = [0.50, 0.50]
        bn.cpt("N11")[{'N6': 1, 'N10': 0}] = [0.50, 0.50]
        bn.cpt("N11")[{'N6': 1, 'N10': 1}] = [0.50, 0.50]
        bn.cpt("N12")[{'N6': 0}] = [0.94, 0.06]
        bn.cpt("N12")[{'N6': 1}] = [0.5, 0.5]
        bn.cpt("N13")[{'N12': 0}] = [0.95, 0.05]
        bn.cpt("N13")[{'N12': 1}] = [0.5, 0.5]
        bn.cpt("N14")[{'N9': 0}] = [0.96, 0.04]
        bn.cpt("N14")[{'N9': 1}] = [0.5, 0.5]

    # Sauvegarder l'image du modèle
    gimg.export(bn, f'static/images/{network}_model.png')
    
    return {
        'accuracy': cr['accuracy'],
        'classification_report': cr,
        'confusion_matrix': cm.tolist()
    }
