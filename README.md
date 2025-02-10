# 🏆 Electra Project
Un tableau de bord interactif pour l'analyse de la fiabilité des réseaux électriques avec des Réseaux Bayésiens Dynamiques et un modèle de classification.

# 📌 Présentation du Projet
Electra Project est une application web développée avec Django qui permet d’analyser et de visualiser la fiabilité des réseaux électriques à l’aide de réseaux bayésiens dynamiques et d’un modèle de classification basé sur l’apprentissage automatique.

L’application permet de :
✔ Visualiser un réseau bayésien dynamique (DBN) des réseaux électriques IEEE 9 et IEEE 14.

✔ Effectuer des inférences sur ces réseaux pour estimer la fiabilité des nœuds.

✔ Construire et entraîner un modèle de classification pour prédire l’état des nœuds en fonction de leur disponibilité.

✔ Générer des rapports synthétiques et des matrices de confusion pour évaluer les performances du modèle.

# 🛠 Technologies Utilisées
🔹 Backend : Django (Python)

🔹 Machine Learning : scikit-learn, pyAgrum

🔹 Visualisation : matplotlib, seaborn, graphviz

🔹 Frontend : HTML, CSS, JavaScript

🔹 Base de données : SQLite (ou autre selon configuration)

# 📂 Structure du Projet
electra_project/
├── electra_project/            # Répertoire principal Django
│   ├── __init__.py
│   ├── settings.py             # Configuration du projet Django
│   ├── urls.py                 # Routes principales de l’application
│   ├── wsgi.py / asgi.py       # Déploiement du serveur
│
├── network_reliability/        # Application Django principale
│   ├── __init__.py
│   ├── admin.py                # Interface d'administration Django
│   ├── apps.py                 # Configuration de l’application Django
│   ├── models.py               # Modèles de base de données Django
│   ├── views.py                # Logique de traitement des requêtes
│   ├── network_analysis.py      # Création et inférence des DBNs
│   ├── neural_network.py        # Construction et entraînement du modèle ML
│   ├── templates/
│   │   ├── index.html           # Interface utilisateur
│   ├── static/
│   │   ├── css/styles.css       # Styles CSS
│   │   ├── js/scripts.js        # Scripts JavaScript
│
├── data/                       # Données utilisées pour l'analyse
│   ├── ieee_9_nodes_reliability.csv
│   ├── ieee_14_nodes_reliability.csv
│
├── manage.py                    # Script de gestion Django
├── electra_project.spec          # Spécification du projet (déploiement)
├── requirements.txt              # Dépendances du projet
├── run_waitress.py               # Script pour exécuter l’application avec Waitress
# 🚀 Installation et Exécution
1️⃣ Cloner le projet

git clone https://github.com/ton_profil/electra_project.git
cd electra_project

2️⃣ Créer un environnement virtuel et installer les dépendances
python -m venv venv
source venv/bin/activate  # Pour Mac/Linux
venv\Scripts\activate     # Pour Windows
pip install -r requirements.txt

3️⃣ Appliquer les migrations et démarrer le serveur

python manage.py migrate
python manage.py runserver
L'application sera accessible sur http://127.0.0.1:8000/
# 📊 Fonctionnalités

✔ Visualisation des réseaux IEEE 9 et IEEE 14 via des graphes bayésiens dynamiques.

✔ Inférence probabiliste sur l’état des nœuds.

✔ Construction et évaluation d’un modèle de classification basé sur scikit-learn.

✔ Affichage des résultats sous forme de rapports, matrices de confusion et graphiques.

# 📌 Exemples de Résultats
🔹 Graphes Bayésiens Dynamiques
Voir static/images
🔹 Matrice de Confusion
(Évaluation des performances du modèle de classification.)
Voir static/images
# ❗ Limitations et Améliorations Futures
🔸 Limitation : Les performances du modèle de classification sont limitées par la taille des données disponibles.
🔸 Améliorations : Intégration d’un modèle plus performant et utilisation d’un jeu de données plus grand pour améliorer la précision.

# 🎯 Contribuer
Les contributions sont les bienvenues ! Si vous souhaitez apporter des améliorations, n’hésitez pas à forker le projet et proposer des pull requests.
