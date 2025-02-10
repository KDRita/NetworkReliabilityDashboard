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
import numpy as np

def load_and_preprocess_data(data_file):
    # Load data from CSV
    df = pd.read_csv(data_file)
    
    # Print columns of df to verify structure
    print("Columns in df:", df.columns)
    
    # Separate features and target
    X = df.drop(columns=['Nodes'])  # Drop 'Nodes' column
    y = df['Nodes']  # Target is the 'Nodes' column (categorical labels)
    
    # Encode categorical labels (Nodes) into numeric form if needed
    X = pd.get_dummies(X)  # Convert categorical variables into dummy/indicator variables
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def build_and_train_model(X_train, y_train):
    # Build MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

def evaluate_model(mlp, X_test, y_test, network):
    # Predict using the trained MLP model
    y_pred = mlp.predict(X_test)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate the Bayesian network image
    bn = build_bayesian_network(network)
    dot = bn2graph.BN2dot(bn)
    
    # Save the dot object to a DOT file
    dot_filename = f'static/images/{network}_model.dot'
    dot.write(dot_filename)
    
    # Convert DOT file to PNG using Graphviz
    png_filename = f'static/images/{network}_model.png'
    
    # Attempt to render PNG, handling potential encoding issues
    try:
        Source.from_file(dot_filename).render(png_filename, format='png', cleanup=True)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError occurred: {e}")
        # If there's an issue, print or log the error and handle accordingly
    
    # Generate the classification report
    cr = classification_report(y_test, y_pred)
    print(f"Evaluation Results:\nAccuracy: {mlp.score(X_test, y_test)}\n\nClassification Report:\n{cr}")

    return {
        'accuracy': mlp.score(X_test, y_test),
        'classification_report': cr,
        'confusion_matrix': cm.tolist()
    }

def build_bayesian_network(network):
    # Build Bayesian network based on network type
    if network == 'ieee_9_nodes':
        bn = gum.BayesNet('IEEE_9_nodes_model')
        for i in range(1, 10):
            bn.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))
        edges = [("N1", "N2"), ("N1", "N3"), ("N2", "N4"), ("N2", "N5"), ("N3", "N6"), ("N4", "N7"), ("N5", "N8"), ("N6", "N9")]
        for edge in edges:
            bn.addArc(edge[0], edge[1])
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

    return bn

def build_and_train_bayesian_network(X_train, y_train, network):
    # Create Bayesian network based on network type
    bn = build_bayesian_network(network)
    
    # Convert X_train to a DataFrame
    feature_names = [f'N{i+1}' for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    
    # Combine features and target into one DataFrame
    train_df = X_train_df.copy()
    train_df['Target'] = y_train.values
    
    # Learn Bayesian network structure and parameters
    learner = gum.BNLearner(train_df, bn)
    learner.useSmoothingPrior(1)
    bn_learned = learner.learnParameters(bn.dag())
    
    return bn_learned

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=False, filename=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap=cmap, fmt=".2f", xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_dataset_distribution(X_train, X_test, filename=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(X_train, ax=ax[0])
    ax[0].set_title('Distribution of Training Dataset')

    sns.histplot(X_test, ax=ax[1])
    ax[1].set_title('Distribution of Test Dataset')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == "__main__":
    data_file = 'C:/Users/PC/Mon Drive/electra_project - Copie/data/ieee_9_nodes_reliability.csv'  # Replace with your data file path
    network = 'ieee_9_nodes'  # Specify your network type

    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_file)

    # Build and train the model
    mlp = build_and_train_model(X_train, y_train)

    # Evaluate the model
    evaluation_results = evaluate_model(mlp, X_test, y_test, network)

    # Directory to save images
    image_directory = 'static/images/'

    # Plot and save confusion matrix
    cm_filename = image_directory + 'confusion_matrix.png'
    plot_confusion_matrix(evaluation_results['confusion_matrix'], target_names=np.unique(y_test), normalize=False, filename=cm_filename)

    # Plot and save dataset distribution
    dataset_filename = image_directory + 'dataset_distribution.png'
    plot_dataset_distribution(X_train, X_test, filename=dataset_filename)

    # Build and train Bayesian network
    bn = build_and_train_bayesian_network(X_train, y_train, network)
    print("Bayesian Network Structure:", bn)
