import os
import csv
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.conf import settings
from .network_analysis import build_and_infer_bn_9, build_and_infer_bn_14,build_and_infer_dbn
from .neural_network_v4 import  load_and_preprocess_data,evaluate_model, build_and_train_bayesian_network, plot_confusion_matrix, plot_dataset_distribution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_reliability_data(file_path):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def dashboard(request):
    # Infer DBNs for 9 and 14 nodes
    prob_9, bn_9 = build_and_infer_bn_9()
    prob_14, bn_14 = build_and_infer_bn_14()

    return render(request, 'index.html', {
        'prob_9': prob_9,
        'prob_14': prob_14,
        'image_9': 'static/images/inference_9_nodes.png',
        'image_14': 'static/images/inference_14_nodes.png',
    })

def data_entry(request):
    if request.method == 'POST':
        data = request.POST.dict()
        csv_file = request.FILES['csv-file']
        file_path = os.path.join(settings.BASE_DIR, 'data', csv_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)
        return JsonResponse({"status": "success", "data": data})
    return JsonResponse({"status": "error", "message": "Invalid request method"})

def custom_static(request, filename):
    file_path = os.path.join(settings.BASE_DIR, 'static/images', filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return HttpResponse(f.read(), content_type="image/png")
    else:
        return HttpResponse(status=404)

def view_csv(request):
    if request.method == 'GET':
        csv_file = request.GET.get('csv-file')
        file_path = os.path.join(settings.BASE_DIR, 'data', csv_file)
        if not os.path.exists(file_path):
            return JsonResponse({"status": "error", "message": "File not found"})
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        return JsonResponse({"status": "success", "data": data})
    return JsonResponse({"status": "error", "message": "Invalid request method"})

def build_and_train_model_view(request):
    network = request.GET.get('network')
    if network not in ['ieee_9_nodes', 'ieee_14_nodes']:
        return JsonResponse({"status": "error", "message": "Invalid network type"})
    
    if network == 'ieee_9_nodes':
        data_file = os.path.join(settings.BASE_DIR, 'data/ieee_9_nodes_reliability.csv')
    else:
        data_file = os.path.join(settings.BASE_DIR, 'data/ieee_14_nodes_reliability.csv')

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, df = load_and_preprocess_data(data_file)

        # Build and train the Bayesian network
        bn = build_and_train_bayesian_network(X_train, y_train, network)

        # Evaluate the Bayesian network
        evaluation_results = evaluate_model(bn, X_test, y_test, network)

        # Save confusion matrix
        cm_filename = 'static/images/confusion_matrix.png'
        plot_confusion_matrix(evaluation_results['confusion_matrix'], target_names=np.unique(y_test), normalize=False, filename=cm_filename)

        # Save dataset distribution
        dataset_filename = 'static/images/dataset_distribution.png'
        plot_dataset_distribution(X_train, X_test, filename=dataset_filename)

        return JsonResponse({
            "status": "success",
            "accuracy": evaluation_results['accuracy'],
            "classification_report": evaluation_results['classification_report'],
            "confusion_matrix": evaluation_results['confusion_matrix']
        })
    except ValueError as e:
        return JsonResponse({"status": "error", "message": str(e)})


def report(request):
    network = request.GET.get('network')
    if network not in ['ieee_9_nodes', 'ieee_14_nodes']:
        return JsonResponse({"status": "error", "message": "Invalid network type"})
    
    report_content = f"Rapport synthétique pour le réseau {network}.\n\n"
    if network == 'ieee_9_nodes':
        edges = [("N1", "N2"), ("N1", "N3"), ("N2", "N4"), ("N2", "N5"), ("N3", "N6"), ("N4", "N7"), ("N5", "N8"), ("N6", "N9")]
    else:
        edges = [
            ("N1", "N2"), ("N1", "N5"), ("N2", "N3"), ("N2", "N4"), ("N3", "N4"),
            ("N4", "N5"), ("N4", "N7"), ("N4", "N9"), ("N5", "N6"), ("N6", "N11"),
            ("N6", "N12"), ("N7", "N8"), ("N7", "N9"), ("N9", "N10"), ("N9", "N14"),
            ("N10", "N11"), ("N12", "N13")
        ]
    data = read_reliability_data(os.path.join(settings.BASE_DIR, f'data/{network}_reliability.csv'))
    probabilities, _ = build_and_infer_dbn(os.path.join(settings.BASE_DIR, f'data/{network}_reliability.csv'), edges)
    
    for node, prob in probabilities.items():
        report_content += f"Probabilité de disponibilité pour {node}: {prob:.2f}\n"

    return JsonResponse({"status": "success", "report": report_content})
