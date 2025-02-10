import pyAgrum as gum
import csv
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gimg

def build_bn_9_nodes():
    bn_9 = gum.BayesNet('IEEE_9_nodes')
    for i in range(1, 10):
        bn_9.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))
    
    edges_9 = [
        ("N1", "N2"), ("N1", "N3"), ("N2", "N4"), ("N2", "N5"),
        ("N3", "N6"), ("N4", "N7"), ("N5", "N8"), ("N6", "N9")
    ]
    for edge in edges_9:
        bn_9.addArc(edge[0], edge[1])
    
    bn_9.cpt("N1").fillWith([0.98, 0.02])
    bn_9.cpt("N2")[{'N1': 0}] = [0.95, 0.05]
    bn_9.cpt("N2")[{'N1': 1}] = [0.5, 0.5]
    bn_9.cpt("N3")[{'N1': 0}] = [0.92, 0.08]
    bn_9.cpt("N3")[{'N1': 1}] = [0.5, 0.5]
    bn_9.cpt("N4")[{'N2': 0}] = [0.90, 0.10]
    bn_9.cpt("N4")[{'N2': 1}] = [0.5, 0.5]
    bn_9.cpt("N5")[{'N2': 0}] = [0.96, 0.04]
    bn_9.cpt("N5")[{'N2': 1}] = [0.5, 0.5]
    bn_9.cpt("N6")[{'N3': 0}] = [0.94, 0.06]
    bn_9.cpt("N6")[{'N3': 1}] = [0.5, 0.5]
    bn_9.cpt("N7")[{'N4': 0}] = [0.93, 0.07]
    bn_9.cpt("N7")[{'N4': 1}] = [0.5, 0.5]
    bn_9.cpt("N8")[{'N5': 0}] = [0.95, 0.05]
    bn_9.cpt("N8")[{'N5': 1}] = [0.5, 0.5]
    bn_9.cpt("N9")[{'N6': 0}] = [0.91, 0.09]
    bn_9.cpt("N9")[{'N6': 1}] = [0.5, 0.5]
    
    return bn_9

def build_bn_14_nodes():
    bn_14 = gum.BayesNet('IEEE_14_nodes')
    for i in range(1, 15):
        bn_14.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))
    
    edges_14 = [
        ("N1", "N2"), ("N1", "N5"), ("N2", "N3"), ("N2", "N4"), ("N3", "N4"),
        ("N4", "N5"), ("N4", "N7"), ("N4", "N9"), ("N5", "N6"), ("N6", "N11"),
        ("N6", "N12"), ("N7", "N8"), ("N7", "N9"), ("N9", "N10"), ("N9", "N14"),
        ("N10", "N11"), ("N12", "N13")
    ]
    for edge in edges_14:
        bn_14.addArc(edge[0], edge[1])
    
    bn_14.cpt("N1").fillWith([0.98, 0.02])
    bn_14.cpt("N2")[{'N1': 0}] = [0.95, 0.05]
    bn_14.cpt("N2")[{'N1': 1}] = [0.5, 0.5]
    bn_14.cpt("N3")[{'N2': 0}] = [0.92, 0.08]
    bn_14.cpt("N3")[{'N2': 1}] = [0.5, 0.5]
    bn_14.cpt("N4")[{'N2': 0, 'N3': 0}] = [0.90, 0.10]
    bn_14.cpt("N4")[{'N2': 0, 'N3': 1}] = [0.50, 0.50]
    bn_14.cpt("N4")[{'N2': 1, 'N3': 0}] = [0.50, 0.50]
    bn_14.cpt("N4")[{'N2': 1, 'N3': 1}] = [0.50, 0.50]
    bn_14.cpt("N5")[{'N1': 0, 'N4': 0}] = [0.96, 0.04]
    bn_14.cpt("N5")[{'N1': 0, 'N4': 1}] = [0.50, 0.50]
    bn_14.cpt("N5")[{'N1': 1, 'N4': 0}] = [0.50, 0.50]
    bn_14.cpt("N5")[{'N1': 1, 'N4': 1}] = [0.50, 0.50]
    bn_14.cpt("N6")[{'N5': 0}] = [0.94, 0.06]
    bn_14.cpt("N6")[{'N5': 1}] = [0.5, 0.5]
    bn_14.cpt("N7")[{'N4': 0}] = [0.93, 0.07]
    bn_14.cpt("N7")[{'N4': 1}] = [0.5, 0.5]
    bn_14.cpt("N8")[{'N7': 0}] = [0.95, 0.05]
    bn_14.cpt("N8")[{'N7': 1}] = [0.5, 0.5]
    bn_14.cpt("N9")[{'N4': 0, 'N7': 0}] = [0.91, 0.09]
    bn_14.cpt("N9")[{'N4': 0, 'N7': 1}] = [0.50, 0.50]
    bn_14.cpt("N9")[{'N4': 1, 'N7': 0}] = [0.50, 0.50]
    bn_14.cpt("N9")[{'N4': 1, 'N7': 1}] = [0.50, 0.50]
    bn_14.cpt("N10")[{'N9': 0}] = [0.92, 0.08]
    bn_14.cpt("N10")[{'N9': 1}] = [0.5, 0.5]
    bn_14.cpt("N11")[{'N6': 0, 'N10': 0}] = [0.93, 0.07]
    bn_14.cpt("N11")[{'N6': 0, 'N10': 1}] = [0.50, 0.50]
    bn_14.cpt("N11")[{'N6': 1, 'N10': 0}] = [0.50, 0.50]
    bn_14.cpt("N11")[{'N6': 1, 'N10': 1}] = [0.50, 0.50]
    bn_14.cpt("N12")[{'N6': 0}] = [0.94, 0.06]
    bn_14.cpt("N12")[{'N6': 1}] = [0.5, 0.5]
    bn_14.cpt("N13")[{'N12': 0}] = [0.95, 0.05]
    bn_14.cpt("N13")[{'N12': 1}] = [0.5, 0.5]
    bn_14.cpt("N14")[{'N9': 0}] = [0.96, 0.04]
    bn_14.cpt("N14")[{'N9': 1}] = [0.5, 0.5]
    
    return bn_14

def read_reliability_data(file_path):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def build_and_infer_dbn(data_file, edges):
    bn = gum.BayesNet('Dynamic_Network')
    data = read_reliability_data(data_file)

    for i in range(1, len(data) + 1):
        bn.add(gum.LabelizedVariable(f'N{i}', f'N{i}', 2))

    for edge in edges:
        if not bn.existsArc(edge[0], edge[1]):
            bn.addArc(edge[0], edge[1])

    for i, row in enumerate(data):
        node = f'N{i + 1}'
        if len(bn.parents(node)) == 0:
            bn.cpt(node).fillWith([float(row['Availability']), 1 - float(row['Availability'])])
        else:
            parent_states = [2] * len(bn.parents(node))
            bn.cpt(node).fillWith([float(row['Availability'])] * (2 ** len(bn.parents(node))) +
                                  [1 - float(row['Availability'])] * (2 ** len(bn.parents(node))))

    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    probabilities = {f'N{i + 1}': ie.posterior(f'N{i + 1}')[0] for i in range(len(data))}
    
    return probabilities, bn

def build_and_infer_bn_9():
    bn_9 = build_bn_9_nodes()
    ie = gum.LazyPropagation(bn_9)
    ie.makeInference()
    probabilities = {f'N{i + 1}': ie.posterior(f'N{i + 1}')[0] for i in range(9)}
    gnb.showBN(bn_9)
    gnb.showInference(bn_9, evs={}, engine=ie)
    gimg.export(bn_9, 'static/images/graph_9_nodes.png')
    gimg.exportInference(bn_9, 'static/images/inference_9_nodes.png', evs={})
    return probabilities, bn_9

def build_and_infer_bn_14():
    bn_14 = build_bn_14_nodes()
    ie = gum.LazyPropagation(bn_14)
    ie.makeInference()
    probabilities = {f'N{i + 1}': ie.posterior(f'N{i + 1}')[0] for i in range(14)}
    gnb.showBN(bn_14)
    gnb.showInference(bn_14, evs={}, engine=ie)
    gimg.export(bn_14, 'static/images/graph_14_nodes.png')
    gimg.exportInference(bn_14, 'static/images/inference_14_nodes.png', evs={})
    return probabilities, bn_14
