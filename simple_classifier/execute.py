from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
import pandas as pd
from config.cfg import cfg
import plotly.graph_objects as go

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']

basketball_player = True  # 'баскетболист'
football_player = False  # 'футболист'

threshold_values = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

accuracy_values = []
f1_score_values = []
recall_values = []
precision_values = []

for threshold in threshold_values:
    predicted_classes = np.where(predictions >= threshold, basketball_player, football_player)
    TP = np.sum((predicted_classes == basketball_player) & (gt == basketball_player))
    FP = np.sum((predicted_classes == basketball_player) & (gt == football_player))
    FN = np.sum((predicted_classes == football_player) & (gt == basketball_player))
    TN = np.sum((predicted_classes == football_player) & (gt == football_player))
    print([TP, FP, FN, TN])
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy_values.append(accuracy)
    f1_score_values.append(f1_score)
    recall_values.append(recall)
    precision_values.append(precision)

hover_data = [f'Accuracy: {acc}<br>F1 Score: {f1}<br>Threshold: {thres}'
              for acc, f1, thres in zip(accuracy_values, f1_score_values, threshold_values)]
fig = go.Figure(data=go.Scatter(x=recall_values, y=precision_values, mode='lines+markers', hovertemplate=hover_data))

fig.update_layout(title='PR Curve', xaxis_title='Recall', yaxis_title='Precision')

fig.show()
