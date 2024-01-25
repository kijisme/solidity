import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def score(labels, logits):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def get_classification_report(labels, logits, output_dict=False):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    return classification_report(labels, prediction, digits=4, output_dict=output_dict)

def get_confusion_matrix(labels, logits):
    logits = nn.functional.softmax(logits, dim=1)
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()  
    return confusion_matrix(labels, prediction)