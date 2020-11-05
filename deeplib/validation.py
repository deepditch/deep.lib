import os
import time
import pathlib
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import label_ranking_loss, coverage_error, label_ranking_average_precision_score

from deeplib.session import *
from deeplib.callbacks import TrainCallback
import deeplib.util as util


class _AccuracyMeter:
    def update(self, output, label): raise NotImplementedError
    def reset(self, output, label): raise NotImplementedError
    def metric(self): raise NotImplementedError


class OneHotAccuracy(_AccuracyMeter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_correct = 0
        self.count = 0
        
    def accuracy(self): 
        return (self.num_correct.double() / float(self.count)).item()

    def metric(self): return self.accuracy()

    def report(self): print(f"Validation Accuracy: {round(self.metric, 4)}")

    def update(self, output, label):
        _, preds = torch.max(output, 1)
        batch_correct = util.to_cpu(torch.sum(preds == label).data)
        self.num_correct += batch_correct
        self.count += label.shape[0]
        return (batch_correct.double() / label.shape[0]).item() 


class NHotAccuracy(_AccuracyMeter):
    def __init__(self, num_classes, threshold=.5):        
        self.num_classes = num_classes
        self.num_true_positives = 0
        self.num_true_negatives = 0
        self.num_false_positives = 0
        self.num_false_negatives = 0
        self.eps = threshold
        self.reset()

    def metric(self): return self.FMeasure()[1] # Macro average FMeasure

    def reset(self):
        self.n = 0    
        self.one_error = 0
        self.coverage = 0
        self.ranking_loss = 0    
        self.average_precision = 0

        self.confusion = [{"true_pos":0, "true_neg":0, "false_pos":0, "false_neg":0, "support": 0} for i in range(self.num_classes)]

    def report(self):
      acc = self.accuracy()
      prec = self.precision()
      rec = self.recall()
      FM = self.FMeasure()

      df = pd.DataFrame(np.array([acc, prec, rec, FM]).T, ["Micro avg.", "Macro avg.", "Weighted avg."], ["Accuracy", "Precision", "Recall", "FMeasure"])

      print(df)

      print("")

      RL = self.RankingLoss()
      AP = self.AveragePrecision()
      O = self.OneError()
      C = self.Coverage()

      df = pd.DataFrame([(O, C, RL, AP)], columns=["One-Error", "Coverage", "Ranking Loss", "Average Precision"])

      print(df.to_string(index=False))
    
    def accuracy(self): 
        micro = (self.num_true_positives + self.num_true_negatives) / (self.num_true_positives + self.num_true_negatives + self.num_false_positives + self.num_false_negatives)
        accuracies = [(cls["true_pos"] + cls["true_neg"]) / (cls["true_pos"] + cls["true_neg"] + cls["false_pos"] + cls["false_neg"]) for cls in self.confusion]
        macro = np.average(accuracies)
        weighted = np.average(accuracies, weights=[cls["support"] for cls in self.confusion])

        return micro, macro, weighted

    def precision(self):
        micro = self.num_true_positives / (self.num_true_positives + self.num_false_positives)
        precisions = [(cls["true_pos"]) / (cls["true_pos"] + cls["false_pos"]) if cls["true_pos"] + cls["false_pos"] else 1 for cls in self.confusion]
        macro = np.average(precisions)
        weighted = np.average(precisions, weights=[cls["support"] for cls in self.confusion])

        return micro, macro, weighted

    def recall(self):
        micro = self.num_true_positives / (self.num_true_positives + self.num_false_negatives)  
        recalls = [(cls["true_pos"]) / (cls["true_pos"] + cls["false_neg"]) for cls in self.confusion if cls["support"] > 0]
        macro = np.average(recalls)
        weighted = np.average(recalls, weights=[cls["support"] for cls in self.confusion if cls["support"] > 0])

        return micro, macro, weighted

    def FMeasure(self):
        micro_p, macro_p, weighted_p = self.precision()
        micro_r, macro_r, weighted_r = self.recall()

        micro = 2 * (micro_p * micro_r) / (micro_p + micro_r)
        macro = 2 * (macro_p * macro_r) / (macro_p + macro_r)
        weighted = 2 * (weighted_p * weighted_r) / (weighted_p + weighted_r)

        return micro, macro, weighted

    def RankingLoss(self): return self.ranking_loss / self.n
    def AveragePrecision(self): return self.average_precision / self.n
    def Coverage(self): return self.coverage / self.n
    def OneError(self): return self.one_error / self.n

    def update_from_numpy(self, preds, labels):
        for pred, label, cls in zip(zip(*preds), zip(*labels), self.confusion):
            true_pos = np.sum([p and l for p, l in zip(pred, label)])
            true_neg = np.sum([not p and not l for p, l in zip(pred, label)])
            false_pos = np.sum([p and not l for p, l in zip(pred, label)])
            false_neg = np.sum([not p and l for p, l in zip(pred, label)])
            
            self.num_true_positives += true_pos
            self.num_true_positives += true_neg
            self.num_false_positives += false_pos
            self.num_false_negatives += false_neg

            cls["true_pos"] += true_pos
            cls["true_neg"] += true_neg
            cls["false_pos"] += false_pos
            cls["false_neg"] += false_neg
            cls["support"] += true_pos + false_neg

        n = len(preds)

        self.n += n
        self.ranking_loss += label_ranking_loss(labels, preds) * n
        self.coverage += coverage_error(labels, preds) * n
        self.average_precision += label_ranking_average_precision_score(labels, preds) * n

        for pred, label in zip(preds, labels):
            lowest_rank_prediction = np.argsort(pred)[-1]
            label = np.argwhere(label)

            if lowest_rank_prediction not in label: 
                self.one_error += 1

    def update(self, outputs, labels):
        outputs = outputs.view(-1, self.num_classes)
        labels = labels.view(-1, self.num_classes)

        mask = labels.any(dim=1)

        outputs = outputs[mask]
        labels = labels[mask]

        preds = util.to_cpu(outputs > self.eps).data.numpy().astype(int)
        labels = util.to_cpu(labels).data.numpy().astype(int)

        self.update_from_numpy(preds, labels)      


class Validator(TrainCallback):
    def __init__(self, dataloader, accuracy_meter=None, metric_name="Accuracy/Validation"):
        self.dataloader = dataloader
        self.accuracy_meter = accuracy_meter
        self.metric_name = metric_name

    def register_metric(self):
        return [self.metric_name, "Loss/Validation"] if self.accuracy_meter is not None else "Loss/Validation"

    def state_dict(self): return ""
    def load_state_dict(self, dict): pass

    def run(self, session, cb_dict):
        if self.accuracy_meter is not None: self.accuracy_meter.reset()
        valLoss = LossMeter()
        with EvalModel(session.model) and torch.no_grad():
            for input, label, *_ in tqdm(self.dataloader, desc="Validating", leave=False):

                if isinstance(label, dict):
                    label = {key: Variable(value) for key, value in label.items()}  
                else:
                    label = Variable(util.to_gpu(label))

                output, input = session.forward(input)
                step_loss = session.criterion(output, label).data
                valLoss.update(step_loss, label.shape[0])
                if self.accuracy_meter is not None:        
                    self.accuracy_meter.update(output, label)
        
        cb_dict["Loss/Validation"] = valLoss.raw_avg
        if self.accuracy_meter is not None: cb_dict[self.metric_name] = self.accuracy_meter.metric()         

    def on_epoch_end(self, session, schedule, cb_dict, *args, **kwargs): 
        self.run(session, cb_dict)
