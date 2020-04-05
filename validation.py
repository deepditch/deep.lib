import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm_notebook as tqdm
import session as sess
from torch.autograd import Variable
import numpy as np
from callbacks import TrainCallback
import util
import time
import pathlib

class _AccuracyMeter:
    def update(self, output, label): raise NotImplementedError
    def reset(self, output, label): raise NotImplementedError
    def metric(self): raise NotImplementedError
    def report(self): print(self.metric())

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
        self.num_false_positives = 0
        self.num_false_negatives = 0
        self.eps = threshold
        self.reset()

    def reset(self):
        self.confusion = [{"true_pos":0, "true_neg":0, "false_pos":0, "false_neg":0, "support": 0} for i in range(self.num_classes)]
    
    def metric(self): 
        precision = self.num_true_positives / (self.num_true_positives + self.num_false_positives)
        recall = self.num_true_positives / (self.num_true_positives + self.num_false_negatives)

        return 2 * (precision * recall) / (precision + recall)

    def precision(self):
        precision = []
        for conf in self.confusion:
            if(conf["true_pos"] == 0):
                precision.append(0)
            else:
                precision.append(conf["true_pos"] / (conf["true_pos"] + conf["false_pos"]))

        return precision

    def recall(self):
        recall = []
        for conf in self.confusion:
            if(conf["true_pos"] == 0):
                recall.append(0)
            else:
                recall.append(conf["true_pos"] / (conf["true_pos"] + conf["false_neg"]))

        return recall

    def FMeasure(self):
        f = []
        precision = self.precision()
        recall = self.recall()
        for r, p in zip(recall, precision):
            if r == 0 or p == 0:
                f.append(0)
            else:
                f.append(2 * (p * r) / (p + r))

        return f

    def update_from_numpy(self, preds, labels):
        for pred, label, detail in zip(zip(*preds), zip(*labels), self.confusion):
            true_pos = np.sum([p and l for p, l in zip(pred, label)])
            true_neg = np.sum([not p and not l for p, l in zip(pred, label)])
            false_pos = np.sum([p and not l for p, l in zip(pred, label)])
            false_neg = np.sum([not p and l for p, l in zip(pred, label)])
            
            self.num_true_positives += true_pos
            self.num_false_positives += false_pos
            self.num_false_negatives += false_neg

            detail["true_pos"] += true_pos
            detail["true_neg"] += true_neg
            detail["false_pos"] += false_pos
            detail["false_neg"] += false_neg
            detail["support"] += true_pos + false_neg

    def update(self, outputs, labels):
        preds = util.to_cpu(torch.where(outputs > self.eps, 1, 0)).data.numpy().astype(int)
        labels = util.to_cpu(labels).data.numpy().astype(int)

        self.update_from_numpy(preds, labels)       


class Validator(TrainCallback):
    def __init__(self, val_data, accuracy_meter=None, save_best=False, model_dir='./'):
        self.val_data = val_data
        self.accuracy_meter = accuracy_meter
        self.best_metric = 0
        self.save_best = save_best
        self.batch = 0
        self.model_dir = model_dir
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    def state_dict(self): return ""
    def load_state_dict(self, dict): pass

    def run(self, session, lossMeter=None):
        self.batch += 1
        if self.accuracy_meter is not None: self.accuracy_meter.reset()
        valLoss = sess.LossMeter()
        with sess.EvalModel(session.model):
            for input, label, *_ in tqdm(self.val_data, desc="Validating", leave=False):
                if isinstance(label, dict):
                    label = {key: Variable(value) for key, value in label.items()}  
                else:
                    label = Variable(util.to_gpu(label))
                output = session.forward(input)
                step_loss = session.criterion(output, label).data
                valLoss.update(step_loss, input.shape[0])
                if self.accuracy_meter is not None:        
                    self.accuracy_meter.update(output, label)
        
        metric = self.accuracy_meter.metric() if self.accuracy_meter is not None else 0
        
        if self.save_best and metric > self.best_metric:
            self.best_metric = metric
            session.add_meta("Best Accuracy", self.best_metric)
            session.save(f'{self.model_dir}/best-{self.batch}-{round(self.best_metric.item(), 6)}')

        if lossMeter is not None:
            print(f"Training Loss: {lossMeter.debias} Validaton Loss: {valLoss.raw_avg}")
        else:
            print(f"Validaton Loss: {valLoss.raw_avg}")

        if self.accuracy_meter is not None:
            self.accuracy_meter.report()
          

    def on_epoch_end(self, session, lossMeter): 
        self.run(session, lossMeter)
