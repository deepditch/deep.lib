import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm_notebook as tqdm, tnrange
import session as sess
from torch.autograd import Variable
import numpy as np
from callbacks import TrainCallback
import util

class _AccuracyMeter:
    def accuracy(self): raise NotImplementedError
    def update(self, output, label): raise NotImplementedError
    def reset(self, output, label): raise NotImplementedError


class OneHotAccuracy(_AccuracyMeter):
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_correct = 0
        self.count = 0
        
    def accuracy(self): 
        return self.num_correct / self.count

    def update(self, output, label):
        _, preds = torch.max(output, 1)
        self.num_correct += torch.sum(preds == label)
        self.count += label.shape[0]


class NHotAccuracy(_AccuracyMeter):
    def __init__(self, num_classes):        
        self.num_classes = num_classes
        self.num_correct = 0
        self.reset()

    def reset(self):
        self.num_correct = 0
        self.count = 0
        self.details = [{"correct_pos":0, "correct_neg":0, "false_pos":0, "false_neg":0} for i in range(self.num_classes)]
    
    def accuracy(self): 
        return self.num_correct / self.count

    def update_from_numpy(self, preds, labels):
        self.count += labels.shape[0] * self.num_classes
        self.num_correct += np.sum(preds == labels)
        for pred, label, detail in zip(zip(*preds), zip(*labels), self.details):
            detail["correct_pos"] += np.sum([p and l for p, l in zip(pred, label)])
            detail["correct_neg"] += np.sum([not p and not l for p, l in zip(pred, label)])
            detail["false_pos"] += np.sum([p and not l for p, l in zip(pred, label)])
            detail["false_neg"] += np.sum([not p and l for p, l in zip(pred, label)])

    def update(self, outputs, labels):
        preds = torch.clamp(torch.round(util.to_cpu(outputs).data), 0, 1).numpy().astype(int)
        labels = util.to_cpu(labels).data.numpy().astype(int)

        self.update_from_numpy(preds, labels)       


class Validator(TrainCallback):
    def __init__(self, val_data, accuracy_meter=None):
        self.val_data = val_data
        self.accuracy_meter = accuracy_meter

    def run(self, session, lossMeter=None):
        self.accuracy_meter.reset()
        valLoss = sess.LossMeter()
        with sess.EvalModel(session.model):
            for input, label, *_ in tqdm(self.val_data, desc="Validating", leave=True):
                label = Variable(util.to_gpu(label))
                output = session.forward(input)
                step_loss = session.criterion(output, label).data.tolist()[0]
                valLoss.update(step_loss, label.shape[0])
                if self.accuracy_meter is not None:        
                    self.accuracy_meter.update(output, label)
        
        val_accuracy = self.accuracy_meter.accuracy() if self.accuracy_meter is not None else 0

        if lossMeter is not None:
            print("Training Loss: %f  Validaton Loss: %f Validation Accuracy: %f" % (lossMeter.debias, valLoss.raw_avg, val_accuracy))
        else:
            print("Validaton Loss: %f Validation Accuracy: %f" % (valLoss.raw_avg, val_accuracy))
          

    def on_epoch_end(self, session, lossMeter): 
        self.run(session, lossMeter)
