import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm_notebook as tqdm, tnrange
import session as sess
from torch.autograd import Variable


class TrainCallback:
    def on_train_begin(self, session): pass
    def on_epoch_begin(self, session): pass
    def on_batch_begin(self, session): pass
    def on_batch_end(self, session, lossMeter): pass
    def on_epoch_end(self, session, lossMeter): pass
    def on_train_end(self, session): pass


class _HitMetric:
    def __call__(self, output, label):
        raise NotImplementedError


class OneHotAccuracy(_HitMetric):
    def __call__(self, output, label):
        _, preds = torch.max(output, 1)
        return torch.sum(preds == label)


class Validator(TrainCallback):
    def __init__(self, val_data, hit_metric=None):
        self.val_data = val_data
        self.hit_metric = hit_metric

    def on_epoch_end(self, session, lossMeter): 
        num_correct = 0
        valLoss = sess.LossMeter()
        with sess.EvalModel(session.model):
            for input, label in tqdm(self.val_data, desc="Validating", leave=False):
                input = Variable(sess.to_gpu(input))
                label = Variable(sess.to_gpu(label)).long()
                output = session.model(input)
                step_loss = session.criterion(output, label).data.tolist()[0]
                valLoss.update(step_loss, label.shape[0])
                if self.hit_metric is not None:        
                    num_correct += self.hit_metric(output, label).data.tolist()[0]
        
        val_accuracy = num_correct/valLoss.count
        print("Training Loss: %f  Validaton Loss: %f Validation Accuracy: %f" % (lossMeter.debias, valLoss.raw_avg, val_accuracy))


class Saver(TrainCallback):
    def __init__(self, dir):
        self.dir = dir

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)


        self.best = 0
        self.epoch = 0

    def on_epoch_end(self, session, lossMeter):
        self.epoch += 1
        session.save('model.%d' % self.epoch)




