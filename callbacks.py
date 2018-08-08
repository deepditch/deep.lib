import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm, tnrange
import session as sess
from torch.autograd import Variable


class TrainCallback:
    def on_train_begin(self, session): pass
    def on_epoch_begin(self, session): pass
    def on_batch_begin(self, session): pass
    def on_batch_end(self, session, loss): pass
    def on_epoch_end(self, session, loss): pass
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

    def on_epoch_end(self, session, loss):
        val_loss = 0.0
        num_correct = 0
        total_examples = 0
        with sess.EvalModel(session.model):
            for input, label in tqdm(self.val_data, desc="Validating", leave=False):
                input = Variable(sess.to_gpu(input))
                label = Variable(sess.to_gpu(label))
                output = session.model(input)
                val_loss += session.criterion(output, label).data.tolist()[0]
                if self.hit_metric is not None:
                    total_examples += label.shape[0]
                    num_correct += self.hit_metric(output, label).data.tolist()[0]
        
        val_loss = val_loss/len(self.val_data)
        val_accuracy = num_correct/total_examples
        print("Training Loss: %f  Validaton Loss: %f Validation Accuracy: %f" % (loss, val_loss, val_accuracy))



