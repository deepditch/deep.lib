import torch
import torch.nn as nn
import torch.optim as optim

class TrainCallback:
    def on_train_begin(self, session): pass
    def on_batch_begin(self, session): pass
    def on_phase_begin(self, session): pass
    def on_epoch_end(self, session): pass
    def on_phase_end(self, session): pass
    def on_batch_end(self, session): pass
    def on_train_end(self, session): pass

class Validator(TrainCallback):
    def __init__(valDataLoader):
        self.valDataLoader = valDataLoader

    def on_epoch_end(self, session):
        with torch.set_grad_enabled(False):
            for input, label in valDataLoader:
                outputs = session.model.forward(inputs)
                loss = session.criterion(label, outputs)
                print("Val Loss: %s" % loss)



