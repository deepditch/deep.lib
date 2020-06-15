import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from validation import _AccuracyMeter, OneHotAccuracy
from session import LossMeter, EvalModel
from callbacks import *
import util
from Loss.triplet import *
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
import pickle


class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self._to_select = to_select
    
    def forward(self, x):
        list = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                list.append((x, name))
        return list


class CustomOneHotAccuracy(OneHotAccuracy):
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, output, label):
        return super().update(output[-1], label)


class TripletRegularizedLossValidator(StatelessTrainCallback):
    def __init__(self, val_data, select, loss_fn=F.multi_margin_loss):
        super().__init__()

        self.val_data = val_data

        self.train_loss = LossMeter()
        self.train_raw_loss = LossMeter()
        self.train_accuracy = CustomOneHotAccuracy()

        self.valid_loss = LossMeter()
        self.valid_raw_loss = LossMeter()
        self.valid_accuracy = CustomOneHotAccuracy()

        self.loss_fn = loss_fn

    def run(self, session):         
        with EvalModel(session.model) and torch.no_grad():
            for input, label, *_ in tqdm(self.val_data, desc="Validating", leave=False):
                label = Variable(util.to_gpu(label))
                output = session.forward(input)   

                valid_loss.update(session.criterion(output, label).data.cpu(), input.shape[0])
                valid_raw_loss.update(self.loss_fn(output[-1], label).data.cpu(), input.shape[0])
                
                self.valid_accuracy.update(output, label)       
        
    def on_epoch_begin(self, session, schedule, cb_dict, *args, **args):
        self.train_loss.reset()
        self.train_raw_loss.reset()
        self.train_accuracy.reset()

        self.valid_loss.reset()
        self.valid_raw_loss.reset()
        self.valid_accuracy.reset() 

    def on_batch_end(self, session, schedule, cb_dict, loss, *args, **args):
        label = Variable(util.to_gpu(label))
        
        self.train_accuracy_meter.update(output, label)
        self.train_loss_meter.update(loss)
        self.train_raw_loss_meter.update(self.loss_fn(output[-1], label).data.cpu(), label.shape[0])
        
    def on_epoch_end(self, session, schedule, cb_dict, *args, **args):        
        self.run(session) 
        self.num_epochs += 1
      
        train_loss = self.train_loss.raw_avg
        train_raw_loss = self.train_raw_loss.raw_avg
        train_triplet_loss = train_loss - train_raw_loss

        valid_loss = self.valid_loss.raw_avg
        valid_raw_loss = self.valid_raw_loss.raw_avg
        valid_triplet_loss = valid_loss - valid_raw_loss

        cb_dict["Loss/Train"] = train_loss
        cb_dict["Unregularized Loss/Train"] = train_raw_loss
        cb_dict["Triplet Loss/Train"] = train_triplet_loss

        cb_dict["Loss/Valid"] = Valid_loss
        cb_dict["Unregularized Loss/Valid"] = Valid_raw_loss
        cb_dict["Triplet Loss/Valid"] = Valid_triplet_loss

    def register_metrics(self):
        return ["Loss/Train", "Unregularized Loss/Train", "Triplet Loss/Train", "Loss/Valid", "Unregularized Loss/Valid", "Triplet Loss/Valid"]


def tensorboard_embeddings(model, select, dataloader, targets, images, board='./runs'):
    old_select = model._to_select
    model._to_select = select
    writer = SummaryWriter(board)
    
    outputs = {name: [] for name in select}
    
    with EvalModel(model):
        for input, label in dataloader:
            output = model.forward(Variable(util.to_gpu(input)))
            for layer in output:    
                outputs[layer[1]].append(layer[0].data.cpu().view(layer[0].size(0), -1))    
                
    for name, output in outputs.items():
        cat = torch.cat(output)
        writer.add_embedding(cat, tag=name, metadata=targets, label_img=images)

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy import stats as s

def compute_embeddings(model, dataloader, max_num):
  outputs = []
  labels = []

  num = 0

  with EvalModel(model) and torch.no_grad():
      for input, label in dataloader:
          output = model.forward(Variable(util.to_gpu(input)))
          if type(output) is list:
              output = output[0]
          outputs.append(output.data.cpu().view(output.size(0), -1)) 
          labels.append(label)   
          num += label.shape[0]

          if num > max_num: break
              
  cat = torch.cat(outputs).numpy()
  labels = torch.cat(labels).numpy()

  return cat, labels

class EmbeddingSpaceVisualizer(StatelessTrainCallback):
  def __init__(self, dataloader, interval=1):
    super().__init__()
    self.dataloader = dataloader
    self.interval = interval
    self.num_epochs = 0

  def on_epoch_begin(self, session, *args, **kwargs):
    self.num_epochs += 1
    if self.num_epochs % self.interval != 0: return
     
    num_to_plot = 200
    embedding, y = compute_embeddings(session.model, self.dataloader, num_to_plot)
    pca = PCA(n_components=2, whiten=True)
    pca.fit(embedding)
    X_pca = pca.transform(embedding)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.show()