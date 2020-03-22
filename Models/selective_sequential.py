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
from tqdm import tqdm_notebook as tqdm
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
        return super().update(output[-1][0], label)

class EmbeddingSpaceValidator(TrainCallback):
    def __init__(self, val_data, select, accuracy_meter_fn, loss_fn=F.multi_margin_loss, model_file=None, tensorboard_dir=None):
        super().__init__()
        self.val_data = val_data
        self.val_accuracy_meter = accuracy_meter_fn()
        self.train_accuracy_meter = accuracy_meter_fn()
        self.select=select
        self.names=["" for x in range(len(self.select) - 1)]
        
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.train_losses = []
        self.train_raw_losses = []

        self.val_losses = []
        self.val_raw_losses = []
        
        self.train_embedding_losses = [[] for x in range(len(self.select) - 1)]
        
        self.train_embedding_loss_meters = [LossMeter() for x in range(len(self.select) - 1)]
        self.val_embedding_losses = [[] for x in range(len(self.select) - 1)]
        
        self.num_batches = 0
        self.num_epochs = 0

        self.model_file = model_file

        self.best_accuracy = 0

        self.writer = SummaryWriter(log_dir=tensorboard_dir) if tensorboard_dir is not None else None    

        self.loss_fn = loss_fn

    def state_dict(self):
        return pickle.dumps({k: self.__dict__[k] for k in set(['num_batches', 'num_epochs', 'model_file', 'best_accuracy'])})

    def run(self, session, lossMeter=None):
        self.val_accuracy_meter.reset()
            
        val_loss = LossMeter()
        val_raw_loss = LossMeter()
        embedding_losses = [LossMeter() for x in range(len(self.select) - 1)]
        
        with EvalModel(session.model):
            for input, label, *_ in tqdm(self.val_data, desc="Validating", leave=False):
                label = Variable(util.to_gpu(label))
                output = session.forward(input)
                
                step_loss = session.criterion(output, label).data.cpu()
          
                val_loss.update(step_loss, input.shape[0])
                
                val_raw_loss.update(self.loss_fn(output[-1][0], label).data.cpu(), input.shape[0])
                
                self.val_accuracy_meter.update(output, label)

                for idx, (layer, embedding_loss) in enumerate(zip(output[:-1], embedding_losses)):           
                    if layer[1] in self.select:
                        self.names[idx] = layer[1]
                        embedding_loss.update(batch_all_triplet_loss(layer[0].view(layer[0].size(0), -1), label, 1).data.cpu())
        
        self.val_losses.append(val_loss.raw_avg.item())
        self.val_raw_losses.append(val_raw_loss.raw_avg.item())
         
        accuracy = self.val_accuracy_meter.accuracy()

        if self.model_file != None and accuracy > self.best_accuracy:
            session.save(self.model_file)
            self.best_accuracy = accuracy
        
        self.val_accuracies.append(accuracy)
              
        for meter, loss in zip(embedding_losses, self.val_embedding_losses):
            loss.append(meter.raw_avg)     
        
    def on_epoch_begin(self, session):
        self.train_accuracy_meter.reset()     
        self.train_raw_loss_meter = LossMeter()
        
    def on_epoch_end(self, session, lossMeter): 
        self.train_accuracies.append(self.train_accuracy_meter.accuracy())
        self.train_losses.append(lossMeter.debias.data.cpu().item())
        
        self.train_raw_losses.append(self.train_raw_loss_meter.raw_avg.data.cpu().item())
        
        self.run(session, lossMeter) 
        self.num_epochs += 1
        
        for meter, loss in zip(self.train_embedding_loss_meters, self.train_embedding_losses):
            loss.append(meter.raw_avg)
            meter.reset()

        
        train_loss = self.train_losses[-1]
        train_raw_loss = self.train_raw_losses[-1]
        train_triplet_loss = train_loss - train_raw_loss

        val_loss = self.val_losses[-1]
        val_raw_loss = self.val_raw_losses[-1]
        val_triplet_loss = val_loss - val_raw_loss
        
        print("\nval accuracy: ", round(self.val_accuracies[-1], 4),
              "train accuracy: ", round(self.train_accuracies[-1], 4),
              "\ntrain loss: ", round(train_loss, 4), 
              " train unreg loss : ", round(train_raw_loss, 4),     
              " train triplet loss : ", round(train_triplet_loss, 4),   
              "\nvalid loss: ", round(val_loss, 4), 
              " valid unreg loss : ", round(val_raw_loss, 4),
              " valid triplet loss : ", round(val_triplet_loss, 4)
              )

        if self.writer is not None:
            self.writer.add_scalars('Loss/Regularized', {'Train':self.train_losses[-1],
                                                        'Test':self.val_losses[-1]}, self.num_batches)

            self.writer.add_scalars('Loss/Unregularized', {'Train':self.train_raw_losses[-1],
                                                            'Test':self.val_raw_losses[-1]}, self.num_batches)

            self.writer.add_scalars('Accuracy', {'Train':self.train_accuracies[-1],
                                            'Test':self.val_accuracies[-1]}, self.num_batches)
    
    def on_batch_end(self, session, lossMeter, output, label):
        label = Variable(util.to_gpu(label))
        self.train_accuracy_meter.update(output, label)
        self.train_raw_loss_meter.update(self.loss_fn(output[-1][0], label).data.cpu(), label.shape[0])
             
        for layer, loss_meter in zip(output[:-1], self.train_embedding_loss_meters):
            if layer[1] in self.select:
                loss_meter.update(batch_all_triplet_loss(layer[0].view(layer[0].size(0), -1), label, 1).data.cpu().item())
            
        self.num_batches += 1

        if self.writer is not None:
            self.writer.add_scalars('Loss/Training Batch', {
                'Regularized': lossMeter.loss,
                'Unregularized': self.train_raw_loss_meter.loss
            }, self.num_batches)

            
    def plot(self, title="", file=None):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))

        fig.suptitle(f"{title} : Best Accuracy {np.max(self.val_accuracies)}", fontsize=14)
            
        ax1.set_title(f"Accuracy per Epoch")
        ax1.plot(np.arange(0, self.num_epochs), self.train_accuracies, label="Training")
        ax1.plot(np.arange(0, self.num_epochs), self.val_accuracies, label="Validation")

        ax2.set_title(f"Regularizezd Loss per Epoch")
        ax2.plot(np.arange(0, self.num_epochs), self.train_losses, label="Training")
        ax2.plot(np.arange(0, self.num_epochs), self.val_losses, label="Validation")

        ax3.set_title(f"Unregularizezd Loss per Epoch")
        ax3.plot(np.arange(0, self.num_epochs), self.train_raw_losses, label="Training")   
        ax3.plot(np.arange(0, self.num_epochs), self.val_raw_losses, label="Validation")
        
        ax4.set_title("Triplet Loss per Epoch")
        for embedding, name in zip(self.train_embedding_losses, self.names):
            ax4.plot(np.arange(0, self.num_epochs), embedding, label=f"Training: {name}")
        
        for embedding, name in zip(self.val_embedding_losses, self.names):
            ax4.plot(np.arange(0, self.num_epochs), embedding, label=f"Validation: {name}")
            
        for ax in (ax1, ax2, ax3, ax4):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))   

        plt.show()

        if file is not None: fig.savefig(file)


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
