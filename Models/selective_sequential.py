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
    def __init__(self, val_data, num_embeddings, accuracy_meter_fn, model_file=None):
        self.val_data = val_data
        self.val_accuracy_meter = accuracy_meter_fn()
        self.train_accuracy_meter = accuracy_meter_fn()
        self.num_embeddings=num_embeddings
        self.names=["" for x in range(self.num_embeddings)]
        
        self.train_accuracies = []
        self.batch_train_accuracies = []
        self.val_accuracies = []
        
        self.train_losses = []
        self.batch_train_losses = []
        self.train_bce_losses = []
        self.val_losses = []
        self.val_bce_losses = []
        
        self.train_embedding_losses = [[] for x in range(self.num_embeddings)]
        
        self.train_embedding_loss_meters = [LossMeter() for x in range(self.num_embeddings)]
        
        self.val_embedding_losses = [[] for x in range(self.num_embeddings)]
        
        self.num_batches = 0
        self.num_epochs = 0
        
        self.epochs = []

        self.model_file = model_file

        self.best_accuracy = 0

    def run(self, session, lossMeter=None):
        self.val_accuracy_meter.reset()
            
        val_loss = LossMeter()
        val_bce_loss = LossMeter()
        embedding_losses = [LossMeter() for x in range(self.num_embeddings)]
        
        with EvalModel(session.model):
            for input, label, *_ in tqdm(self.val_data, desc="Validating", leave=True):
                label = Variable(util.to_gpu(label))
                output = session.forward(input)
                
                step_loss = session.criterion(output, label).data.cpu()
                
                val_loss.update(step_loss, input.shape[0])
                
                val_bce_loss.update(F.cross_entropy(output[-1][0], label).data.cpu(), input.shape[0])
                
                self.val_accuracy_meter.update(output, label)

                for idx, (layer, embedding_loss) in enumerate(zip(output[:-1], embedding_losses)):
                    self.names[idx] = layer[1]
                    embedding_loss.update(batch_all_triplet_loss(layer[0].view(layer[0].size(0), -1), label, 1).data.cpu())
        
        self.val_losses.append(val_loss.raw_avg.item())
        self.val_bce_losses.append(val_bce_loss.raw_avg.item())
         
        accuracy = self.val_accuracy_meter.accuracy()

        if self.model_file != None and accuracy > self.best_accuracy:
            session.save(self.model_file)
            self.base_accuracy = accuracy
        
        self.val_accuracies.append(accuracy)
              
        for meter, loss in zip(embedding_losses, self.val_embedding_losses):
            loss.append(meter.raw_avg)     
        
    def on_epoch_begin(self, session):
        self.train_accuracy_meter.reset()     
        self.train_bce_loss_meter = LossMeter()
        
    def on_epoch_end(self, session, lossMeter): 
        self.train_accuracies.append(self.train_accuracy_meter.accuracy())
        self.train_losses.append(lossMeter.debias.data.cpu().item())
        
        self.train_bce_losses.append(self.train_bce_loss_meter.raw_avg.data.cpu().item())
        
        self.run(session, lossMeter) 
        self.epochs.append(self.num_batches)
        self.num_epochs += 1
        
        for meter, loss in zip(self.train_embedding_loss_meters, self.train_embedding_losses)
            loss.append(meter.raw_avg)
            meter.reset()
        
        print("\nval accuracy: ", round(self.val_accuracies[-1], 4),
              "\ntrain loss: ", round(self.train_losses[-1], 4) , 
              " train BCE : ", round(self.train_bce_losses[-1], 4) ,       
              "\nvalid loss: ", round(self.val_losses[-1], 4), 
              " valid BCE : ", round(self.val_bce_losses[-1], 4))
    
    def on_batch_end(self, session, lossMeter, output, label):
        label = Variable(util.to_gpu(label))
        batch_accuracy = self.train_accuracy_meter.update(output, label)
        self.batch_train_accuracies.append(batch_accuracy)
        self.batch_train_losses.append(lossMeter.loss.data.cpu().item())   
        self.train_bce_loss_meter.update(F.cross_entropy(output[-1][0], label).data.cpu(), label.shape[0])
             
        for layer, loss_meter in zip(output[:-1], self.train_embedding_loss_meters):
            loss_meter.update(batch_all_triplet_loss(layer[0].view(layer[0].size(0), -1), label, 1).data.cpu().item())
            
        self.num_batches += 1
            
    def plot(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))
        
        #ax.plot(np.arange(self.num_batches), self.batch_train_accuracies)
        #legend.append("Train accuracy per batch")
        
        #ax.plot(np.arange(self.num_batches), self.batch_train_losses)
        #legend.append("Train loss per batch")
            
        ax1.plot(self.epochs, self.train_accuracies, '-o', label="Training accuracy per epoch")

        ax1.plot(self.epochs, self.val_accuracies, '-o', label="Validation accuracy per epoch")
        
        ax2.plot(self.epochs, self.train_losses, '-o', label="Training loss per epoch")
        
        ax2.plot(self.epochs, self.val_losses, '-o', label="Validation loss per epoch")
        
        ax3.plot(self.epochs, self.train_bce_losses, '-o', label="Training BCE loss per epoch")
        
        ax3.plot(self.epochs, self.val_bce_losses, '-o', label="Validation BCE loss per epoch")
        
        for embedding, name in zip(self.train_embedding_losses, self.names):
            ax4.plot(self.epochs, embedding, label=f"Train {name} embedding triplet loss")
        
        for embedding, name in zip(self.val_embedding_losses, self.names):
            ax4.plot(self.epochs, embedding, '-o', label=f"Validation {name} embedding triplet loss")
            
        for ax in (ax1, ax2, ax3, ax4):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))   

        plt.show()


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
