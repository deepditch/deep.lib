import torch
import util
from session import *
from tqdm.notebook import tqdm
from callbacks import *
    
class AdvertorchCallback(TrainCallback):
  def __init__(self, dataloader, adversary, interval=1, model_file=None):
    super().__init__()
    self.dataloader = dataloader
    self.adversary = adversary
    self.interval = interval
    self.num_epochs = 0
    self.best = 0
    self.model_file = model_file

  def state_dict(self): return {}
  def load_state_dict(self, state_dict): pass

  def run(self, session, log=True):
    correct = 0
    total = 0

    for data, label in tqdm(self.dataloader, leave=False):
        data, label = data.to("cuda"), label.to("cuda")
        perturbed_data = self.adversary.perturb(data, label)
        total += len(label)
        
        # Re-classify the perturbed image
        output = model(perturbed_data)
        if isinstance(output, list): output = output[-1]  
        _, final_pred = output.max(1)
        correct += torch.sum(final_pred == label)

    acc = correct/float(total)
    if log: print("{} Accuracy = {} / {} = {:.4f}".format(self.adversary.__class__.__name__, correct, total, acc*100))
    return acc

  def on_epoch_end(self, session, lossMeter):
    self.num_epochs += 1
    if self.num_epochs % self.interval != 0: return

    if lossMeter is not None:
        print(f"Training Loss: {lossMeter.debias}")
     
    accuracy = self.run(session)

    if self.model_file != None:
        if accuracy > self.best:
            self.best = accuracy
            session.save(self.model_file)

    session.add_meta(f"{self.adversary.__class__.__name__} Epoch {self.num_epochs}", str(accuracy))
    
