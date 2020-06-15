import torch
import util
from session import *
from tqdm.notebook import tqdm
from callbacks import *
    
class AdvertorchCallback(StatelessTrainCallback):
  def __init__(self, dataloader, adversary, metric_name="Adversary", interval=1):
    super().__init__()
    self.dataloader = dataloader
    self.adversary = adversary
    self.interval = interval
    self.num_epochs = 0
    self.metric_name = metric_name

  def register_metric(self):
    return self.metric_name

  def run(self, session):
    correct = 0
    total = 0

    for data, label in tqdm(self.dataloader, leave=False):
        data, label = data.to("cuda"), label.to("cuda")
        perturbed_data = self.adversary.perturb(data, label)
        total += len(label)
        
        # Re-classify the perturbed image
        output = session.model(perturbed_data)
        if isinstance(output, list): output = output[-1]  
        _, final_pred = output.max(1)
        correct += torch.sum(final_pred == label)

    acc = correct/float(total)
    return acc

  def on_epoch_end(self, session, schedule, cb_dict, *args, **kwargs):
    self.num_epochs += 1
    if self.num_epochs % self.interval != 0: 
      cb_dict[self.metric_name] = None
      return
     
    accuracy = self.run(session)

    cb_dict[self.metric_name] = accuracy

    session.add_meta(f"{self.metric_name} Epoch {self.num_epochs}", str(accuracy))
    
