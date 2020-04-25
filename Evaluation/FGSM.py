import torch
import util
from session import *
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from callbacks import *
import torch.nn.functional as F


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_test(model, dataloader, epsilon):
    with EvalModel(model):
      # Accuracy counter
      correct = 0
      total = 0
      adv_examples = []

      # Loop over all examples in test set
      for data, target in tqdm(dataloader, desc=f"Epsilon={epsilon}", leave=False):
          total += len(target)

          # Send the data and label to the device
          data, target = util.to_gpu(data), util.to_gpu(target)

          # Set requires_grad attribute of tensor. Important for Attack
          data.requires_grad = True

          # Forward pass the data through the model
          output = model(data)
          if isinstance(output, list): output = output[-1]

          output = F.log_softmax(output, dim=1)

          _, init_pred = output.max(1) # get the index of the max log-probability

          # If the initial prediction is wrong, dont bother attacking, just move on
          mask = init_pred == target

          output = output[mask]
          target = target[mask]

          if len(target) == 0: continue         

          # Calculate the loss
          loss = F.nll_loss(output, target)

          # Zero all existing gradients
          model.zero_grad()

          # Calculate gradients of model in backward pass
          loss.backward()

          # Collect datagrad
          data_grad = data.grad.data

          # Call FGSM Attack
          perturbed_data = fgsm_attack(data[mask], epsilon, data_grad[mask])

          # Re-classify the perturbed image

          output = model(perturbed_data)
          if isinstance(output, list): output = output[-1]  

          # Check for success
          _, final_pred = output.max(1) # get the index of the max log-probability

          correct += torch.sum(final_pred == target)
              

      # Calculate final accuracy for this epsilon
      final_acc = correct/float(total)
      print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

      # Return the accuracy and an adversarial example
      return final_acc, adv_examples


def fgsm_test_range(model, dataloader, epsilons, save_examples=False):
    with EvalModel(model):
        accuracies = []
        examples = []

        # Run test for each epsilon
        for eps in tqdm(epsilons, desc="FGSM", leave=False):
            acc, ex = fgsm_test(model, dataloader, eps)
            accuracies.append(acc)
            if save_examples: examples.append(ex)

        return accuracies, examples


def fgsm_plot(epsilons, accuracies, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
        
    ax.plot(epsilons, accuracies, "*-", label=label)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")

    if label is not None:
        ax.legend()

    ax.grid()

    return ax
    

class FGSM(TrainCallback):
  def __init__(self, dataloader, epsilons=[.05, .1, .15, .2], interval=20):
    super().__init__()
    self.dataloader = dataloader
    self.epsilons = epsilons
    self.interval = interval
    self.num_epochs = 0

  def state_dict(self): return {}
  def load_state_dict(self, state_dict): pass

  def run(self, session):
    accuracies, _ = fgsm_test_range(session.model, self.dataloader, self.epsilons)

    ax = fgsm_plot(self.epsilons, accuracies)

    plt.show()

    session.model.zero_grad()
    session.optimizer.zero_grad()

    return accuracies

  def on_epoch_end(self, session, lossMeter):
    self.num_epochs += 1
    if self.num_epochs % self.interval != 0: return
     
    accuracies = self.run(session)

    string = "\n".join([f"Epsilon={eps} Accuracy={acc}" for eps, acc in zip(self.epsilons, accuracies)])
    session.add_meta(f"FGSM Epoch {self.num_epochs}", string)
    
