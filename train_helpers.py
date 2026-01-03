import time
import torch
from torch import nn
from d2l import torch as d2l

def train_and_eval(model, name, epochs=10, batchsize=128, resize=False):
    # Determine the resize value based on input type
    if isinstance(resize, tuple):
        # If user provided a specific tuple like (64, 64)
        data = d2l.FashionMNIST(batch_size=batchsize, resize=resize)
    elif resize is True:
        # If user just said True, use the default 224x224
        data = d2l.FashionMNIST(batch_size=batchsize, resize=(224, 224))
    else:
        # If resize is False (or any other value), don't resize
        data = d2l.FashionMNIST(batch_size=batchsize)

  # Explicitly move the model to GPU if available
  if torch.cuda.is_available():
      model = model.to(torch.device('cuda'))

  trainer = d2l.Trainer(max_epochs=epochs, num_gpus=1) # Changed to d2l.torch.Trainer # num_gpus=1 tells trainer to use GPU

  # The initial forward pass for Lazy layers needs the input on the correct device
  dummy_input = next(iter(data.val_dataloader()))[0]
  if torch.cuda.is_available():
      dummy_input = dummy_input.to(torch.device('cuda'))
  model.forward(dummy_input)

  model.apply(init_cnn)

  start = time.perf_counter()
  trainer.fit(model, data) # Trainer should handle moving data to GPU during training
  end = time.perf_counter()

  accuracy = evaluate_model(model, data.val_dataloader())
  training_time = end - start
  parameter_count = count_parameters(model)
  average_time_per_epoch = (end - start) / epochs # Calculate average time first
  print(f"Average time per epoch: {average_time_per_epoch:.4f} seconds") # Then format the result
  print(f"Accuracy: {accuracies[name]}")
  print(f"Parameter count: {parameter_count[name]}")
  return accuracy, training_time, parameter_count

def evaluate_model(model, data_loader):
    metric = d2l.Accumulator(2)
    model.eval()
    # Get the device of the model's parameters (assuming model is already on the correct device)
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in data_loader:
            # Move input data and labels to the same device as the model
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            metric.add(d2l.accuracy(y_hat, y), y.numel()) # Changed to d2l.torch.accuracy
    return metric[0] / metric[1]

def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)