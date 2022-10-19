import torch
from torch import nn
import torch.nn.functional as F

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class cs19b014NN(nn.Module):
  def __init__(self, model_type, loader_size, classes, config=None):
    super(cs19b014NN, self).__init__()
    self.classes = classes
    self.model_type = model_type
    self.config = config
    self.loader_size = loader_size

    if model_type == 0:
      self.linear_relu_stack = nn.Sequential(
          nn.Conv2d(loader_size[1], 16, 5),
          nn.ReLu(),
          nn.MaxPool2d(2,2),
          nn.Conv2d(16, 50, 5),
          nn.ReLu(),
          nn.MaxPool2d(2,2)
      )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        x = torch.flatten(x, 1)
        x = F.relu(nn.Linear(len(x), self.classess)(x))
        if model_type == 0:
          return x
        else:
          for i in range(len(config)):
              x = nn.Conv2d(config[i][0], config[i][1], config[i][2], stride=config[i][3], padding=config[i][4])(x)
              x = nn.ReLu()(x)
              x = nn.MaxPool2d(2, 2)(x)
              x = torch.flatten(x, 1)
              x = F.relu(nn.Linear(len(x), self.classess)(x))
        
        return x

  # ... your code ...
  # ... write init and forward functions appropriately ...
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  train_features, train_labels = next(iter(train_data_loader))
  loader_size = train_features.size()
  classes=None
  for X, y in dataloader:
    classes = len(y)
    break
  model = cs19b014NN(0, loader_size, classes)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters())

  for epoch in range(n_epochs):
    for batch, (X, y) in enumerate(train_data_loader):
        # Compute prediction and loss
        pred_ = model(X)
        softmax = nn.Softmax(dim=1)
        pred = softmax(pred_)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  print ('Returning model... (rollnumber: 14)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  train_features, train_labels = next(iter(train_data_loader))
  loader_size = train_features.size()
  classes=None
  for X, y in dataloader:
    classes = len(y)
    break
  model = cs19b014NN(1, loader_size, classes, config=config)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters())

  for epoch in range(n_epochs):
    for batch, (X, y) in enumerate(train_data_loader):
        # Compute prediction and loss
        pred_ = model(X)
        softmax = nn.Softmax(dim=1)
        pred = softmax(pred_)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  print ('Returning model... (rollnumber: 14)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  test_loss, correct = 0, 0

  with torch.no_grad():
      for X, y in test_data_loader:
          pred = model1(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  accuracy_val = 100*correct
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
  print ('Returning metrics... (rollnumber: 14)')
  
  return accuracy_val, precision_val, recall_val, f1score_val
