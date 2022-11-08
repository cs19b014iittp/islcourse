import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore')

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
class cs19b014NN(nn.Module):
    def __init__(self, image_size, classes, config):
        super().__init__()
        self.config = config
        layers = []
        image_length = image_size[0]
        image_height = image_size[1]
        for i in range(len(self.config)):
          layers.append(("conv"+str(i), (nn.Conv2d(config[i][0], config[i][1], config[i][2], stride=config[i][3], padding=config[i][4]))))
          if config[i][4] != 'same':
            image_length = (image_size - config[i][2][0] + 2*config[i][4]) / config[i][3]
            image_height = (image_size - config[i][2][1] + 2*config[i][4]) / config[i][3]

        print(f"size of tensor after passing through convolutional layers: ({image_length}, {image_height})")
        self.conv_layers = nn.Sequential(OrderedDict(layers))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(image_height*image_length*config[-1][1], 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        pred_probab = self.softmax(x)
        return pred_probab

def loss_fn(pred, y):
  one_hot = torch.nn.functional.one_hot(y, 10)

  log = torch.log(pred)
  mult = -torch.mul(one_hot, log)
  
  return torch.div(torch.sum(mult), len(y))

def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  classes=len(train_data_loader.dataset.classes)
  image_size = None
  for X, y in train_data_loader:
    image_size = (X.size()[2], X.size()[3])
    break

  model = cs19b014NN(image_size, classes, config=config).to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  for epoch in range(n_epochs):
    for batch, (X, y) in enumerate(train_data_loader):
      X, y = X.to(device), y.to(device)

      pred = model(X)
      loss = loss_fn(pred, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
  return model

def test_model(model1=None, test_data_loader=None):
  model1 = model1.to(device)
  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  test_loss, correct = 0, 0

  y_pred = []
  y_true = []

  with torch.no_grad():
      for X, y in test_data_loader:
        X, y = X.to(device), y.to(device)
        pred = model1(X)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.argmax(axis=1).cpu().numpy())

  accuracy_val = accuracy_score(y_true, y_pred)
  precision_val = precision_score(y_true, y_pred, average='macro')
  recall_val = recall_score(y_true, y_pred, average='macro')
  f1score_val = f1_score(y_true, y_pred, average='macro')

  print(f"accuracy : {(100*accuracy_val):>0.1f}%")
  print("precision:", precision_val)
  print("recal    :", recall_val)
  print("f1score  :", f1score_val)
  return accuracy_val, precision_val, recall_val, f1score_val
