import torch
from torch import nn
import torch.nn.functional as F

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class cs19b014NN(nn.Module, model_type, loader_size, classes, config=None):
  def __init__(self):
    super().__init__()

    self.linear_relu_stack = nn.Sequential(
      # nn.Linear(28*28, 512),
      # nn.ReLU(),
      # nn.Linear(512, 512),
      # nn.ReLU(),
      # nn.Linear(512, 10),
      if model_type === 1:
        for i in range(len(config)):
            nn.Conv2d(config[i][0], config[i][1], config[i][2], stride=config[i][3], padding=config[i][4])
            nn.ReLu();
            nn.MaxPool2d(2, 2)
      else:
        nn.Conv2d(loader_size[1], 16, 5)
        nn.ReLu()
        nn.MaxPool2d(2,2)
        nn.Conv2d(16, 50, 5)
        nn.ReLu()
        nn.MaxPool2d(2,2)
        nn.Linear(28*28, 512),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        # nn.ReLU(),
    )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        x = torch.flatten(x, 1)
        x = self.fc3(x)
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



  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required
  return model
  
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: xx)')
  
  return accuracy_val, precision_val, recall_val, f1score_val
