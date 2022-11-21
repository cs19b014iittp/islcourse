# kali
import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_blobs, make_circles, load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  # pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_points, centers=5)
  # write your code ...
  return X,y

def get_data_circles(n_points=100):
  # pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_points)
  # write your code ...
  return X,y

def get_data_mnist():
  # pass
  # write your code here
  # Refer to sklearn data sets
  X,y = load_digits(return_X_y=True)
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  # pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  km = KMeans(k)
  km.fit(X) 
  # this is the KMeans object
  # write your code ...
  return km

def assign_kmeans(km=None,X=None):
  # pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  # pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = homogeneity_score(ypred_1, ypred_2), completeness_score(ypred_1, ypred_2), v_measure_score(ypred_1, ypred_2) # you need to write your code to find proper values
  return h,c,v

###### PART 2 ######

def build_lr_model(X=None, y=None):
  # pass
  lr_model = LogisticRegression(fit_intercept=False)
  lr_model.fit(X,y)
  # write your code...
  # Build logistic regression, refer to sklearn
  return lr_model

def build_rf_model(X=None, y=None):
  # pass
  rf_model = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
  rf_model.fit(X,y)
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  # pass
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  y_pred = model1.predict(X)
  acc = accuracy_score(y, y_pred)
  prec = precision_score(y, y_pred, average='macro')
  rec = recall_score(y, y_pred, average='macro')
  f1 = f1_score(y, y_pred, average='macro')
  auc = roc_auc_score(y, model1.predict_proba(X), multi_class='ovr')

  # write your code here...
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {"C":np.logspace(-4,4,7), "penalty":["l1","l2"]}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = { 
    'n_estimators': [1, 10, 100],
    'max_depth' : [1,10,None],
    'criterion' :['gini', 'entropy']
  }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  grid_search_cv = GridSearchCV(model, param_grid = param_grid, scoring=metrics, refit='accuracy',cv=cv,return_train_score=True)
  grid_search_cv.fit(X, y)
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
  cv_results = pd.DataFrame.from_dict(grid_search_cv.cv_results_)
  cv_results = pd.DataFrame.from_dict(gs.cv_results_)
  # print(cv_results)
  df = cv_results[['params', 'mean_test_Accuracy', 'rank_test_Accuracy', 'mean_test_AUC', 'rank_test_AUC']]
  top_accuracy = df.loc[df['rank_test_Accuracy']==1]['mean_test_Accuracy'].iloc[0]
  top_auc = df.loc[df['rank_test_AUC']==1]['mean_test_AUC'].iloc[0]
  
  top1_scores = top_accuracy, top_auc
  
  return top1_scores

###### PART 3 ######

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super().__init__()
    # super(MyNN,self)
    
    self.fc_encoder = nn.Linear(inp_dim, hid_dim) # write your code inp_dim to hid_dim mapper
    self.fc_decoder = nn.Linear(hid_dim, inp_dim) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Linear(hid_dim, num_classes) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLU() #write your code - relu object
    self.softmax = nn.Softmax() #write your code - softmax object
    self.flatten = nn.Flatten()
    
  def forward(self,x):
    # x = self.flatten(x) # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    one_hot = torch.nn.functional.one_hot(yground, 10)
    log = torch.log(y_pred)
    mult = -torch.mul(one_hot, log)

    lc1 = torch.div(torch.sum(mult), len(y)) # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X,y = load_digits(return_X_y=True)
  X, y = torch.from_numpy(X), torch.from_numpy(y)
  # write your code
  return X,y

def get_loss_on_single_point(mynn,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  # the lossval should have grad_fn attribute set
  lossval.requires_grad_(True)
  return lossval

def train_combined_encdec_predictor(mynn,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimizer.step()
    
  return mynn
