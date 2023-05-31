import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
import ast
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import F1Score, ConfusionMatrix, ROC, PrecisionRecallCurve,Precision,Recall
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,precision_recall_curve
import copy
import wandb
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision

#Special imports
from torchvision.ops import sigmoid_focal_loss
from torch.autograd import Variable
from CustomAugmentations import RandomGammaCorrection,mixup_data,mixup_criterion,PVLEAD,ZeroPad,LabelSmoothBCELoss,GaussianNoise
import random

from torchvision.transforms import ToTensor,Resize

model_id = random.randint(0,10000)

torch.manual_seed(0)

# Hyperparameters
n_epochs = 100
batch_size = 64
learning_rate = 5e-6
momentum = 0
decay_gamma = 0.95
label_smoothing = 0.05
weight_decay = 0.05
architecture = 'MobileNetV2'#'resnet50' #'resnet18'#'MobileNetV2'  # 'vgg13'#'resnet18'#'vgg16'#'resnet18'  # 'SimpleFF'#resnet34'#'InceptionV3'#'resnet152' # 'FineTune'
optimizer = 'Adam'  # 'Adam'#'SGD'
loss = 'BCE'#'BCE'  # 'BCEWithLogits'#'BCEWithLogits'  # 'CrossEntropy'#'FocalLoss'
sigma = 0.00  # For gaussian noise added.

#For variable net
conv_layers = 5

# For focal loss
gamma = 2
alpha = 0.25

# Handling image size
padding = True
padding_mode = 'constant'  # reflect'

# Use synthetic data?
syn_type = 'PoissonNormal'#'PoissonNormal' #'PoissonNormal'  # 'Mixed'#'NewPoisson'  # 'Poisson'#'Gaussian' #Poisson
synthetic_data = False
#Poisson_blend_type = 'Poisson_Normal' #Poisson_Normal

#Enrich with other dataset?
enrich = False

# Weighted cross entropy
loss_weights = torch.tensor([1])

#Undersample majority class 
undersample = False 

# Flags for fine Tuning.
fine_tune = True
model_path = '/work3/s204137/Models/FinalModels/MobileNetV2_7991_testf1_0.29357796907424927.tar'
model_thres = 0.3135

#Warmup setup
warmup = False
warmup_steps = 10

# Transforms
mixup = False
# Transforms
t_forms = [
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    Resize((430,430)),#ZeroPad((430,430))
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(contrast=0.25,brightness = 0.25),
    transforms.RandomRotation(2),
    transforms.RandomRotation([-90,90]),
    #GaussianNoise(mean=0, std=0.1),
    #transforms.GaussianBlur(5),
    transforms.Normalize((0.51, 0.51, 0.51), (0.15, 0.15, 0.15)) #padding_mode(132, 132, 132), (25, 25, 25)
]
transform_train = transforms.Compose(t_forms)

test_transform = transforms.Compose(
    [
     transforms.PILToTensor(),
     transforms.ConvertImageDtype(torch.float),
     Resize((430,430)),#ZeroPad((430,430))
     transforms.Normalize((0.51, 0.51, 0.51), (0.15, 0.15, 0.15))
    ]
)

train_set = torchvision.datasets.ImageFolder('/work3/s204137/ImageFolderComp/Train',transform = transform_train)
l = len(train_set)
val_set = torchvision.datasets.ImageFolder('/work3/s204137/ImageFolderComp/Val',transform = test_transform)
test_set = torchvision.datasets.ImageFolder('/work3/s204137/ImageFolderComp/Test',transform = test_transform)

targetlst = list(train_set.targets)
#Synthetic
if synthetic_data:
	
    if syn_type == 'Mixed':
       org_train_set = torchvision.datasets.ImageFolder('/work3/s204137/ImageFolderComp/Train',transform = transform_train)
       syn_set_poisson =  torchvision.datasets.ImageFolder(f'/work3/s204137/ImageFolderCompPoissonNormal',transform = transform_train)
       syn_set_gaus = torchvision.datasets.ImageFolder(f'/work3/s204137/ImageFolderCompGaussian',transform = transform_train)
       targetlst = list(syn_set_poisson.targets)+list(syn_set_gaus.targets)+targetlst
       train_set = torch.utils.data.ConcatDataset([train_set, syn_set_poisson,syn_set_gaus])
    else: 
      syn_set = torchvision.datasets.ImageFolder(f'/work3/s204137/ImageFolderComp{syn_type}',transform = transform_train)
      targetlst = targetlst+list(syn_set.targets)
      train_set = torch.utils.data.ConcatDataset([train_set, syn_set])

neg_weight,pos_weigth = 1/targetlst.count(0),1/targetlst.count(1)
random_sampler_weights = [neg_weight if x == 0 else pos_weigth for x in targetlst]

sampler = WeightedRandomSampler(num_samples = len(train_set),
                                replacement = True,
                                weights = random_sampler_weights) 

train_loader = DataLoader(train_set,batch_size=batch_size,sampler = sampler) if not synthetic_data else DataLoader(train_set,batch_size=batch_size,shuffle = True)
#train_loader = DataLoader(train_set,batch_size=batch_size,shuffle = True)

#Instance the org trainloader
if warmup:
   org_train_set = torchvision.datasets.ImageFolder('/work3/s204137/ImageFolderComp/Train',transform = transform_train) 
   org_targetlst = list(org_train_set.targets)
   org_neg_weight,org_pos_weigth = 1/org_targetlst.count(0),1/org_targetlst.count(1)
   org_random_sampler_weights = [org_neg_weight if x == 0 else org_pos_weigth for x in org_targetlst]
   org_sampler = WeightedRandomSampler(num_samples = len(org_train_set),
                                replacement = True,
                                weights = org_random_sampler_weights) 
                                
   org_train_loader = DataLoader(org_train_set,batch_size=batch_size,sampler = org_sampler) if not synthetic_data else DataLoader(org_train_set,batch_size=batch_size,shuffle = True)

val_loader = DataLoader(val_set,batch_size=batch_size)
test_loader = DataLoader(test_set,batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hp_dict = {
    'n_epochs': n_epochs,
    'batch_size':batch_size ,
    'learning_rate': learning_rate,
    'momentum': momentum,
    'decay_gamma': decay_gamma,
    'label_smoothing': label_smoothing,
    'weight_decay': weight_decay,
    'architecture': architecture,
    'optimizer': optimizer,
    'loss': loss,
    'gamma': gamma,
    'alpha': alpha,
    'syn_type': syn_type,
    'synthetic_data': synthetic_data,
    'loss_weights': loss_weights,
    'Sigma': sigma,
    'Padding': padding,
    'Data_len': l,
    'Fine_tune': model_path if fine_tune else False,
    'Transforms': [t.__repr__() for t in t_forms],
    'mixup': mixup,
    'model_id': model_id,
    'NegWeight':neg_weight,
	'PosWeight':pos_weigth
}

wandb.init(
    # set the wandb project where this run will be logged
    project="BachelorProjectBinaryFinal",
    # track hyperparameters and run metadata
    config=hp_dict
)


# Load the pretrained model
if architecture == 'resnet18':
    model = models.resnet18(pretrained=True)
elif architecture == 'resnet34':
    model = models.resnet34(pretrained=True)
elif architecture == 'resnet50':
    model = models.resnet50(pretrained=True)
elif architecture == 'resnet101':
    model = models.resnet101(pretrained=True)
elif architecture == 'resnet152':
    model = models.resnet152(pretrained=True)
elif architecture == 'vgg13':
    model = models.vgg13(pretrained=True)
elif architecture == 'vgg16':
    model = models.vgg16(pretrained=True)
elif architecture == 'vgg19':
    model = models.vgg19(pretrained=True)
elif architecture == 'InceptionV3':
    model = models.inception_v3(pretrained=True)
elif architecture == 'MobileNetV2':
    model = models.mobilenet_v2(pretrained=True)
elif architecture == 'VariableConvNet':
	model = VariableConvNet(in_channels=3, out_channels=16, conv_layers=conv_layers).to(device)
elif architecture == 'Vit_B_16':
	#weights = models.ViT_B_16_Weights.IMAGENET1K_V1
	model = models.vit_b_16()

# Make the modules for the different architectures
class MobileNetModules(nn.Module):
    def __init__(self, model):
        super(MobileNetModules, self).__init__()
        self.mobilenet = model
        #Input size is 440x440x1
        self.mobilenet.features[0][0] = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.mobilenet.classifier[1] = nn.Linear(1280, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.sigmoid(x)
        x = torch.ravel(x)
        return x

class ResnetModules(nn.Module):
    def __init__(self, model):
        super(ResnetModules, self).__init__()
        self.resnet = model
        self.out_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.out_dim, 1)
        self.sigmoid = nn.Sigmoid()
        # Change input size to be of size 320x320x1
        if padding:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # Resize image to input size of the model
        x = self.resnet(x)
        #x = self.sigmoid(x)
        x = torch.ravel(x)
        return x


class VGGModules(nn.Module):
    def __init__(self, model):
        super(VGGModules, self).__init__()
        self.vgg = model
        self.vgg.classifier[6] = nn.Linear(4096, 2)
        

    def forward(self, x):
        x = self.vgg(x)
        # x = nn.Sigmoid()(x)
        return x


class InceptionModules(nn.Module):
    def __init__(self, model):
        super(InceptionModules, self).__init__()
        self.inception = model
        self.inception.fc = nn.Linear(2048, 2)
        self.inception.AuxLogits.fc = nn.Linear(768, 2)

    def forward(self, x):
        x = self.inception(x)
        # x = nn.Sigmoid()(x)
        return x
        
class VIT_B_16(nn.Module):
    def __init__(self, model, num_classes):
        super(VIT_B_16, self).__init__()

        # Load the pre-trained VIT-B-16 model
        self.model = model

        self.model.heads.head = nn.Linear(in_features=768, out_features=1, bias=True)

        # Activation function
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the modified model
        logits = self.model(x)

        # Apply activation function
        logits = self.activation(logits)
        logits =  self.sigmoid(logits)
        logits = torch.ravel(logits)

        return logits

if architecture == 'resnet18' or architecture == 'resnet34' or architecture == 'resnet50' or architecture == 'resnet101' or architecture == 'resnet152':
    model = ResnetModules(model).to(device)
    if fine_tune:
        model.load_state_dict(torch.load(model_path))
elif architecture == 'vgg16' or architecture == 'vgg19' or architecture == 'vgg13':
    model = VGGModules(model).to(device)
    
    for param in model.parameters():
        param.requires_grad = True
    
    if fine_tune:
        model.load_state_dict(torch.load(model_path))
        
elif architecture == 'InceptionV3':
    model = InceptionModules(model).to(device)
    if fine_tune:
        model.load_state_dict(torch.load(model_path))
elif architecture == 'MobileNetV2':
    model = MobileNetModules(model).to(device)
    if fine_tune:
        model.load_state_dict(torch.load(model_path))
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
elif architecture == 'Vit_B_16':
    model = VIT_B_16(model,1).to(device)

###############################Loss#####################################
if loss == 'BCE':
    criterion = nn.BCELoss(reduction='mean')#LabelSmoothBCELoss(smoothing = label_smoothing)#nn.BCELoss(reduction='mean')
elif loss == 'BCEWithLogits':
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device), reduction='mean')
elif loss == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device), label_smoothing=label_smoothing)
elif loss == 'FocalLoss':
    # criterion = FocalLoss(alpha = alpha,gamma=gamma,weight = loss_weights.to(device))
    criterion = sigmoid_focal_loss  # FocalLoss(gamma=gamma,weights=loss_weights.to(device))

###############################Optimizer################################
if optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)#optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.0001, max_lr=0.01,step_size_up = 15,mode = 'triangular2') #optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)


# Define the training function
def train(dataloader, model, loss_fn, optimizer,loss_type,mixup):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    predictions, labels = [], []
    eps = label_smoothing
    sigmoid = nn.Sigmoid()
    for batch, (X,y) in enumerate(dataloader):
        X, yt = X.to(device), y.to(device)
        y = (1 - 2 * eps) * yt + eps
        
        if mixup: 
           inputs, targets_a, targets_b, lam = mixup_data(X, y, 1)
           inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        # Compute prediction error
        pred = model(X)
        y = y.to(torch.float64).cuda()
        pred = pred.to(torch.float64).cuda()
        
        #If mixup is used change the criterion to follow implementation from: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
        if mixup:
           targets_a = targets_a.to(torch.float64)
           targets_b = targets_b.to(torch.float64)
           loss = mixup_criterion(loss_fn, pred, targets_a, targets_b, lam)
        else:
           loss = loss_fn(pred, y) if loss_type != 'FocalLoss' else loss_fn(pred, y, reduction='mean')
          
        losses.append(float(loss.item()))
        predictions.append(pred)
        labels.append(yt)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    predictions = sigmoid(torch.cat(predictions))
    labels = torch.cat(labels)
    f1 = F1Score(task="binary", num_labels=2, average=None, threshold=float(0.5)).to(device)
    f_score = f1_score = f1(predictions, labels)
    
    return (sum(losses) / len(losses),f_score)

def find_optimal_threshold(precisions, recalls, thresholds):
    """
    Given a list of precisions, recalls, and corresponding thresholds, finds the optimal threshold
    that maximizes the F1 score (harmonic mean of precision and recall).

    Args:
    - precisions (list): A list of precisions.
    - recalls (list): A list of recalls.
    - thresholds (list): A list of thresholds corresponding to the precisions and recalls.

    Returns:
    - The optimal threshold (float).
    """

    # Calculate F1 scores for each threshold
    f1_scores = [2 * (p * r) / (p + r) if p + r > 0 else 0 for p, r in zip(precisions, recalls)]

    # Find the index of the threshold with the highest F1 score
    max_index = f1_scores.index(max(f1_scores))

    # Return the corresponding threshold
    return thresholds[max_index],f1_scores[max_index]

# Define the test function
def test(dataloader, model, loss_fn,loss_type,threshold = 0.5):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    predictions, labels = [], []
    c = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            
            predictions.append(pred)
            labels.append(y)
            
            y = y.to(torch.float64).cuda()
            pred = pred.to(torch.float64)
            test_loss += loss_fn(pred, y).item() if loss_type != 'FocalLoss' else loss_fn(pred, y, reduction='mean').item()

    confmat = ConfusionMatrix(task="binary", num_classes=2, threshold=float(threshold)).to(device)
    predictions = sigmoid(torch.cat(predictions))
    labels = torch.cat(labels)
    cm = confmat(torch.ravel(predictions), torch.ravel(labels))
    return test_loss / num_batches, cm, predictions, labels




if warmup: 
   warmup_optimizer = optim.Adam(model.parameters(), lr = 1e-5, weight_decay=weight_decay)
   for i in range(warmup_steps):
       train_loss,f_train_score = train(org_train_loader, model, criterion, warmup_optimizer,loss,mixup)
       val_loss, cm, predictions, labels = test(val_loader, model, criterion,loss)
       #Metrics 
       predictions = predictions.cpu().numpy()
       labels = labels.cpu().numpy()
       precision, recall, thresholds = precision_recall_curve(labels, predictions)
       thres,f1_score = find_optimal_threshold(precision, recall, thresholds)
       
       wandb.log({
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "F1 Score": f1_score,
        "WarmupStep": i,
        'F_train': f_train_score})



#Run training and testing
best_f1 = 0
e_best = 0
for t in range(n_epochs):
    #break  
    train_loss,f_train_score = train(train_loader, model, criterion, optimizer,loss,mixup)
    val_loss, cm, predictions, labels = test(val_loader, model, criterion,loss)
    scheduler.step()
    
    #Metrics 
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    thres,f1_score = find_optimal_threshold(precision, recall, thresholds)
    
    #f1_score = f1(predictions[:, 1], labels[:, 1])
    if f1_score > best_f1:
        best_f1 = f1_score
        best_thres = thres
        best_model_path = f'/work3/s204137/Models/{architecture}_id_{model_id}_epoch_{t}.tar'#Change the +44 later 
        torch.save(model.state_dict(),best_model_path)
        e_best = 0   

    #Create a dataframe with the negative predictions and labels
    negative_labels = labels == 0
    positive_labels = labels == 1

    #Create a dataframe with the negative predictions and labels
    negative_predictions = predictions[negative_labels]
    positive_predictions = predictions[positive_labels]

    #Create dataframes with the predictions
    negative_predictions = pd.DataFrame(negative_predictions, columns=['Negative'])
    positive_predictions = pd.DataFrame(positive_predictions, columns=['Positive'])
    
    plt.clf()
    #Plot the confusion matrix
    fig1,ax1 = plt.subplots(1)
    sns.heatmap(cm.cpu().numpy(), annot=True,ax = ax1,linewidth=.5,fmt='g',cmap = 'Blues',cbar = False,xticklabels= ['Negative','Positive'],yticklabels= ['Negative','Positive'])
    ax1.set_xlabel('Predicted', fontsize = 12) # x-axis label with fontsize 15
    ax1.set_ylabel('Ground Truth', fontsize = 12) # y-axis label with fontsize 15
	
    #Precision recall curve
    fig2,ax2 = plt.subplots(1)
    prec,recall, thresholds = precision_recall_curve(labels, predictions)
    sns.lineplot(x = recall, y = prec,ax = ax2,ci = None)
    ax2.set_xlabel('Recall', fontsize = 12) # x-axis label with fontsize 15
    ax2.set_ylabel('Prec.', fontsize = 12) # x-axis label with fontsize 15
    ax2.set_ylim([0, 1])

    wandb.log({
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "F1 Score": f1_score,
        "Positive Predictions": wandb.Table(dataframe=positive_predictions),
        "Negative Predictions": wandb.Table(dataframe=negative_predictions),
        "Confusion Matrix": wandb.Image(fig1),
        "Epoch": t,
        "Precision Recall Curve": wandb.Image(fig2),
        "Threshold": thres,
        "lr": optimizer.param_groups[0]['lr'],
        'F_train': f_train_score
    })
    e_best += 1
    
    if e_best > 10:
       break
      
    plt.close('all')

# Run on the test set
#Load the model from early stopping. 
state_dict = torch.load(best_model_path)
model.load_state_dict(state_dict)
#best_thres = 0.3135
#Run Test
test_loss, cm, predictions, labels = test(test_loader, model, criterion,loss,threshold = best_thres)
f1 = F1Score(task="binary", num_labels=2, average=None, threshold=float(best_thres)).to(device)
precision = Precision(task = 'binary', num_classes=2, threshold=float(best_thres)).to(device)
recall = Recall(task = 'binary', num_classes=2, threshold=float(best_thres)).to(device)

f1_score = f1(predictions, labels)
prec,recall = precision(predictions, labels),recall(predictions, labels)
save_path = f'/work3/s204137/Models/FinalModels/{architecture}_{model_id}_testf1_{f1_score}.tar'
torch.save(model.state_dict(),save_path)

#Create a dataframe with the negative predictions and labels
negative_labels = labels == 0
positive_labels = labels == 1

#Create a dataframe with the negative predictions and labels
negative_predictions = predictions[negative_labels]
positive_predictions = predictions[positive_labels]

#Create dataframes with the predictions
negative_predictions = pd.DataFrame(negative_predictions.cpu(), columns=['Negative'])
positive_predictions = pd.DataFrame(positive_predictions.cpu(), columns=['Positive'])

plt.clf()
#Plot the confusion matrix
fig1,ax1 = plt.subplots(1)
sns.heatmap(cm.cpu().numpy(), annot=True,ax = ax1,linewidth=.5,fmt='g',cmap = 'Blues',cbar = False)
    
#Log the test results
wandb.log({
    "Test Loss": test_loss,
    "Test F1 Score": f1_score,
    "Test Precission":prec ,
    "Test Recall": recall,
    "Test Thres": float(best_thres),
    "Test Positive Predictions": wandb.Table(dataframe=positive_predictions),
    "Test Negative Predictions": wandb.Table(dataframe=negative_predictions),
    "Test Confusion Matrix": wandb.Image(fig1)
})

if synthetic_data:
    
    syn_test_class = torchvision.datasets.ImageFolder('/work3/s204137/ImageFolderCompPoissonNormalTest',transform = test_transform)
    syn_test_loader = DataLoader(syn_test_class, batch_size=32, shuffle=False, num_workers=0)
    
    #Run test
    test_loss, cm, predictions, labels = test(syn_test_loader, model, criterion,loss,threshold = model_thres)
	
    #Create a dataframe with the negative predictions and labels
    negative_labels = labels == 0
    positive_labels = labels == 1

    #Create a dataframe with the negative predictions and labels
    negative_predictions = predictions[negative_labels]
    positive_predictions = predictions[positive_labels]

    #Create dataframes with the predictions
    negative_predictions = pd.DataFrame(negative_predictions.cpu(), columns=['Negative'])
    positive_predictions = pd.DataFrame(positive_predictions.cpu(), columns=['Positive'])

    plt.clf()
    #Plot the confusion matrix
    fig1,ax1 = plt.subplots(1)
    sns.heatmap(cm.cpu().numpy(), annot=True,ax = ax1,linewidth=.5,fmt='g',cmap = 'Blues',cbar = False)
    
	#Log the test results
    wandb.log({
        "Syn Test Loss": test_loss,
        "Syn Test F1 Score": f1_score,
        "Syn Test Positive Predictions": wandb.Table(dataframe=positive_predictions),
        "Syn Test Negative Predictions": wandb.Table(dataframe=negative_predictions),
        "Syn Test Confusion Matrix": wandb.Image(fig1)
        })

wandb.finish()
