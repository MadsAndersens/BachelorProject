import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import F1Score,ConfusionMatrix,ROC,PrecisionRecallCurve
from sklearn.metrics import ConfusionMatrixDisplay
import copy
import wandb
from torchvision.utils import make_grid

#Import huggin face transformer model for image classification
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Hyperparameters
n_epochs = 120
batch_size_train = 128
batch_size_test = 128
learning_rate = 1e-5
momentum = 0.5
decay_gamma = 0.0
label_smoothing = 0.01
weight_decay = 0.2
architecture = 'resnet152'
optimizer = 'Adam'#'Adam'#'SGD'
loss = 'FocalLoss'#'CrossEntropy'#'FocalLoss'
sigma = 1.0 # For gaussian noise added.

#For focal loss
gamma = 1.5
alpha = 0.5

#Use synthetic data?
syn_type = 'Poisson'#'Gaussian' #Poisson
synthetic_data = True

loss_weights = torch.tensor([1, 4])

hp_dict = {
    'n_epochs': n_epochs,
    'batch_size_train': batch_size_train,
    'batch_size_test': batch_size_test,
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
    'loss_weights': loss_weights
}

converters = {
    'Label': lambda x: ast.literal_eval(x),
    'MaskDir': lambda x: ast.literal_eval(x) if str(x) != 'nan' else x
}
root_dir = ''#'/Users/madsandersen/PycharmProjects/BscProjektData/'
# Load the DataSet.csv file
train_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Train.csv', converters=converters)
val_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Val.csv', converters=converters)
synthetic_set = pd.read_csv(f'/work3/s204137/Poisson/Data/Synthetic/SyntheticData.csv')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a class for the data set
class SolarELDataSyn(Dataset):

    def __init__(self, DataFrame,synthetic_dataframe, transform=None):
        self.data_set = DataFrame
        self.synthetic_data_set = synthetic_dataframe

        #Create variable in both indicating if the image is synthetic or not
        self.data_set['Synthetic'] = False
        self.synthetic_data_set['Synthetic'] = True

        #Set the roots
        self.syn_root = '/work3/s204137/Poisson'#'/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject'
        self.non_syn_root = '/zhome/b4/b/156509/BachelorProject /BachelorProject/Data/'

        # Append the synthetic data set to the original data set
        self.data_set = pd.concat([self.data_set,self.synthetic_data_set])
        self.transform = transform

        # Synthetic data is stored in a different folder so create a variable in the dataframes containing the root
        self.data_set['root'] = self.data_set.apply(lambda x: self.syn_root if x['Synthetic'] else self.non_syn_root, axis=1)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_name = self.data_set['ImageDir'].iloc[idx]
        is_synthetic = self.data_set['Synthetic'].iloc[idx]

        #Use the different roots
        if is_synthetic:
            img_name = f'{self.syn_root}/{img_name}'
        else:
            img_name = f'{self.non_syn_root}/{img_name}'

        # Open the image
        image = Image.open(img_name)
        image = transforms.ToTensor()(image)

        # Resize the image to 224x224
        image = transforms.Resize((224, 224))(image)

        # Make three channels
        image = torch.cat((image, image, image), 0)

        label = self.data_set['Label'].iloc[idx]
        label = self.one_hot_encode(label if not is_synthetic else [label])

        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample

    def one_hot_encode(self, label):
        # One hot encoding
        place_holder = torch.tensor([0, 0], dtype=torch.float32)

        if 'Negative' in label:
            place_holder[0] = 1
        else:
            place_holder[1] = 1

        return place_holder


# This is for the none synthetic data since it does not need to take all the paths into account
class SolarELData(Dataset):

    def __init__(self, DataFrame, transform=None):
        self.data_set = DataFrame
        self.transform = transform
        self.root = 'BachelorProject/Data/'
        # self.root = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data'

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        img_name = self.data_set['ImageDir'].iloc[idx]
        img_name = f'{self.root}/{img_name}'
        image = Image.open(img_name)
        image = transforms.ToTensor()(image)

        # Resize the image to 224x224
        #image = transforms.Resize((224, 224))(image)
        #pad the image
        image = self.pad_image(image)

        # Make three channels
        image = torch.cat((image, image, image), 0)

        label = self.data_set['Label'].iloc[idx]
        label = self.one_hot_encode(label)

        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample

    def pad_image(self,image):
        max_widt, max_height = 430, 430
        width, height = image.shape[1], image.shape[2]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = transforms.Pad((0, 0, pad_widt, pad_height))(image)
        return image

    def one_hot_encode(self, label):
        # One hot encoding
        place_holder = torch.tensor([0, 0], dtype=torch.float32)

        if 'Negative' in label:
            place_holder[0] = 1
        else:
            place_holder[1] = 1

        return place_holder

# Create the data loaders
synthetic_class = SolarELDataSyn(train_set,synthetic_set)
#train_set = SolarELDataSyn(train_set)
val_class = SolarELData(val_set)

train_loader = DataLoader(synthetic_class, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_class, batch_size=batch_size_test, shuffle=True)

# Init wandb
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="BachelorProject",
    # track hyperparameters and run metadata
    config= hp_dict
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
elif architecture == 'vgg16':
    model = models.vgg16(pretrained=True)
elif architecture == 'vgg19':
    model = models.vgg19(pretrained=True)
elif architecture == 'InceptionV3':
    model = models.inception_v3(pretrained=True)

# Make the modules for the different architectures
class ResnetModules(nn.Module):
    def __init__(self, model):
        super(ResnetModules, self).__init__()
        self.resnet = model
        self.out_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.out_dim, 2)
        # Change input size to be of size 440x440x1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # Resize image to input size of the model
        #x = transforms.Resize((320, 320))(x)
        x = self.resnet(x)
        x = nn.Sigmoid()(x)
        return x

class VGGModules(nn.Module):
    def __init__(self, model):
        super(VGGModules, self).__init__()
        self.vgg = model
        self.vgg.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.vgg(x)
        x = nn.Sigmoid()(x)
        return x

class InceptionModules(nn.Module):
    def __init__(self, model):
        super(InceptionModules, self).__init__()
        self.inception = model
        self.inception.fc = nn.Linear(2048, 2)

        #Change the output size of the auxillary classifier
        self.inception.AuxLogits.fc = nn.Linear(768, 2)



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        #self.logits = logits
        self.reduce = reduce

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce
        if self.reduce:
            focal_loss = focal_loss.mean()
        return focal_loss


if architecture == 'resnet18' or architecture == 'resnet34' or architecture == 'resnet50' or architecture == 'resnet101' or architecture == 'resnet152':
    model = ResnetModules(model).to(device)
elif architecture == 'vgg16' or architecture == 'vgg19':
    model = VGGModules(model).to(device)
elif architecture == 'InceptionV3':
    model = InceptionModules(model).to(device)

# Define the loss function and optimizer
#criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device), label_smoothing=label_smoothing)
if loss == 'BCE':
    criterion = nn.BCELoss(weight=loss_weights.to(device), reduction='mean',label_smoothing=label_smoothing)
elif loss == 'BCEWithLogits':
    criterion = nn.BCEWithLogitsLoss(weight=loss_weights.to(device), reduction='mean')
elif loss == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device), label_smoothing=label_smoothing)
elif loss == 'FocalLoss':
    criterion = FocalLoss(weight=loss_weights.to(device), gamma=gamma)


if optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)


# this function finds the best threshold for each class by maximizing the F1 score
def get_thresholds(n_thresholds, precission, recall):
    fscore = (2*precission*recall)/(precission+recall)
    idx = torch.argmax(fscore)
    thres = n_thresholds[idx]
    return thres


# Function for converting probabilities to labels given thresholds
def convert_to_labels(probs, thres):
    labels = []
    for i in range(len(probs)):
        if probs[i] >= thres[i]:
            labels.append(1)
        else:
            labels.append(0)
    return labels


# Define a function for training the model
def train_model(model, train_loader, validation_loader, optimizer, loss_fn, n_epochs, device):
    f1 = F1Score(task="binary", num_labels=2, average=None).to(device)
    confmat = ConfusionMatrix(task="binary", num_classes=2).to(device)
    pr_curve = PrecisionRecallCurve(task="binary")
    # ml_auroc = MultilabelAUROC(num_labels=5, average=None, thresholds=5)

    f1_score_best = -1
    for epoch in range(n_epochs):
        model.train()
        sum_train_loss = 0
        for idx, sample in enumerate(train_loader):
            images = sample['image']
            labels = sample['label']

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            if architecture != 'InceptionV3':
                outputs = model(images)
                train_loss = loss_fn(F.softmax(outputs), labels)
            else:
                outputs, aux = model(images)
                train_loss = loss_fn(outputs, labels) + 0.1 * loss_fn(aux, labels)

            # backprop
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            sum_train_loss += train_loss.item()
            break

        ################### Here starts the validation part.#############################
        val_predictions, val_labels = [], []
        sum_loss_val = 0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for idx, sample_val in enumerate(validation_loader):
                images_val = sample_val['image']
                labels_val = sample_val['label']

                images = images_val.to(device)
                labels = labels_val.to(device)
                outputs = model(images)
                loss_val = loss_fn(outputs, labels)
                val_predictions.append(F.softmax(outputs))
                val_labels.append(labels)
                sum_loss_val += loss_val.item()

        val_predictions = torch.vstack(val_predictions)
        val_labels = torch.vstack(val_labels)

        # Call the Roc function
        fpr, tpr, thresholds = pr_curve(val_predictions.cpu(), val_labels.type(torch.int64).cpu())
        thres = get_thresholds(thresholds, tpr, fpr)  # obtain optimal thresholds
        val_predictions = val_predictions[:,1]
        val_predictions = convert_to_labels(val_predictions.cpu(), thres).to(device)

        f1_score_val = f1(val_predictions, val_labels)
        Matrix = confmat(val_predictions, val_labels)

        if f1_score_val > f1_score_best:
            f1_score_best = f1_score_val
            best_model = copy.deepcopy(model)

        # print(f1_score_val)
        disp = ConfusionMatrixDisplay(confusion_matrix=Matrix.cpu().detach().numpy(),
                                      display_labels=['Negative', 'Positive'])
        disp.plot()

        wandb.log({"train_loss": sum_train_loss,
                   "val_loss": sum_loss_val,
                   "F1": f1_score_val,
                   "epoch": epoch,
                   "confMat": plt})

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        # ax = ax.ravel()
        l = ['Failure', 'Negative']
        b = np.linspace(1, 0, num=fpr[0].shape[0])

        sns.lineplot(y=tpr, x=fpr, errorbar=None)

        sns.lineplot(x=b, y=b, linestyle="dashed", color='Black')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(l)
        ax.set_title('PrecRecall curve')
        wandb.log({"PrecRecall": wandb.Image(plt)})

        scheduler.step()

    torch.save(best_model.state_dict(), f'Models/Synthetic{architecture}CwBinary')


# Make a grid that showcases each class and its corresponding image


train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, device)
wandb.finish()



