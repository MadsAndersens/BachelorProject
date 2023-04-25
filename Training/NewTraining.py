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
from torchmetrics import F1Score, ConfusionMatrix, ROC, PrecisionRecallCurve
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,PrecisionRecallDisplay
import copy
import wandb
from torchvision.utils import make_grid

from torchvision.ops import sigmoid_focal_loss
from CustomAugmentations import RandomGammaCorrection

torch.manual_seed(0)
# Hyperparameters
n_epochs = 50
batch_size_train = 64
batch_size_test = 64
learning_rate = 1e-6
momentum = 0.1
decay_gamma = 0.0
label_smoothing = 0.0
weight_decay = 0.1
architecture = 'resnet18'  # 'vgg13'#'resnet18'#'vgg16'#'resnet18'  # 'SimpleFF'#resnet34'#'InceptionV3'#'resnet152' # 'FineTune'
optimizer = 'SGD'  # 'Adam'#'SGD'
loss = 'BCEWithLogits'  # 'BCEWithLogits'#'BCEWithLogits'  # 'CrossEntropy'#'FocalLoss'
sigma = 0.00  # For gaussian noise added.

# For focal loss
gamma = 2
alpha = 0.20

# Handling image size
padding = True
padding_mode = 'constant'  # reflect'

# Use synthetic data?
syn_type = 'Mixed'  # 'Mixed'#'NewPoisson'  # 'Poisson'#'Gaussian' #Poisson
synthetic_data = False

# Weighted cross entropy
loss_weights = torch.tensor([1, 10])

# Flags for fine Tuning.
fine_tune = False
model_path = '/zhome/b4/b/156509/BachelorProject /Models/VitusConfig'

# Transforms
# Transforms
t_forms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(contrast=0.25),
    transforms.RandomRotation(2),
    transforms.RandomResizedCrop(size=(430, 430), scale=(0.7, 1.0)),
    RandomGammaCorrection(gamma_range=(2 / 3, 3 / 2)),
    transforms.Normalize((132, 132, 132), (25, 25, 25))
]
transform_train = transforms.Compose(t_forms)

test_transform = transforms.Compose(
    [transforms.Normalize((132, 132, 132), (25, 25, 25))]
)

converters = {
    'Label': lambda x: ast.literal_eval(x),
    'MaskDir': lambda x: ast.literal_eval(x) if str(x) != 'nan' else x
}
root_dir = ''  # '/Users/madsandersen/PycharmProjects/BscProjektData/'
# Load the DataSet.csv file
train_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Train.csv', converters=converters)
val_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Val.csv', converters=converters)

if synthetic_data:
    if syn_type != 'Mixed':
        # synthetic_set = pd.read_csv(f'/work3/s204137/{syn_type}/Data/Synthetic/SyntheticData.csv')
        synthetic_set = pd.read_csv(f'/work3/s204137/Mixed/{syn_type}/SyntheticData.csv')
    elif syn_type == 'Mixed':
        # syn_gauss = pd.read_csv(f'/work3/s204137/Gaussian/Data/Synthetic/SyntheticData.csv')
        syn_gauss = pd.read_csv(f'/work3/s204137/Mixed/Gaussian/SyntheticData.csv')
        # syn_pois = pd.read_csv(f'/work3/s204137/Poisson/Data/Synthetic/SyntheticData.csv')
        syn_pois = pd.read_csv(f'/work3/s204137/Mixed/Poisson/SyntheticData.csv')
        synthetic_set = pd.concat([syn_gauss, syn_pois])

l = len(synthetic_set) if synthetic_data else len(train_set)

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
    'loss_weights': loss_weights,
    'Sigma': sigma,
    'Padding': padding,
    'Data_len': l,
    'Fine_tune': fine_tune,
    'Transforms': [t.__repr__() for t in t_forms]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create a class for the data set
class SolarELDataSyn(Dataset):

    def __init__(self, DataFrame, synthetic_dataframe, transform=None):
        self.data_set = DataFrame
        self.synthetic_data_set = synthetic_dataframe

        # Create variable in both indicating if the image is synthetic or not
        self.data_set['Synthetic'] = False
        self.synthetic_data_set['Synthetic'] = True

        # Set the roots
        if syn_type == 'Mixed':
            self.syn_root_gaussian = f'/work3/s204137/Mixed/Gaussian'
            self.syn_root_poisson = f'/work3/s204137/Mixed/Poisson'

        self.syn_root = f'/work3/s204137/Mixed/{syn_type}'  # '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject'
        self.non_syn_root = '/zhome/b4/b/156509/BachelorProject /BachelorProject/Data/'

        # Append the synthetic data set to the original data set
        self.data_set = pd.concat([self.data_set, self.synthetic_data_set])
        self.transform = transform

        # Synthetic data is stored in a different folder so create a variable in the dataframes containing the root
        self.data_set['root'] = self.data_set.apply(lambda x: self.syn_root if x['Synthetic'] else self.non_syn_root,
                                                    axis=1)
        self.classes = ['Positive', 'Negative']

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):

        is_synthetic = self.data_set['Synthetic'].iloc[idx]

        if is_synthetic:
            img_name = self.data_set['ImageDir'].iloc[idx][23:]
        else:
            img_name = self.data_set['ImageDir'].iloc[idx]

        # Use the different roots
        if is_synthetic:
            if syn_type == 'Mixed':
                img_name_G = f'{self.syn_root_gaussian}/{img_name}'
                img_name_P = f'{self.syn_root_poisson}/{img_name}'

            else:
                img_name = f'{self.syn_root}/{img_name}'
        else:
            img_name = f'{self.non_syn_root}/{img_name}'

        # Open the image
        if syn_type == 'Mixed' and is_synthetic:
            try:
                image = Image.open(img_name_G)
            except:
                image = Image.open(img_name_P)
        else:
            image = Image.open(img_name)

        image = transforms.ToTensor()(image) * 255

        # Resize the image to 224x224
        if not padding:
            if architecture != 'InceptionV3':
                image = transforms.Resize((224, 224))(image)
            else:
                image = transforms.Resize((299, 299))(image)
        else:
            image = self.pad_image(image)

        # Make three channels
        if image.shape[0] == 1:
            image = torch.cat((image, image, image), 0)

        label = self.data_set['Label'].iloc[idx]
        label = self.one_hot_encode(label if not is_synthetic else [label])

        if self.transform is not None:
            image = self.transform(image)
            image = self.gaussian_noise(image)

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

    def gaussian_noise(self, img):
        out = img + sigma * torch.randn_like(img)
        return out

    def pad_image(self, image):
        max_widt, max_height = 430, 430
        width, height = image.shape[2], image.shape[1]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = transforms.Pad((0, 0, pad_widt, pad_height), padding_mode=padding_mode)(image)
        return image


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
        image = transforms.ToTensor()(image) * 255

        if not padding:
            if architecture != 'InceptionV3':
                image = transforms.Resize((224, 224))(image)
            else:
                image = transforms.Resize((299, 299))(image)
        else:
            image = self.pad_image(image)

        # Make three channels
        image = torch.cat((image, image, image), 0)

        label = self.data_set['Label'].iloc[idx]
        label = self.one_hot_encode(label)

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

    def pad_image(self, image):
        max_widt, max_height = 430, 430
        width, height = image.shape[2], image.shape[1]
        pad_widt, pad_height = max_widt - width, max_height - height
        image = transforms.Pad((0, 0, pad_widt, pad_height), padding_mode=padding_mode)(image)
        return image


# Create the data loaders
if synthetic_data:
    synthetic_class = SolarELDataSyn(train_set, synthetic_set, transform=transform_train)
else:
    train_class = SolarELData(train_set, transform=transform_train)

# train_set = SolarELDataSyn(train_set)
val_class = SolarELData(val_set, transform=test_transform)

if synthetic_data:
    train_loader = DataLoader(synthetic_class, batch_size=batch_size_train, shuffle=True, num_workers=0)
else:
    train_loader = DataLoader(train_class, batch_size=batch_size_test, shuffle=True, num_workers=0)

val_loader = DataLoader(val_class, batch_size=batch_size_test, shuffle=True, num_workers=0)

# Init wandb
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="BachelorProjectBinary",
    # track hyperparameters and run metadata
    config=hp_dict
)


# Load the pretrained model
if architecture == 'resnet18':
    model = models.resnet18(pretrained=False)
elif architecture == 'resnet34':
    model = models.resnet34(pretrained=False)
elif architecture == 'resnet50':
    model = models.resnet50(pretrained=False)
elif architecture == 'resnet101':
    model = models.resnet101(pretrained=False)
elif architecture == 'resnet152':
    model = models.resnet152(pretrained=False)
elif architecture == 'vgg13':
    model = models.vgg13(pretrained=False)
elif architecture == 'vgg16':
    model = models.vgg16(pretrained=False)
elif architecture == 'vgg19':
    model = models.vgg19(pretrained=False)
elif architecture == 'InceptionV3':
    model = models.inception_v3(pretrained=False)
elif architecture == 'MobileNetV2':
    model = models.mobilenet_v2(pretrained=False)

# Make the modules for the different architectures
class ResnetModules(nn.Module):
    def __init__(self, model):
        super(ResnetModules, self).__init__()
        self.resnet = model
        self.out_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.out_dim, 2)
        # Change input size to be of size 320x320x1
        if padding:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # Resize image to input size of the model
        # x = transforms.Resize((320, 320))(x)
        x = self.resnet(x)
        # x = nn.Sigmoid()(x)
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=torch.tensor([1, 1]), reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.logits = logits
        self.reduce = reduce
        self.weight = weight

    def forward(self, pred_logits, target):
        # pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, weight=self.weight, reduction='none')
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce
        if self.reduce:
            focal_loss = focal_loss.mean()
        return focal_loss

    ###############################architecture################################


if architecture == 'resnet18' or architecture == 'resnet34' or architecture == 'resnet50' or architecture == 'resnet101' or architecture == 'resnet152':
    model = ResnetModules(model).to(device)
    if fine_tune:
        model.load_state_dict(torch.load(model_path))
elif architecture == 'vgg16' or architecture == 'vgg19' or architecture == 'vgg13':
    model = VGGModules(model).to(device)
    if fine_tune:
        model.load_state_dict(torch.load(model_path))
elif architecture == 'InceptionV3':
    model = InceptionModules(model).to(device)
    if fine_tune:
        model.load_state_dict(torch.load(model_path))

# Define the loss function and optimizer
###############################Loss#####################################
if loss == 'BCE':
    criterion = nn.BCELoss(weight=loss_weights.to(device), reduction='mean', label_smoothing=label_smoothing)
elif loss == 'BCEWithLogits':
    criterion = nn.BCEWithLogitsLoss(weight=loss_weights.to(device), reduction='mean')
elif loss == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device), label_smoothing=label_smoothing)
elif loss == 'FocalLoss':
    # criterion = FocalLoss(alpha = alpha,gamma=gamma,weight = loss_weights.to(device))
    criterion = sigmoid_focal_loss  # FocalLoss(gamma=gamma,weights=loss_weights.to(device))

###############################Optimizer################################
if optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)

# Define the training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    for batch, sample in enumerate(dataloader):
        X,y = sample['image'], sample['label']
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(losses) / len(losses)


# Define the test function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    predictions, labels = [], []
    c = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions.append(F.softmax(pred, dim=1))
            labels.append(y)
            test_loss += loss_fn(pred, y).item()

    confmat = ConfusionMatrix(task="binary", num_classes=2, threshold=float(0.5)).to(device)
    predictions = torch.vstack(predictions)
    labels = torch.vstack(labels)
    cm = confmat(predictions[:, 1], labels[:, 1])
    return test_loss / num_batches, cm, predictions, labels

#Run training and testing
best_f1 = 0
f1 = F1Score(task="binary", num_labels=2, average=None, threshold=float(0.5)).to(device)
for t in range(n_epochs):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_loss, cm, predictions, labels = test(val_loader, model, criterion)
    scheduler.step()
    f1_score = f1(predictions[:, 1], labels[:, 1])
    if f1_score > best_f1:
        best_f1 = f1_score
        torch.save(model.state_dict(), f'/zhome/b4/b/156509/BachelorProject /Models/{architecture}_{best_f1}')

    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    #Create a dataframe with the negative predictions and labels
    negative_labels = labels[:, 0]
    positive_labels = labels[:, 1]

    #Create a dataframe with the negative predictions and labels
    negative_predictions = predictions[negative_labels,0]
    positive_predictions = predictions[positive_labels,1]

    #Create dataframes with the predictions
    negative_predictions = pd.DataFrame(negative_predictions, columns=['Negative'])
    positive_predictions = pd.DataFrame(positive_predictions, columns=['Positive'])

    #Plot the confusion matrix
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm.numpy(), display_labels=['Negative', 'Positive'])

    #Precision recall curve
    dispPR = PrecisionRecallDisplay.from_predictions(labels[:, 1].to(int), predictions[:, 1])

    wandb.log({
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "F1 Score": f1_score,
        "Positive Predictions": wandb.Table(dataframe=positive_predictions),
        "Negative Predictions": wandb.Table(dataframe=negative_predictions),
        "Confusion Matrix": wandb.Image(disp.plot()),
        "Epoch": t,
        "Precision Recall Curve": wandb.Image(dispPR.plot())
    })

# Run on the test set
test_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Val.csv', converters=converters)
test_class = SolarELData(test_set, root_dir, transform = test_transform)
test_loader = DataLoader(test_class, batch_size = batch_size_test, shuffle = True, num_workers = 0)

test_loss, cm, predictions, labels = test(test_loader, model, criterion)
f1_score = f1(predictions[:, 1], labels[:, 1])

#Create a dataframe with the negative predictions and labels
negative_labels = labels[:, 0]
positive_labels = labels[:, 1]

#Create a dataframe with the negative predictions and labels
negative_predictions = predictions[negative_labels,0]
positive_predictions = predictions[positive_labels,1]

#Create dataframes with the predictions
negative_predictions = pd.DataFrame(negative_predictions, columns=['Negative'])
positive_predictions = pd.DataFrame(positive_predictions, columns=['Positive'])

plt.clf()
#Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm.numpy(), display_labels=['Negative', 'Positive'])

#Log the test results
wandb.log({
    "Test Loss": test_loss,
    "Test F1 Score": f1_score,
    "Test Positive Predictions": wandb.Table(dataframe=positive_predictions),
    "Test Negative Predictions": wandb.Table(dataframe=negative_predictions),
    "Test Confusion Matrix": wandb.Image(disp.plot())
})


