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
from torchmetrics import F1Score,ConfusionMatrix,ROC
from sklearn.metrics import ConfusionMatrixDisplay
import copy
import wandb

# Hyperparameters
n_epochs = 120
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.00001
momentum = 0.5
decay_gamma = 0.0
label_smoothing = 0.1
weight_decay = 0.1
architecture = 'resnet18'

loss_weights = torch.tensor(
    [1]*5  # Negative
)

converters = {
    'Label': lambda x: ast.literal_eval(x),
    'MaskDir': lambda x: ast.literal_eval(x) if str(x) != 'nan' else x
}
root_dir = '//'
# Load the DataSet.csv file
train_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Train.csv', converters=converters)
val_set = pd.read_csv(f'{root_dir}BachelorProject/Data/VitusData/Val.csv', converters=converters)
synthetic_set = pd.read_csv(f'{root_dir}BachelorProject/Data/Synthetic/SyntheticData.csv')


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
        self.syn_root = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject'
        self.non_syn_root = '/Users/madsandersen/PycharmProjects/BscProjektData/BachelorProject/Data'

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
        #image = transforms.Resize((224, 224))(image)

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
        place_holder = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)

        if 'CrackA' in label:
            place_holder[0] = 1

        if 'CrackB' in label:
            place_holder[1] = 1

        if 'CrackC' in label:
            place_holder[2] = 1

        if 'FingerFailure' in label:
            place_holder[3] = 1

        if 'Negative' in label:
            place_holder[4] = 1

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
        image = transforms.Resize((224, 224))(image)
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
        place_holder = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        if 'CrackA' in label:
            place_holder[0] = 1
        if 'CrackB' in label:
            place_holder[1] = 1
        if 'CrackC' in label:
            place_holder[2] = 1
        if 'FingerFailure' in label:
            place_holder[3] = 1
        if 'Negative' in label:
            place_holder[4] = 1
        return place_holder

# Create the data loaders
synthetic_class = SolarELDataSyn(train_set,synthetic_set)
#train_set = SolarELDataSyn(train_set)
val_class = SolarELData(val_set)

train_loader = DataLoader(synthetic_class, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_class, batch_size=batch_size_test, shuffle=True)

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
        self.resnet.fc = nn.Linear(self.out_dim, 5)
        # Change input size to be of size 320x320x1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # Resize image to input size of the model
        x = transforms.Resize((320, 320))(x)
        x = self.resnet(x)
        x = nn.Sigmoid()(x)
        return x

class VGGModules(nn.Module):
    def __init__(self, model):
        super(VGGModules, self).__init__()
        self.vgg = model
        self.vgg.classifier[6] = nn.Linear(4096, 5)

    def forward(self, x):
        x = self.vgg(x)
        x = nn.Sigmoid()(x)
        return x

class InceptionModules(nn.Module):
    def __init__(self, model):
        super(InceptionModules, self).__init__()
        self.inception = model
        self.inception.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.inception(x)
        x = nn.Sigmoid()(x)
        return x

if architecture == 'resnet18' or architecture == 'resnet34' or architecture == 'resnet50' or architecture == 'resnet101' or architecture == 'resnet152':
    model = ResnetModules(model).to(device)
elif architecture == 'vgg16' or architecture == 'vgg19':
    model = VGGModules(model).to(device)
elif architecture == 'InceptionV3':
    model = InceptionModules(model).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device), label_smoothing=label_smoothing)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)


# Define a function for training the model
def train_model(model, train_loader, validation_loader, optimizer, loss_fn, n_epochs, device):
    f1 = F1Score(task="multilabel", num_labels=5, average=None).to(device)
    f1_macro = F1Score(task="multilabel", num_labels=5, average='macro').to(device)
    confmat = ConfusionMatrix(task="multilabel", num_labels=5).to(device)
    roc = ROC(task='multilabel', num_labels=5)

    f1_score_best = torch.tensor([-1] * 5)
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

        # Here starts the validation part.
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
                val_predictions.append(outputs)
                val_labels.append(labels)
                sum_loss_val += loss_val.item()

        val_predictions = torch.vstack(val_predictions)
        val_labels = torch.vstack(val_labels)

        f1_score_val = f1(val_predictions, val_labels)
        f1_macro_average = f1_macro(val_predictions, val_labels)

        Matrix = confmat(torch.round(val_predictions).type(torch.int64), val_labels.type(torch.int64))

        if f1_score_val.sum() > f1_score_best.sum():
            f1_score_best = f1_score_val
            best_model = copy.deepcopy(model)

        # Create the plot so it can be logged to wandb
        fig, ax = plt.subplots(1, 5)
        plt.tight_layout()
        lab_names = ['A', 'B', 'C', 'FF', 'N']
        for i in range(5):
            ConfusionMatrixDisplay(confusion_matrix=Matrix[i, :, :].cpu().detach().numpy(),
                                   display_labels=['NoL', lab_names[i]]).plot(ax=ax[i], colorbar=False)

        # Call the Roc function
        fpr, tpr, thresholds = roc(val_predictions.cpu(), val_labels.type(torch.int64).cpu())

        wandb.log({"train_loss": sum_train_loss,
                   "val_loss": sum_loss_val,
                   "F1_A": f1_score_val[0],
                   "F1_B": f1_score_val[1],
                   "F1_C": f1_score_val[2],
                   "F1_Finger": f1_score_val[3],
                   "F1_Negative": f1_score_val[4],
                   "F1_Macro_average": f1_macro_average,
                   "epoch": epoch,
                   "conf_mat": plt})

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        # ax = ax.ravel()
        l = ['Crack A', 'Crack B', 'Crack C', 'Finger Failure', 'Negative']
        b = np.linspace(0, 1, num=fpr[0].shape[0])
        for i in range(5):  # Plot for each ROC line
            sns.lineplot(y=tpr[i], x=fpr[i], errorbar=None)

        sns.lineplot(x=b, y=b, linestyle="dashed", color='Black')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(l)
        ax.set_title('ROC curve')

        wandb.log({"RocCurve": wandb.Image(plt)})

        scheduler.step()

    torch.save(best_model.state_dict(), f'Models/Synthetic{architecture}CwRandRotWMacro')


train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, device)
wandb.finish()



