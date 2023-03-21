import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
# from tqdm import tqdm
from torchmetrics import F1Score, ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
import ast
import copy

# Local imports
# from DataRepresentation import SolarELData, train_data, test_data

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
    [1 - (206 / 36052),  # Crack A
     1 - (292 / 36052),  # Crack B
     1 - (88 / 36052),  # Crack C
     1 - (267 / 36052),  # Finger Failure
     1 - (35542 / 36052)]  # Negative
)

converters = {
    'Label': lambda x: ast.literal_eval(x),
    'MaskDir': lambda x: ast.literal_eval(x) if str(x) != 'nan' else x
}
# Load the DataSet.csv file
train_set = pd.read_csv('BachelorProject/Data/VitusData/Train.csv', converters=converters)
val_set = pd.read_csv('BachelorProject/Data/VitusData/Val.csv', converters=converters)


# Create a class for the data set
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
        image = ToTensor()(image)

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


# Transforms
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),#transforms.RandomRotation([0, 180]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

test_transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Create the data sets
train_data = SolarELData(train_set, transform=transform_train)
val_data = SolarELData(val_set, transform=test_transform)

# Init wandb
# start a new wandb run to track this script
#wandb.init(
#    # set the wandb project where this run will be logged
#    project="BachelorProject",
#
#    # track hyperparameters and run metadata
#    config={
#        "learning_rate": learning_rate,
#        "architecture": f"{architecture}PreTrainedRandRot",
#        "dataset": "FaultyCells",
#        "epochs": n_epochs,
#        "weights": loss_weights,
#        "BatchSize": batch_size_train,
#        "LabelSmoothing": label_smoothing,
#        "WeightDecay": weight_decay,
#        "DecayGamma": decay_gamma
#    }
#)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           num_workers=0)

validation_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batch_size_test,
                                                shuffle=True,
                                                num_workers=0)

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



class ResnetModules(nn.Module):
    def __init__(self, model):
        super(ResnetModules, self).__init__()
        self.resnet = model
        self.out_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.out_dim, 5)

    def forward(self, x):
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

        Matrix = confmat(val_predictions, val_labels)

        if f1_score_val.sum() > f1_score_best.sum():
            f1_score_best = f1_score_val
            best_model = copy.deepcopy(model)

        # Create the plot so it can be logged to wandb
        fig, ax = plt.subplots(1, 5)
        lab_names = ['Crack A', 'Crack B', 'Crack C', 'Finger Failure', 'Negative']
        for i in range(5):
            ConfusionMatrixDisplay(confusion_matrix=Matrix[0, :, :].cpu().detach().numpy(), display_labels=lab_names, ax=ax[i])
        # ax[i].set_title(lab_names[i])

        # disp = ConfusionMatrixDisplay(confusion_matrix=Matrix.cpu().detach().numpy(),display_labels=lab_names)
        #disp.plot()

        # print(f1_score_val)

        #wandb.log({"train_loss": sum_train_loss,
        #           "val_loss": sum_loss_val,
        #           "F1_A": f1_score_val[0],
        #           "F1_B": f1_score_val[1],
        #           "F1_C": f1_score_val[2],
        #           "F1_Finger": f1_score_val[3],
        #           "F1_Negative": f1_score_val[4],
        #           "F1_Macro_average": f1_macro_average,
        #           "epoch": epoch,
        #           "conf_mat": plt})

        scheduler.step()

    torch.save(best_model.state_dict(), 'Models/NoAugmentationsVGG16CwRandRotWMacro')


train_model(model, train_loader, validation_loader, optimizer, criterion, n_epochs, device)
wandb.finish()

# if __name__ == '__main__':
# Train the model
# train_model(model, train_loader, validation_loader, optimizer, criterion, n_epochs, device)
# wandb.finish()


