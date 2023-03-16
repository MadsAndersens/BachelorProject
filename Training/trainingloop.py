import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
from tqdm import tqdm
from torchmetrics import F1Score
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor


# Local imports
from DataRepresentation import SolarELData, train_data, test_data

# Hyperparameters
n_epochs = 3
batch_size_train = 4
batch_size_test = 4
learning_rate = 0.01
momentum = 0.5
decay_gamma = 0.9
label_smoothing = 0.1
loss_weights = torch.tensor(
                            [1-0.1, # Crack A
                             1-0.1, # Crack B
                             1-0.1, # Crack C
                             1-0.1, # Finger Failure
                             1-0.9] # Negative
                            )

# Load the DataSet.csv file
train_set = pd.read_csv('../Data/VitusData/Train.csv')
val_set = pd.read_csv('../Data/VitusData/Val.csv')
test_set = pd.read_csv('../Data/VitusData/Test.csv')

# Create a class for the data set
class SolarELData(Dataset):
    def __init__(self, DataFrame, transform=None):
        self.data_set = DataFrame
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        #print(self.data_set)
        img_name = self.data_set.iloc[idx, 0]
        image = Image.open(img_name)
        image = ToTensor()(image)

        # Resize the image to 224x224
        image = transforms.Resize((224, 224))(image)
        # Make three channels
        image = torch.cat((image, image, image), 0)

        label = self.data_set.iloc[idx, 1]
        label = self.one_hot_encode(label)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def one_hot_encode(self, label):
        # One hot encoding
        if label == 'Crack A':
            label = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        elif label == 'Crack B':
            label = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)
        elif label == 'Crack C':
            label = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
        elif label == 'Finger Failure':
            label = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)
        else:
            label = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)
        return label

#Transforms
transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

test_transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# Create the data sets
train_loader = SolarELData(train_set, transform=transform_train)
val_loader = SolarELData(val_set, transform=test_transform)


#Init wandb
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="BachelorProject",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "FaultyCells",
        "epochs": n_epochs,
    }
)

#Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size_test,
                                          shuffle=False,
                                          num_workers=2)

# Load the pretrained model
model = models.resnet18(pretrained = False)


class ResnetModules(nn.Module):
    def __init__(self,model):
        super(ResnetModules, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.resnet(x)
        return x

class VGGModules(nn.Module):
    def __init__(self,model):
        super(VGGModules, self).__init__()
        self.vgg = model
        self.vgg.classifier[6] = nn.Linear(4096, 5)

    def forward(self, x):
        x = self.vgg(x)
        return x

class InceptionModules(nn.Module):
    def __init__(self,model):
        super(InceptionModules, self).__init__()
        self.inception = model
        self.inception.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.inception(x)
        return x


model = ResnetModules(model).to(device)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight = loss_weights,label_smoothing=label_smoothing)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)

# Define a function for training the model
def train_model(model,train_loader,validation_loader,optimizer,loss_fn,n_epochs,device):
    f1 = F1Score(task="multiclass", num_classes=5)

    for epoch in range(n_epochs):
        model.train()
        sum_train_loss = 0
        for idx,(images,labels) in enumerate(tqdm(train_loader)):
            images.to(device)
            labels.to(device)
            outputs = model(images)
            train_loss = loss_fn(outputs,labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            sum_train_loss += train_loss.item()

            if idx%100 == 0:
                print(sum_train_loss)

        # Here starts the validation part.
        val_predictions,val_labels = [], []
        sum_loss_val = 0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for idx,(image,labels) in enumerate(tqdm(validation_loader)):
                images.to(device)
                labels.to(device)
                outputs = model(images)
                loss_val = loss_fn(outputs,labels)
                val_predictions.append(F.softmax(outputs))
                val_labels.append(labels)
                sum_loss_val += loss_val.item()

        val_predictions = torch.stack(val_predictions,dim = 0)
        val_labels = torch.stack(val_labels, dim=0)

        f1_score_val = f1(val_predictions,val_labels)
        print(f1_score_val)


        wandb.log({"train_loss": train_loss,
                   "val_loss": loss_val,
                   "epoch": epoch,
                   "F1": f1_score_val})

    scheduler.step()

#train_model(model,train_loader,test_loader,optimizer,criterion,n_epochs,device)
#wandb.finish()



if __name__ == '__main__':
    # Train the model
    train_model(model,train_loader,test_loader,optimizer,criterion,n_epochs,device)
    wandb.finish()


