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

# Local imports
from DataRepresentation import SolarELData, train_data, test_data

# Hyperparameters
n_epochs = 3
batch_size_train = 4
batch_size_test = 4
learning_rate = 0.01
momentum = 0.5
decay_gamma = 0.9
loss_weights = torch.tensor([1-0.1,1-0.1,1-0.1,1-0.1,1-0.9])

#Init wandb
# start a new wandb run to track this script
#wandb.init(
#    # set the wandb project where this run will be logged
#    project="BachelorProject",
#
#    # track hyperparameters and run metadata
#    config={
#        "learning_rate": learning_rate,
#        "architecture": "CNN",
#        "dataset": "FaultyCells",
#        "epochs": n_epochs,
#    }
#)

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


model = ResnetModules(model).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight = loss_weights)
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


        #wandb.log({"train_loss": train_loss,
        #           "val_loss": loss_val,
        #           "epoch": epoch,
        #           "F1": f1_score_val})

    scheduler.step()

if __name__ == '__main__':
   # Train the model
    train_model(model,train_loader,test_loader,optimizer,criterion,n_epochs,device)
    #wandb.finish()


