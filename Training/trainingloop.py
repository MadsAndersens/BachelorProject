import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
from tqdm import tqdm


# Local imports
from DataRepresentation import SolarELData, train_data, test_data

# Hyperparameters
n_epochs = 10
batch_size_train = 4
batch_size_test = 4
learning_rate = 0.01
momentum = 0.5

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


model = ResnetModules(model).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

if __name__ == '__main__':
    # Train the model
    running_loss = 0.0

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # Training loop
        for i, data in tqdm(enumerate(train_loader, 0)):
            # get the inputs
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                wandb.log({"train_loss": loss})

    # Validation loop
        for i, data in tqdm(enumerate(test_loader, 0)):
            # get the inputs
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            with torch.no_grad():
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)
                wandb.log({"test_loss": loss})

            # print statistics
            running_loss += loss.item()


    wandb.finish()


