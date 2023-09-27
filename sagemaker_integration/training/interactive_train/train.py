import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import runhouse as rh

import dotenv

dotenv.load_dotenv()


class TrainModel(rh.Module):
    def __init__(self, learning_rate, batch_size, num_epochs):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.optimizer = None
        self.criterion = None
        self.model = None

    def load_model(self):
        self.model = MyModel()

    def load_optimizer(self):
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_epoch(self, dataloader):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your model architecture here
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Define your dataset class
class MyDataset(data.Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # Launch the SageMaker cluster
    sm_cluster = rh.sagemaker_cluster(name="rh-sagemaker-training",
                                      role=os.getenv("AWS_ROLE_ARN"),
                                      instance_type="ml.g5.2xlarge").up_if_not().save()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()

    # Define command-line arguments with default values
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # Load the MNIST dataset and apply transformations
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    # Extract the training data and targets
    train_data = (
            train_dataset.data.view(-1, 28 * 28).float() / 255.0
    )  # Flatten and normalize the input
    train_targets = train_dataset.targets

    dataset = MyDataset(train_data, train_targets)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    RemoteModel = TrainModel(learning_rate, batch_size, num_epochs).to(sm_cluster,
                                                                       env=["./", "torchvision", "torch"])
    RemoteModel.load_model()
    RemoteModel.load_optimizer()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        RemoteModel.train_epoch(dataloader)

    # Save for future re-use
    RemoteModel.save("train_model")
