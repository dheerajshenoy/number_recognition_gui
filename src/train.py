import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from CSVDataset import CSVDataset
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()

model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

dataset = CSVDataset("./data/mnist_train.csv", transform)

train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

for epoch in range(25):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.2f}%")

torch.save(model.state_dict(), "model.pth")
