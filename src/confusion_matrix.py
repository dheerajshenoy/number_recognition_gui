import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import Model
from torch.utils.data import DataLoader
from CSVDataset import CSVDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_state_dict(torch.load("../model.pth"))
model.to(device)
model.eval()

dataset = CSVDataset("../data/mnist_test.csv")

test_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Usage
true_labels, predicted_labels = evaluate_model(model, test_loader, device=device)

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
