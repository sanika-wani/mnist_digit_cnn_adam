# phase3_train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import load_mnist
from model import MNISTCNN
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


EPOCHS = 5
LR = 0.001
BATCH_SIZE = 64


train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)


model = MNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nTest Accuracy: {accuracy:.4f}")


cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)


report = classification_report(all_labels, all_preds)
print("Classification Report:\n", report)


torch.save(model.state_dict(), "mnist_cnn.pt")
print("Model saved as mnist_cnn.pt")
