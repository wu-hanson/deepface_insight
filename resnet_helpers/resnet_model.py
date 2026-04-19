import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import time
from sklearn.metrics import accuracy_score


def get_resnet_model(num_classes: int = 2, pretrained: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)


    model.fc = nn.Linear(model.fc.in_features, num_classes)
    

    return model


def load_resnet_model(path: str, device: torch.device, num_classes: int = 2):

    model = get_resnet_model(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    total_batches = len(loader)

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
            current_loss = running_loss / total_samples
            current_acc = running_correct / total_samples
            print(f"    Batch [{batch_idx+1}/{total_batches}] - Loss: {current_loss:.4f} - Acc: {current_acc:.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_labels, all_preds


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5, scheduler=None):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

    model.load_state_dict(best_model_wts)
    return model, history