"""Train and evaluate the CNN on the 3-kinds-of-pneumonia chest X-ray dataset.

Usage:  python train.py --data path/to/dataset
The dataset folder must contain one sub-folder per class (COVID-19, Normal,
Pneumonia-Bacterial, Pneumonia-Viral), as in the Kaggle dataset.
"""
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from model import build_model

SEED, IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 42, 150, 16, 15, 1e-3
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class TransformSubset(Dataset):
    """Subset of an ImageFolder with its own transform (augmentation only for training)."""
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    base = datasets.ImageFolder(args.data)           # PIL images, no transform yet
    class_names = base.classes
    targets = np.array(base.targets)

    # 70 / 15 / 15 stratified split
    rng = np.random.RandomState(SEED)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(len(class_names)):
        idx = rng.permutation(np.where(targets == c)[0])
        n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
        train_idx += idx[:n_tr].tolist()
        val_idx += idx[n_tr:n_tr + n_va].tolist()
        test_idx += idx[n_tr + n_va:].tolist()
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # convert grayscale X-rays to 3 channels
    to_rgb = lambda img: img.convert("RGB")
    train_tf = transforms.Compose([transforms.Lambda(to_rgb), train_tf])
    eval_tf = transforms.Compose([transforms.Lambda(to_rgb), eval_tf])

    train_loader = DataLoader(TransformSubset(base, train_idx, train_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TransformSubset(base, val_idx, eval_tf), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TransformSubset(base, test_idx, eval_tf), batch_size=BATCH_SIZE)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        tr_loss, tr_acc = loss_sum / n, correct / n
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc)):
            hist[k].append(v)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "best_model.pth")

    # ---- test with the best-validation checkpoint ----
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = torch.softmax(model(x.to(device)), 1).cpu().numpy()
            y_prob.append(p); y_pred += p.argmax(1).tolist(); y_true += y.tolist()
    y_prob = np.vstack(y_prob)
    print(f"\nTest accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right"); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.colorbar(im); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].plot(hist["train_acc"], label="Train Acc"); axs[0].plot(hist["val_acc"], label="Val Acc")
    axs[0].set_title("Training vs Validation Accuracy"); axs[0].set_xlabel("Epochs"); axs[0].legend()
    axs[1].plot(hist["train_loss"], label="Train Loss"); axs[1].plot(hist["val_loss"], label="Val Loss")
    axs[1].set_title("Training vs Validation Loss"); axs[1].set_xlabel("Epochs"); axs[1].legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()

    # one-vs-rest ROC for every class
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves (one vs rest)"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
