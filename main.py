import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.simple_cnn import SimpleCNN
from data.dataset import get_dataloaders
from utils import save_plots


with open("configs.json", 'r') as f:
    configs = json.load(f)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, dataloader, optimizer, criterion):
    model.train()
    num_correct = 0
    total_loss  = 0

    for _, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

    acc = 100 * num_correct / (configs["batch_size"]*len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, dataloader, criterion):
    model.eval()
    num_correct = 0.0
    total_loss = 0.0

    for _, (images, labels) in enumerate(dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

    acc = 100 * num_correct / (configs["batch_size"]*len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss


def test(model,dataloader):
    model.eval()
    pred_labels = []
    true_labels = []
    
    for _, (images, labels) in enumerate(dataloader):
        images = images.to(DEVICE)

        with torch.inference_mode():
            outputs = model(images)

        outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
        pred_labels.extend(outputs)
        true_labels.extend(labels.cpu().numpy().tolist())
      
    return pred_labels, true_labels


def main():
    train_loader, val_loader, test_loader = get_dataloaders(configs["train_dir"], configs["test_dir"], configs)
    model = SimpleCNN(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs["learning_rate"], weight_decay=configs["weight_decay"])

    best_valacc = 0.0
    best_model = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(configs["epochs"]):
        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_acc, train_loss = train(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            configs["epochs"],
            train_acc,
            train_loss,
            curr_lr))
        
        val_acc, val_loss = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

        if val_acc >= best_valacc:
            best_valacc = val_acc
            best_model = model

    save_plots(train_losses, val_losses, train_accs, val_accs, configs["plots_dir"])
    
    pred_labels, true_labels = test(best_model, test_loader)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels,)
    recall = recall_score(true_labels, pred_labels,)
    f1 = f1_score(true_labels, pred_labels,)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__=="__main__":
    main()