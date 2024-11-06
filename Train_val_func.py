from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch



def train_epoch(model, dataloader, criterion, optimizer, epoch, num_epochs, device, output_dir= None):

    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}/{num_epochs}')

    for sequences, labels in progress_bar:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # output = outputs.view_as(labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.detach().cpu().numpy())
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    epoch_loss = running_loss / len(dataloader)
    all_preds = np.argmax(all_preds, axis=1)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, accuracy, recall, precision, f1, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, epoch, num_epochs, device):
    model.to(device)

    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_preds_rounded = []
    progress_bar = tqdm(dataloader, desc=f'Validation Epoch {epoch+1}/{num_epochs}')

    with torch.no_grad():
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            outputs = torch.softmax(outputs, dim=1)
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            # preds for multilabel classification
            preds = np.argmax(outputs.cpu().numpy(), axis=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
            all_preds_rounded.extend(preds)

        epoch_loss = running_loss / len(dataloader)
        all_preds = np.argmax(all_preds, axis=1)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        return epoch_loss, accuracy, recall, precision, f1, all_preds, all_labels
    
    