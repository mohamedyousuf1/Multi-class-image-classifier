import torch
import os
import random
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import datetime
import argparse
import matplotlib.pyplot as plt
from torchvision import models



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
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_parent_path, transform=False, device="cuda"): 

        self.images_parent_path = images_parent_path
        self.transform = transform
        self.device = device
        #check if the path exists
        if not os.path.exists(self.images_parent_path):
            raise Exception(f"Path {self.images_parent_path} does not exist")
        
        # get subdirectories names and assign them to classes
        self.classes = os.listdir(self.images_parent_path)
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # get all images paths
        self.images = []
        for cls in self.classes:
            class_path = os.path.join(self.images_parent_path, cls)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                # check if the file is an image
                if not image_path.endswith((".jpg", ".jpeg", ".png")):
                    continue
                self.images.append((image_path, self.class_to_idx[cls]))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        # load image if the file is an image
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            raise Exception(f"Could not load image {image_path}")
            # remove the image from the dictionary



        

        if self.transform:
            image = self.apply_transform(image)
        
        # make labels in proper shape for softmax based on the number of classes
        # label = torch.tensor(label).view(1)

        # resize image to 70x70
        image = transforms.functional.resize(image, (70, 70))

        # pil to tensor
        image = transforms.functional.to_tensor(image).to(self.device)





        return image , label

    def get_classes(self):
        return self.classes
    
    def get_class_to_idx(self):
        return self.class_to_idx
    
    def apply_transform(self, image):

        # apply rotation
        if random.random() > 0.5:
            # apply rotation for 0, 90, 180, 270 degrees
            rotation = random.choice([0, 90, 180, 270])
            image = transforms.functional.rotate(image, rotation)

        # apply flip
        if random.random() > 0.5:
            # apply flip horizontally
            image = transforms.functional.hflip(image)
        
        # apply color jitter
        image = transforms.functional.adjust_brightness(image, brightness_factor=random.uniform(0.5, 1.5))
        image = transforms.functional.adjust_contrast(image, contrast_factor=random.uniform(0.5, 1.5))
        image = transforms.functional.adjust_saturation(image, saturation_factor=random.uniform(0.5, 1.5))
        image = transforms.functional.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

        # add affine transformation
        if random.random() > 0.5:
            image = transforms.functional.affine(image, angle=random.uniform(-10, 10), translate=(0, 0), scale=random.uniform(0.9, 1.1), shear=random.uniform(-10, 10))

        return image
    

# Define a more compact feature extractor for each channel
class CompactFeatureExtractor(nn.Module):
    def __init__(self, num_slices=5):
        super(CompactFeatureExtractor, self).__init__()
        # Define layers for compact feature extraction
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1->8 channels
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample 70x70 -> 35x35
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8->16 channels
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample 35x35 -> 17x17
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16->32 channels
        self.pool3 = nn.MaxPool2d(2, 2)  # Downsample 17x17 -> 8x8
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32->64 channels
        self.pool4 = nn.MaxPool2d(2, 2)  # Downsample 8x8 -> 4x4
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64->128 channels
        self.pool5 = nn.MaxPool2d(2, 2)  # Downsample 4x4 -> 2x2
        self.dropout = nn.Dropout(0.30)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 8)  # Reducing fully connected size

    def forward(self, x):
        # Add a channel dimension
        x = F.relu(self.conv1(x))  # Squeeze the channel dimension
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x

# Define the main model with 5 channels (feature extractors) and concatenation
class MultiChannelModel(nn.Module):
    def __init__(self):
        super(MultiChannelModel, self).__init__()        
        # Create a feature extractor for each input slice
        self.feature_extractor1 = CompactFeatureExtractor()
        self.feature_extractor2 = CompactFeatureExtractor()
        self.feature_extractor3 = CompactFeatureExtractor()


        
        # 1D convolution for capturing sequential information across slices
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        
        # Fully connected layers after concatenation
        self.fc_concat1 = nn.Linear(8 * 8, 8)  # Concatenation of 5 channels, each of size 16
        self.fc_concat2 = nn.Linear(8, 3)       # Final output layer for binary classification
    
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)  # Shape: [bs, 1, 70, 70]
        # Process each slice through its feature extractor (extract each slice and pass)
        # normalizing the input to be between 0 and 1 using min and max
        x1 = self.feature_extractor1(inputs[:,:, 0, :, :])  # Slice 1
        x2 = self.feature_extractor2(inputs[:,:, 1, :, :])  # Slice 2
        x3 = self.feature_extractor3(inputs[:,:, 2, :, :])  # Slice 3

        
        # Concatenate the outputs from all feature extractors
        x_concat = torch.stack((x1, x2, x3), dim=1)  # Stack along slice dimension (batch_size, num_slices, features)
        x_conv1d = self.conv1d(x_concat)  # Apply 1D conv across the slice dimension
        x_conv1d = self.fc_concat2(x_conv1d)  # Pass through fully connected layer
        
        # Flatten and pass through fully connected layers
        x = x_conv1d.view(x_conv1d.size(0), -1)
        # output = F.softmax(x, dim=3)
        
        return x
    

class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = torch.nn.Linear(64*17*17, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# use densenet model
class DenseNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(DenseNet3DClassifier, self).__init__()
        # Load a pre-trained DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.densenet.features.conv0 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the classifier layer to output a single value
        self.densenet.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)  # Convert list to tensor
        x = self.densenet(x) # Pass through the DenseNet model
        return x  
    

if __name__ == "__main__":
    # Argument parser for configurations
    parser = argparse.ArgumentParser(description='Polyp shape classification using CNN models')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for dataloaders.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--model', type=str, default='cnn3d', help='Type of model to use.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloaders.')
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'mse', 'focal'], help='Loss function to use.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use.')
    parser.add_argument('--train-path', type=str, default='/media/maali008/TCIA/Classify_polyp/Dataset/train', help='Path to the training data.')
    parser.add_argument('--validation-path', type=str, default='/media/maali008/TCIA/Classify_polyp/Dataset/val', help='Path to the validation data.')
    # parser.add_argument('--class-weight', type=float, default=None, help='Class weight for the loss function.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
    parser.add_argument('--resume', type=str, default=None, help='Path to the model checkpoint to resume training.')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay for the optimizer.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory to save images from')
    args = parser.parse_args()

    device = torch.device(args.device)

    train_images_parent_path = args.train_path
    dataset_train = ImageDataset(train_images_parent_path, transform=True, device=device)

    val_images_parent_path = args.validation_path
    dataset_val = ImageDataset(val_images_parent_path, transform=False, device=device)


    print(f"Number of classes: {len(dataset_train.get_classes())}")
    print(f"Class to index mapping: {dataset_train.get_class_to_idx()}")
    print(f"Number of training images: {len(dataset_train)}")
    print(f"Number of validation images: {len(dataset_val)}")

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size= args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the model
    if args.resume:
        model = torch.load(args.resume)
    else:
        model_dict = {
            # 'cnn3d': CNN3DClassifier,
            'densenet3d': DenseNet3DClassifier,
            # 'resnet3d': ResNet3DClassifier,
            # 'attention': ZoomIn3DCNN,
            # 'densenet_att': DenseNet3DClassifierWithSE,
            # 'mobilenet': MobileNet3DClassifier,
            # 'resnext': ResNeXt3DClassifier,
            # 'wide_resnet': WideResNet3DClassifier,
            # 'shufflenet': ShuffleNet3DClassifier,
            # 'retinanet': EfficientNet3DClassifier,
            # 'inception': Inception3DClassifier,
            # 'vit': ViT3DClassifier,
            # 'mobilenetv3': MobileNetV3,
            # 'sufflenetv2':ShuffleNetv2,
            'MVN_slices': MultiChannelModel,
            # 'MVN_slices_v2': MultiChannelModel_V2,
            # 'SENet3D': SENet2DClassifier,
        }

        # Select the model class based on the argument
        model_class = model_dict.get(args.model)

    if model_class is None:
        raise ValueError(f"Unknown model: {args.model}")    # Define the loss function and optimizer
    
    model.to(device)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create folder to store the model and results
    if args.output_dir is not None:
        folder_name = args.output_dir
    else:
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"results_with_aug/{args.model}/{date_time}_{args.loss}_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}"
        
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(f"Results will be saved in {folder_name}")


    # Save model architecture and summary
    with open(f"{folder_name}/model_summary.txt", "w") as f:
        f.write(str(model))
        f.write('\n')
        # summary_str = summary(model, input_size=(5, 512, 512), device=args.device)
        # f.write(str(summary_str))
        # calculate the number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        f.write(f"Number of parameters: {num_params}")


    # Create csv file to store the results
    results_file = open(f"{folder_name}/results.csv", "w")
    results_file.write("epoch,train_loss,val_loss_balanced,val_acc_balanced,val_recall_balanced,val_precision_balanced,val_f1_balanced\n")
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    train_recalls = []
    train_precisions = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_recalls = []
    val_precisions = []
    val_f1s = []
    val_losses_balannced = []
    val_accuracies_balanced = []
    val_recalls_balanced = []
    val_precisions_balanced = []
    val_f1s_balanced = []

    
    # Train the model
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for epoch in range(args.num_epochs):

        train_loss, train_accuracy, train_recall, train_precision, train_f1, train_all_pred, train_all_label  = train_epoch(model, train_dataloader, criterion, optimizer, epoch, args.num_epochs, device=device)
        
        val_loss_balannced , val_acc_balanced, val_recall_balanced, val_precision_balanced, val_f1_balanced, val_all_preds_rounded_b, all_labels_b   = validate_epoch(model, 
                                                                                                                             val_dataloader, criterion, epoch, args.num_epochs, device= device)
        
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss_balannced:.4f}')
        print(f'Val Acc: {val_acc_balanced:.4f}, Val Recall: {val_recall_balanced:.4f}, Val Precision: {val_precision_balanced:.4f}, Val F1: {val_f1_balanced:.4f}')
        print('--------------------------------------------------')
        results_file.write(f"{epoch+1},{train_loss},{val_loss_balannced},{val_acc_balanced},{val_recall_balanced},{val_precision_balanced},{val_f1_balanced}\n")
        results_file.flush() # Write to file immediately

        # Save the best model instance based on validation loss and F1 score
        if epoch == 0:
            best_loss = val_loss_balannced
            best_f1 = val_f1_balanced
            best_recall = val_recall_balanced
            checkpoint_folder_path = f"{folder_name}/checkpoints"
            if not os.path.exists(checkpoint_folder_path):
                os.makedirs(checkpoint_folder_path)
            torch.save(model, f'{checkpoint_folder_path}/best_loss_balanced.pt')
            torch.save(model, f'{checkpoint_folder_path}/best_f1_balanced.pt')
            torch.save(model, f'{checkpoint_folder_path}/best_recall_balanced.pt')

        else:
            if val_loss_balannced < best_loss:
                best_loss = val_loss_balannced
                torch.save(model, f'{checkpoint_folder_path}/best_loss_balanced.pt')
            if val_f1_balanced > best_f1:
                best_f1 = val_f1_balanced
                torch.save(model, f'{checkpoint_folder_path}/best_f1_balanced.pt')
            if val_recall_balanced > best_recall:
                best_recall = val_recall_balanced
                torch.save(model, f'{checkpoint_folder_path}/best_recall_balanced.pt')
        

        # Store metrics for plotting
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_recalls.append(train_recall)
        train_precisions.append(train_precision)
        train_f1s.append(train_f1)
        val_losses_balannced.append(val_loss_balannced)
        val_accuracies_balanced.append(val_acc_balanced)
        val_recalls_balanced.append(val_recall_balanced)
        val_precisions_balanced.append(val_precision_balanced)
        val_f1s_balanced.append(val_f1_balanced)

        epochs = list(range(0, epoch + 1))

        # Plot metrics
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metrics Val vs Train')
        axs[0, 0].plot(epochs, val_accuracies_balanced, label='Validation Accuracy')
        axs[0, 0].plot(epochs, train_accuracies, label='Train Accuracy')
        axs[0, 0].set_title('Validation Accuracy')
        axs[0, 1].plot(epochs, val_recalls_balanced, label='Validation Recall')
        axs[0, 1].plot(epochs, train_recalls, label='Train Recall')
        axs[0, 1].set_title('Validation Recall')
        axs[1, 0].plot(epochs, val_precisions_balanced, label='Validation Precision')
        axs[1, 0].plot(epochs, train_precisions, label='Train Precision')
        axs[1, 0].set_title('Validation Precision')
        axs[1, 1].plot(epochs, val_f1s_balanced, label='Validation F1 Score')
        axs[1, 1].plot(epochs, train_f1s, label='Train F1 Score')
        axs[1, 1].set_title('Validation F1 Score')
        axs[1, 2].plot(epochs, val_losses_balannced, label='Validation Loss')
        axs[1, 2].plot(epochs, train_losses, label='Train Loss')
        axs[1, 2].set_title('Validation Loss')
        for ax in axs.flat:
            ax.set(xlabel='Epochs', ylabel='Metrics')
            ax.legend()
        # fig.text(0.01, 0.01, json.dumps(args), fontsize=10, ha='left', wrap=True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{folder_name}/metrics_calid_vs_train.png')