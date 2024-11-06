import torch
import os

import datetime
import argparse
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import json
from Dataset import *
from Train_val_func import *
from Models import *


# Function to save the arguments in a json file
def save_args(args, output_dir):
    args_dict = vars(args) if hasattr(args, '__dict__') else args
    with open(f'{output_dir}/args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)
    # Argument parser for configurations
    parser = argparse.ArgumentParser(description='Polyp shape classification using CNN models')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for dataloaders.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--model', type=str, default='densenet3d', help='Type of model to use.')
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
        
        model = model_class(num_slices=3, num_classes=len(dataset_train.get_classes()))

    
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

    # Save the arguments in the results folder
    save_args(args, folder_name)


    # export classes names and label in json file
    with open(f"{folder_name}/classes.json", "w") as f:
        json.dump(dataset_train.get_class_to_idx(), f)
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