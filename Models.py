from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

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