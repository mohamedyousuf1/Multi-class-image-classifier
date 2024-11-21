from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch


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
    


# Define a simple 3D CNN model for polyp classificationimport torch

class CNN3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1, dropout_rate=0.3):
        super(CNN3DClassifier, self).__init__()
        
        # Initial convolution with bottleneck
        self.conv1 = nn.Conv2d(num_slices, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bottleneck1 = nn.Conv2d(32, 16, kernel_size=1, padding=0)

        # Second convolution with skip connection
        self.conv2 = nn.Conv2d(16 + 5, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bottleneck2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)

        # Third convolution with skip connection
        self.conv3 = nn.Conv2d(48, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling
        self.pool = nn.MaxPool2d(8, 8)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer initialization
        self._initialize_fc1(num_slices)
        self.fc1 = nn.Linear(self.fc1_input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _initialize_fc1(self, num_slices):
        dummy_input = torch.zeros(1, num_slices, 512, 512)
        dummy_output = self._forward_conv_layers(dummy_input)
        self.fc1_input_size = dummy_output.view(1, -1).size(1)

    def _forward_conv_layers(self, x):
        # rsize the input to match [batch_size, num_slices, 256, 256] instead of [batch_size, num_slices, 512, 512]
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bottleneck1(x1))
        
        x2 = torch.cat((x1, x), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = F.relu(self.bottleneck2(x2))
        
        x3 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        
        return self.pool(x3)

    def forward(self, x):
        x = self._forward_conv_layers(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



class ZoomIn3DCNN(nn.Module):
    def __init__(self, num_slices=5, dropout_rate=0.5):
        super(ZoomIn3DCNN, self).__init__()
        
        # Stage 1: Coarse Attention at Slice Level
        self.coarse_attention = nn.Sequential(
            nn.Conv2d(num_slices, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Stage 2: Fine Attention at Patch Level
        self.fine_attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Convolutional layers
        # First conv layer: Increased filters
        self.conv1 = nn.Conv2d(num_slices, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for stable learning
        
        # Second conv layer: More filters
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third conv layer: Deeper representation
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling to reduce the number of parameters
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 64)  # Increased number of neurons
        self.fc2 = nn.Linear(64, 1)  # Final output layer

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output(self, x):
        """Passes a dummy input through conv and pooling layers to calculate the output size."""
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x.size(1)  # Return the flattened size

    def forward(self, x):
        # Stage 1: Apply Coarse Attention on Slices
        attention_weights = self.coarse_attention(x ) 
        
        # Expand attention weights across channels
        attention_weights = attention_weights.expand(-1, x.size(1), -1, -1)  # Shape becomes [32, 5, 512, 512]

        # Apply attention weights to the input
        x = x.squeeze(2) * attention_weights  
        
        # Reshape input to match Conv2d input expectations
        # x = x.view(x.size(0), -1, x.size(3), x.size(4))  # Shape: [batch_size, 5, 512, 512] -> [batch_size, 5 (channels), 512, 512]
        
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Stage 2: Apply Fine Attention on patches
        fine_attention_weights = self.fine_attention(x)
        x = x * fine_attention_weights
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Final output layer for binary classification

        return x

# Define a more compact feature extractor for each channel
class CompactFeatureExtractor(nn.Module):
    def __init__(self, num_slices=5):
        super(CompactFeatureExtractor, self).__init__()
        # Define layers for compact feature extraction
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1->8 channels
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample 512x512 -> 256x256
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8->16 channels
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample 256x256 -> 128x128
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16->32 channels
        self.pool3 = nn.MaxPool2d(2, 2)  # Downsample 128x128 -> 64x64
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32->64 channels
        self.pool4 = nn.MaxPool2d(2, 2)  # Downsample 64x64 -> 32x32
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64->128 channels
        self.pool5 = nn.MaxPool2d(2, 2)  # Downsample 32x32 -> 16x16
        self.dropout = nn.Dropout(0.30)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 8)  # Reducing fully connected size

    def forward(self, x):
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
    def __init__(self, num_slices=5):
        super(MultiChannelModel, self).__init__()        
        # Create a feature extractor for each input slice
        self.feature_extractor1 = CompactFeatureExtractor()
        self.feature_extractor2 = CompactFeatureExtractor()
        self.feature_extractor3 = CompactFeatureExtractor()
        self.feature_extractor4 = CompactFeatureExtractor()
        self.feature_extractor5 = CompactFeatureExtractor()

        
        # 1D convolution for capturing sequential information across slices
        self.conv1d = nn.Conv1d(in_channels=5, out_channels=8, kernel_size=3, padding=1)
        
        # Fully connected layers after concatenation
        self.fc_concat1 = nn.Linear(8 * 8, 8)  # Concatenation of 5 channels, each of size 16
        self.fc_concat2 = nn.Linear(8, 1)       # Final output layer for binary classification
    
    def forward(self, inputs):
        # Process each slice through its feature extractor (extract each slice and pass)
        # normalizing the input to be between 0 and 1 using min and max
        x1 = self.feature_extractor1(inputs[:, 0, :, :, :])  # Slice 1
        x2 = self.feature_extractor2(inputs[:, 1, :, :, :])  # Slice 2
        x3 = self.feature_extractor3(inputs[:, 2, :, :, :])  # Slice 3
        x4 = self.feature_extractor4(inputs[:, 3, :, :, :])  # Slice 4
        x5 = self.feature_extractor5(inputs[:, 4, :, :, :])  # Slice 5
        
        # Concatenate the outputs from all feature extractors
        x_concat = torch.stack((x1, x2, x3, x4, x5), dim=1)  # Stack along slice dimension (batch_size, num_slices, features)
        x_conv1d = self.conv1d(x_concat)  # Apply 1D conv across the slice dimension
        
        # Flatten and pass through fully connected layers
        x = x_conv1d.view(x_conv1d.size(0), -1)
        x = F.relu(self.fc_concat1(x))
        output = self.fc_concat2(x)  # Binary classification output
        
        return output

# Define a more compact feature extractor for each channel with BatchNorm and improved architecture
class CompactFeatureExtractor(nn.Module):
    def __init__(self, num_slices=5):
        super(CompactFeatureExtractor, self).__init__()
        # Define layers for compact feature extraction
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1->8 channels
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample 512x512 -> 256x256
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8->16 channels
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample 256x256 -> 128x128
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16->32 channels
        self.pool3 = nn.MaxPool2d(2, 2)  # Downsample 128x128 -> 64x64
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32->64 channels
        self.pool4 = nn.MaxPool2d(2, 2)  # Downsample 64x64 -> 32x32
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64->128 channels
        self.pool5 = nn.MaxPool2d(2, 2)  # Downsample 32x32 -> 16x16
        self.dropout = nn.Dropout(0.30)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 8)  # Reducing fully connected size
        self.fc2 = nn.Linear(8, 1)  # Final output layer

    def forward(self, x):
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
        x = self.fc1(x)
        x_out = self.fc2(x)
        return x, x_out

# Define the main model with 5 channels (feature extractors) and concatenation


# Use DenseNet model
class DenseNet3DClassifier_1ch(nn.Module):
    def __init__(self, num_slices=5, num_classes=3):  # Adjust num_classes as needed
        super(DenseNet3DClassifier_1ch, self).__init__()
        # Load a pre-trained DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.densenet.features.conv0 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the classifier layer to output the correct number of classes
        self.densenet.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)  # Convert list to tensor
        x = self.densenet(x)  # Pass through the DenseNet model
        return x

class MultiChannelModel_V2(nn.Module):
    def __init__(self, num_slices=5, num_classes = 1):
        super(MultiChannelModel_V2, self).__init__()        
        # Create a feature extractor for each input slice
        self.feature_extractor1 = DenseNet3DClassifier_1ch( num_slices=1, num_classes= num_classes)
        self.feature_extractor2 = DenseNet3DClassifier_1ch( num_slices=1, num_classes= num_classes)
        self.feature_extractor3 = DenseNet3DClassifier_1ch( num_slices=1, num_classes= num_classes)

    def forward(self, inputs):
        # Process each slice through its feature extractor
        x1_out = self.feature_extractor1(inputs[:, 0, :, :].unsqueeze(1))  # Slice 1
        x2_out = self.feature_extractor2(inputs[:, 1, :, :].unsqueeze(1))
        x3_out = self.feature_extractor3(inputs[:, 2, :, :].unsqueeze(1))


        # Stack the outputs from all feature extractors
        output = torch.stack((x1_out, x2_out, x3_out), dim=1)
        
        # Apply softmax to get probabilities
        # probs = torch.softmax(output, dim=2)  # Shape: [batch_size, num_slices, num_classes]

        # Average the probabilities across the slices
        avg_probs = torch.mean(output, dim=1)  # Shape: [batch_size, num_classes]

        return avg_probs

    

# add attention to densenet model
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global average pooling
        se = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, channels)
        se = F.relu(self.fc1(se))
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch_size, channels, 1, 1)
        return x * se  # Scale the original input

class DenseNet3DClassifierWithSE(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(DenseNet3DClassifierWithSE, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(1024, num_classes)

        # Add SE blocks after each denseblock
        self.se_block1 = SEBlock(256)  # Adjust input channels based on the denseblock outputs
        self.se_block2 = SEBlock(512)
        self.se_block3 = SEBlock(1024)

    def forward(self, x):
        x = self.densenet.features.conv0(x.squeeze(2))
        x = self.densenet.features.norm0(x)
        x = F.relu(x)
        x = self.densenet.features.pool0(x)

        x = self.densenet.features.denseblock1(x)
        x = self.se_block1(x)  # Apply SE block after denseblock1

        x = self.densenet.features.transition1(x)
        x = self.densenet.features.denseblock2(x)
        x = self.se_block2(x)  # Apply SE block after denseblock2

        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        x = self.se_block3(x)  # Apply SE block after denseblock3

        x = self.densenet.features.transition3(x)
        x = self.densenet.features.denseblock4(x)

        x = self.densenet.features.norm5(x)
        x = F.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.densenet.classifier(x)
        return x

# use resnet model
class ResNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(ResNet3DClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.resnet.conv1 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the classifier layer to output a single value
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the model
        
    
# use vgg model
class VGG3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(VGG3DClassifier, self).__init__()
        # Load a pre-trained VGG model
        self.vgg = models.vgg16(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.vgg.features[0] = nn.Conv2d(num_slices, 64, kernel_size=3, padding=1)
        
        # Replace the classifier layer to output a single value
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the




    

# use mobilenet model
class MobileNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(MobileNet3DClassifier, self).__init__()
        # Load a pre-trained MobileNet model
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.mobilenet.features[0][0] = nn.Conv2d(num_slices, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace the classifier layer to output a single value
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the
    
# use squeezenet model
class SqueezeNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(SqueezeNet3DClassifier, self).__init__()
        # Load a pre-trained SqueezeNet model
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.squeezenet.features[0] = nn.Conv2d(num_slices, 96, kernel_size=7, stride=2)
        
        # Replace the classifier layer to output a single value
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.squeezenet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the
    
# use resnext model
class ResNeXt3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(ResNeXt3DClassifier, self).__init__()
        # Load a pre-trained ResNeXt model
        self.resnext = models.resnext50_32x4d(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.resnext.conv1 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the classifier layer to output a single value
        self.resnext.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnext(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the
    
# use wide resnet model
class WideResNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(WideResNet3DClassifier, self).__init__()
        # Load a pre-trained Wide ResNet model
        self.wideresnet = models.wide_resnet50_2(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.wideresnet.conv1 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the classifier layer to output a single value
        self.wideresnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.wideresnet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the
    
# use shufflenet model
class ShuffleNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(ShuffleNet3DClassifier, self).__init__()
        # Load a pre-trained ShuffleNet model
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        
        # Replace the first convolution layer to adjust for the number of input channels
        self.shufflenet.conv1[0] = nn.Conv2d(num_slices, 24, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace the classifier layer to output a single value
        self.shufflenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.shufflenet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the


# use retinanet model
class EfficientNet3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(EfficientNet3DClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.features[0][0] = nn.Conv2d(num_slices, 32, kernel_size=3, stride=2, padding=1)
        self.efficientnet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.efficientnet(x.squeeze(2))

# use inception model
class Inception3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(Inception3DClassifier, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(num_slices, 32, kernel_size=3, stride=2)

        # Replace the final fully connected layer for classification
        self.inception.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.inception(x.squeeze(2))

from torchvision.models import vit_b_16

class ViT3DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):
        super(ViT3DClassifier, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        
        # Replace the projection layer for the number of input channels
        self.vit.conv_proj = nn.Conv2d(num_slices, self.vit.conv_proj.out_channels, kernel_size=16, stride=16)
        
        # Replace the final classification layer
        self.vit.heads[-1] = nn.Linear(self.vit.heads[-1].in_features, num_classes)

    def forward(self, x):
        x = F.interpolate(x.squeeze(2), size=(224, 224), mode='bilinear', align_corners=False)

        return self.vit(x.squeeze(2))




class MobileNetV3(nn.Module):
    def __init__(self, num_slices, num_classes=1):
        super(MobileNetV3, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(num_slices, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
    
    def forward(self, x):
        return self.mobilenet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the model

class ShuffleNetv2(nn.Module):
    def __init__(self, num_slices, num_classes = 1):
        super(ShuffleNetv2, self).__init__()
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        self.shufflenet.conv1[0] = nn.Conv2d(num_slices, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.shufflenet.fc = nn.Linear(self.shufflenet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.shufflenet(x.squeeze(2))  # Squeeze the second dimension (num_slices) and pass through the model


class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock2D, self).__init__()
        
        # Global average pooling, followed by two fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze operation: global average pooling
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        
        # Fully connected layers with reduction and sigmoid activation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        
        # Excitation operation: scale the input
        return x * y.expand_as(x)

# Define a 3D CNN model with Squeeze-and-Excitation blocks
class SENet2DClassifier(nn.Module):
    def __init__(self, num_slices=5, num_classes=1):  # num_slices used as channels
        super(SENet2DClassifier, self).__init__()
        
        # Use Conv2d, with num_slices as input channels
        self.conv1 = nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Replace SEBlock3D and AdaptiveAvgPool3d with 2D versions
        self.se_block = SEBlock2D(64)  # Assume SEBlock2D exists or modify SEBlock3D accordingly
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Forward pass through conv1 and bn1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.se_block(x)
        
        # Pool and classify
        x = self.pool(x).view(x.size(0), -1)  # Flatten before fc layer
        return self.fc(x)
