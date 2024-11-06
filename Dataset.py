
import os
import torch
import random
from PIL import Image
from torchvision import transforms


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