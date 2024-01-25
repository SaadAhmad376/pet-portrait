import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split_ratio):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.splt_ratio = split_ratio
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image

    def define_transform(self):

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # Add more transformations if needed
        ])




# Instantiate the dataset
dataset = CustomImageDataset(root_dir='path_to_dataset')

# Splitting the dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, )
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
