import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

class ASLCNNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        all_labels = [label for label in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, label))]

    
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)

       
        for label in all_labels:
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

       
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.encoded_labels[idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def get_label_mapping(self):
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))


