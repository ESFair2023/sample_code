from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, path, group):
        super(MyDataset, self).__init__()
        self.path = path
        self.group = group

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize(72),  # 256
            transforms.CenterCrop(64),  # 224
            transforms.ToTensor(),
            normalize,
        ])

        self.class_dict = {'BCC': 0, 'BKL': 1, 'MEL': 2, 'NV': 3, 'unknown': 4, 'VASC': 5} #label dictionary
        self.transform = transform
        self.samples = self._make_dataset() #make dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, self.group

    def _make_dataset(self):
        samples = []
        for class_name in self.class_dict:
            class_dir = os.path.join(self.path, class_name)
            label = self.class_dict[class_name]
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    img_path = os.path.join(class_dir, file_name)
                    samples.append((img_path, label))
        return samples


