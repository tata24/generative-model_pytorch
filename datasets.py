from torch.utils.data import Dataset
import os
from PIL import Image


class FaceDataset(Dataset):

    def __init__(self, root, transforms=None, target_transform=None):
        super(FaceDataset, self).__init__()
        self.img_list = []
        if os.path.isdir(root):
            self.img_list = [root + img_name for img_name in os.listdir(root)
                             if img_name.endswith('.jpg')]
        self.transforms = transforms
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        if self.transforms is not None:
            img = self.transforms(img)  # 是否进行transforms
        return img

    def __len__(self):
        return len(self.img_list)

