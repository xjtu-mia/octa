import os
import natsort
from PIL import Image  # PIL（HWC）与torch（CHW）的设置 transforms.ToTensor()(img) transforms.ToPILImage()(img)
import torch
import torchvision
from torchvision import transforms
import torch.utils.data
from torch.utils.data import Dataset
from torch.autograd import Variable


class Images_Dataset(Dataset):
    def __init__(self, train_dir, label_dir,):
        """
        class for getting data
        :param train_dir: path of train images
        :param label_dir: path of label images
        """
        self.train_dir = train_dir
        self.train_A_dir = os.path.join(train_dir, 'data_A')
        self.train_B_dir = os.path.join(train_dir, 'data_B')
        self.train_C_dir = os.path.join(train_dir, 'data_C')
        self.train_D_dir = os.path.join(train_dir, 'data_D')

        self.label_dir = label_dir
        self.train_A_images = natsort.natsorted(os.listdir(self.train_A_dir))
        self.train_B_images = natsort.natsorted(os.listdir(self.train_B_dir))
        self.train_C_images = natsort.natsorted(os.listdir(self.train_C_dir))
        self.train_D_images = natsort.natsorted(os.listdir(self.train_D_dir))

        self.label_images = natsort.natsorted(os.listdir(label_dir))

    def __len__(self):
        # print(f'Train images number is {len(self.train_images)};')
        # print(f'Label images number is {len(self.label_images)};')
        return len(self.train_A_images)

    def __getitem__(self, idx):
        train_A_image = Image.open(os.path.join(self.train_A_dir, self.train_A_images[idx])).convert('L')
        train_B_image = Image.open(os.path.join(self.train_B_dir, self.train_B_images[idx])).convert('L')
        train_C_image = Image.open(os.path.join(self.train_C_dir, self.train_C_images[idx])).convert('L')
        train_D_image = Image.open(os.path.join(self.train_D_dir, self.train_D_images[idx])).convert('L')

        label_image = Image.open(os.path.join(self.label_dir, self.label_images[idx]))

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # 将HWC转换成CHW, 再转换为float型，再除255
                # transforms.Normalize(mean=[0.5], std=[0.5])
            ]  # Normalize归一化数据
        )

        label_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5]) #为什么label 不用归一化？
             ]
        )

        train_A_image = train_transform(train_A_image)
        train_B_image = train_transform(train_B_image)
        train_C_image = train_transform(train_C_image)
        train_D_image = train_transform(train_D_image)

        train_image = torch.cat((train_A_image, train_B_image, train_C_image, train_D_image), dim=0)
        label_image = label_transform(label_image)

        return train_image, label_image


if __name__ == '__main__':
    print("My Data Loder")
