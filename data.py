import os
import torch
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToPILImage

# https://public.roboflow.com/object-detection/hands/1
# These values' in egohands dataset is not valid
# rm PUZZLE_LIVINGROOM_B_T_frame_2515_jpg.rf.7ec731ca107b1d44900ba77edf6760d0
# rm CHESS_LIVINGROOM_H_T_frame_2252_jpg.rf.7322645320a1702358af377c6701874e


class VOCDataset(VisionDataset):
    def __init__(self, root, image_set="train", transform=None, target_transform=None):
        super(VOCDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.image_set = image_set
        self.image_dir = os.path.join(root, image_set)
        self.annotations_dir = os.path.join(root, image_set)
        self.images = [
            image for image in os.listdir(self.image_dir) if image.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annotation_path = os.path.join(
            self.annotations_dir, img_name.replace(".jpg", ".xml")
        )
        image = read_image(img_path)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, boxes


def visualize_voc_sample(sample):
    image, boxes = sample
    image = image.permute(1, 2, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        plt.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color="r",
            linewidth=2,
        )
    plt.axis("off")
    plt.show()


class AugmentVOCDataset(VisionDataset):
    def __init__(self, root, image_set="train", transform=None):
        self.dataset = VOCDataset(root=root, image_set=image_set, transform=None)
        self.transform = transform
        self.augmentations = A.Compose(
            [
                A.RandomScale(scale_limit=(0.5, 2.0)),
                A.Resize(192, 192),
                A.HorizontalFlip(p=0.5),
                A.ZoomBlur(p=0.2),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.Blur(p=0.2),
                A.GaussNoise(p=0.2),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, boxes = self.dataset[idx]
        image = np.transpose(image.numpy(), (1, 2, 0))
        boxes = [[box[0], box[1], box[2], box[3]] for box in boxes]
        augmented = self.augmentations(
            image=image, bboxes=boxes, labels=np.zeros(len(boxes))
        )
        image = augmented["image"]
        boxes = np.array(augmented["bboxes"])

        if self.transform:
            image = self.transform(image)

        return image, boxes


class HighEncodedVocDataset(VisionDataset):
    def __init__(self, root, image_set="train", transform=None, encoding_size=11):
        self.dataset = AugmentVOCDataset(root=root, image_set=image_set, transform=None)
        self.transform = transform
        self.encoding_size = encoding_size
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, boxes = self.dataset[idx]
        grid_size = self.encoding_size
        encoding = torch.zeros(grid_size, grid_size)

        image_height, image_width = image.shape[1], image.shape[2]

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin_norm = xmin / image_width
            ymin_norm = ymin / image_height
            xmax_norm = xmax / image_width
            ymax_norm = ymax / image_height
            center_x = (xmin_norm + xmax_norm) / 2
            center_y = (ymin_norm + ymax_norm) / 2
            grid_x = int(center_x * grid_size)
            grid_y = int(center_y * grid_size)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                encoding[grid_y, grid_x] = 1

        image_pil = self.to_pil(image)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil, encoding


def visualize_encoded_sample(sample, encoding_size=11):
    image, encoding = sample
    image = image.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(encoding, cmap="binary", origin="lower")
    plt.title("Encoded Grid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, color="gray", linewidth=0.5)
    plt.xticks(range(encoding_size))
    plt.yticks(range(encoding_size))
    plt.gca().invert_yaxis()

    plt.show()
