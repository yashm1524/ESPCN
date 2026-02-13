import os
import cv2
import random
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader


class SRTrainDataset(IterableDataset):
    def __init__(self, dirpath_lr, dirpath_hr, scaling_factor, patch_size, stride, augment=True):
        """ Training Dataset for GOPRO structure with DATA AUGMENTATION
        
        :param dirpath_lr: path to LR training images directory (e.g., train/LR_x4)
        :param dirpath_hr: path to HR training images directory (e.g., train/HR)
        :param scaling_factor: up-scaling factor to use, should match LR folder name
        :param patch_size: size of sub-images to be extracted from LR images
        :param stride: stride used for extracting sub-images
        :param augment: whether to apply data augmentation (flips, rotations)
        """
        
        self.dirpath_lr = dirpath_lr
        self.dirpath_hr = dirpath_hr
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        
        # Get sorted list of LR and HR images
        self.lr_images = sorted(glob(os.path.join(self.dirpath_lr, "*.png")))
        self.hr_images = sorted(glob(os.path.join(self.dirpath_hr, "*.png")))
        
        # Verify matching pairs
        assert len(self.lr_images) == len(self.hr_images), \
            f"Mismatch: {len(self.lr_images)} LR images vs {len(self.hr_images)} HR images"
        
        print(f"Training dataset: {len(self.lr_images)} image pairs")
        if self.augment:
            print("Data augmentation: ENABLED (flips + rotations)")

    def apply_augmentation(self, lr_crop, hr_crop):
        """ Apply random augmentation to image patches
        
        :param lr_crop: LR patch (H, W)
        :param hr_crop: HR patch (rH, rW)
        :return: augmented lr_crop, hr_crop
        """
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            lr_crop = np.fliplr(lr_crop).copy()
            hr_crop = np.fliplr(hr_crop).copy()
        
        # Random vertical flip (50% chance)
        if random.random() > 0.5:
            lr_crop = np.flipud(lr_crop).copy()
            hr_crop = np.flipud(hr_crop).copy()
        
        # Random 90° rotation (0°, 90°, 180°, 270°)
        k = random.randint(0, 3)
        if k > 0:
            lr_crop = np.rot90(lr_crop, k).copy()
            hr_crop = np.rot90(hr_crop, k).copy()
        
        return lr_crop, hr_crop

    def __iter__(self):
        for lr_path, hr_path in zip(self.lr_images, self.hr_images):
            # Load LR and HR images
            lr_image = Image.open(lr_path).convert('RGB')
            hr_image = Image.open(hr_path).convert('RGB')
            
            # Verify dimensions match scaling factor
            assert hr_image.width == lr_image.width * self.scaling_factor, \
                f"Width mismatch: HR={hr_image.width}, LR*scale={lr_image.width * self.scaling_factor}"
            assert hr_image.height == lr_image.height * self.scaling_factor, \
                f"Height mismatch: HR={hr_image.height}, LR*scale={lr_image.height * self.scaling_factor}"
            
            # Convert to numpy arrays
            lr_image = np.array(lr_image).astype(np.float32)
            hr_image = np.array(hr_image).astype(np.float32)
            
            # Convert RGB to YCbCr
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            
            # Extract Y channel (luminance)
            lr_y = lr_image[:, :, 0]
            hr_y = hr_image[:, :, 0]
            
            # Extract sub-images (patches)
            rows = lr_y.shape[0]
            cols = lr_y.shape[1]
            
            for i in range(0, rows - self.patch_size + 1, self.stride):
                for j in range(0, cols - self.patch_size + 1, self.stride):
                    # LR crop: patch_size x patch_size
                    lr_crop = lr_y[i:i + self.patch_size, j:j + self.patch_size]
                    
                    # HR crop: (patch_size * scaling_factor) x (patch_size * scaling_factor)
                    hr_crop = hr_y[
                        i * self.scaling_factor:i * self.scaling_factor + self.patch_size * self.scaling_factor,
                        j * self.scaling_factor:j * self.scaling_factor + self.patch_size * self.scaling_factor
                    ]
                    
                    # Apply augmentation if enabled
                    if self.augment:
                        lr_crop, hr_crop = self.apply_augmentation(lr_crop, hr_crop)
                    
                    # Normalize to [0, 1]
                    lr_crop = np.expand_dims(lr_crop / 255.0, axis=0)
                    hr_crop = np.expand_dims(hr_crop / 255.0, axis=0)
                    
                    yield lr_crop, hr_crop

    def __len__(self):
        return len(self.lr_images)


class SRValidDataset(IterableDataset):
    def __init__(self, dirpath_lr, dirpath_hr, scaling_factor):
        """ Validation Dataset for GOPRO structure (NO augmentation)
        
        :param dirpath_lr: path to LR validation images directory (e.g., test/LR_x4)
        :param dirpath_hr: path to HR validation images directory (e.g., test/HR)
        :param scaling_factor: up-scaling factor to use
        """
        
        self.dirpath_lr = dirpath_lr
        self.dirpath_hr = dirpath_hr
        self.scaling_factor = scaling_factor
        
        # Get sorted list of LR and HR images
        self.lr_images = sorted(glob(os.path.join(self.dirpath_lr, "*.png")))
        self.hr_images = sorted(glob(os.path.join(self.dirpath_hr, "*.png")))
        
        # Verify matching pairs
        assert len(self.lr_images) == len(self.hr_images), \
            f"Mismatch: {len(self.lr_images)} LR images vs {len(self.hr_images)} HR images"
        
        print(f"Validation dataset: {len(self.lr_images)} image pairs")

    def __iter__(self):
        for lr_path, hr_path in zip(self.lr_images, self.hr_images):
            # Load LR and HR images
            lr_image = Image.open(lr_path).convert('RGB')
            hr_image = Image.open(hr_path).convert('RGB')
            
            # Convert to numpy arrays
            lr_image = np.array(lr_image).astype(np.float32)
            hr_image = np.array(hr_image).astype(np.float32)
            
            # Convert RGB to YCbCr
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            
            # Extract Y channel
            lr_y = lr_image[:, :, 0]
            hr_y = hr_image[:, :, 0]
            
            # Normalize to [0, 1]
            lr_y = np.expand_dims(lr_y / 255.0, axis=0)
            hr_y = np.expand_dims(hr_y / 255.0, axis=0)
            
            yield lr_y, hr_y

    def __len__(self):
        return len(self.lr_images)


# Ref.: https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def get_data_loader(dirpath_train_lr, dirpath_train_hr, dirpath_val_lr, dirpath_val_hr,
                   scaling_factor, patch_size, stride, batch_size=16, augment=True):
    """ Function to return train/val data loader for GOPRO dataset
    
    :param dirpath_train_lr: path to LR training images (e.g., GOPRO_SR/train/LR_x4)
    :param dirpath_train_hr: path to HR training images (e.g., GOPRO_SR/train/HR)
    :param dirpath_val_lr: path to LR validation images (e.g., GOPRO_SR/test/LR_x4)
    :param dirpath_val_hr: path to HR validation images (e.g., GOPRO_SR/test/HR)
    :param scaling_factor: scaling factor (4 for GOPRO dataset)
    :param patch_size: size of sub-images extracted from LR images
    :param stride: sub-images extraction stride
    :param batch_size: training batch size
    :param augment: enable data augmentation for training
    :return: training and validation data loaders
    """
    
    # Training dataset with augmentation
    dataset = SRTrainDataset(
        dirpath_lr=dirpath_train_lr,
        dirpath_hr=dirpath_train_hr,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
        stride=stride,
        augment=augment
    )
    train_dataset = ShuffleDataset(dataset, 1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Validation dataset (no augmentation)
    valid_dataset = SRValidDataset(
        dirpath_lr=dirpath_val_lr,
        dirpath_hr=dirpath_val_hr,
        scaling_factor=scaling_factor
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test DataLoaders with GOPRO structure
    train_loader, val_loader = get_data_loader(
        dirpath_train_lr="./GOPRO_SR/train/LR_x4",
        dirpath_train_hr="./GOPRO_SR/train/HR",
        dirpath_val_lr="./GOPRO_SR/test/LR_x4",
        dirpath_val_hr="./GOPRO_SR/test/HR",
        scaling_factor=4,
        patch_size=17,
        stride=13,
        augment=True
    )
    
    print("\n--- Testing Training Loader ---")
    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"Batch {idx}: LR shape: {lr_image.shape}, HR shape: {hr_image.shape}")
        if idx >= 2:
            break
    
    print("\n--- Testing Validation Loader ---")
    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"Batch {idx}: LR shape: {lr_image.shape}, HR shape: {hr_image.shape}")
        if idx >= 2:
            break
