""" CNN Dataset preparation functions """
from unet.shared import *
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
import glob
from datetime import datetime
import random

torch.manual_seed(2022) # Setting random seed so that augmentations can be reproduced.

# From https://discuss.pytorch.org/t/beginner-how-do-i-write-a-custom-dataset-that-allows-me-to-return-image-and-its-target-image-not-label-as-a-pair/13988/4
# And https://discuss.pytorch.org/t/how-make-customised-dataset-for-semantic-segmentation/30881


def create_npy_list(image_directory, img_string):
    """A function that returns a list of the names of the SAR/MODIS and labelled .npy files in a directory. These lists can
    then be used as an argument for the Dataset class instantiation. The function also checks that the specified directory 
    contains matching sar or MODIS/labelled pairs -- specifically, a label.npy file for each image file."""

    img_names = sorted(glob.glob(str(image_directory) + '/*_' + img_string + '.npy'))
    label_names = sorted(glob.glob(str(image_directory) + '/*_labels.npy'))

    # In-depth file-by-file check for matching sar-label pairs in the directory -- assuming  each sar image has a corresponding
    # labeled image.
    img_label_pairs = []
    for image in img_names:
        expected_label_name = image.replace(img_string, "labels")
        if expected_label_name in label_names:
            img_label_pairs.append((image, expected_label_name))
        else:
            raise Exception(f'{img_string} tile name {image} does not have a matching labeled tile.')
   
    return img_label_pairs


def small_sample(dataset):
    """A sample 1/10 the size of the dataset for faster code development.
       Parameter: dataset: (list) file paths of all data in the set.
       Returns: small_set: (list) 10 % of those files, randomly selected.
    """
    tenth = round(len(dataset)/10)
    random.shuffle(dataset)
    small_set = dataset[0:tenth]
    return small_set


def split_data(dataset, val_percent, batch_size, workers):
    """Split dataset into training and validation sets.
       Parameters: 
            dataset: (list) paths of all numpy tile files in set.
            val_percent: (numerical type) % of data to use for validation.
            batch_size: (int) number of tiles per batch.
            workers: (int) parallelism factor (CPU or GPU).
       Returns:
            n_val: (int) number of tiles in validation set.
            n_train: (int) number of tiles in training set.
            train_loader: (dataloader object) to train with.
            val_loader: (dataloader object) to validate with.
    """
    # Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    return n_val, n_train, train_loader, val_loader


def split_img_list(img_list, val_percent):
    """Splits the img list pairs (created in create_npy_list) into separate sets for training and validation.
    Inputs: img_list = a 2-column list containing the image-label pairs.
            val_percent = percentage of entire dataset used for validation.
    Outputs: train_img_list = list containing the img_list for the training data.
             val_img_list = list containing the img_list for the validation data.
             n_train = number of images in the training dataset.
             n_val = number of images in the validation dataset.
    """
    random.seed(2022)

    n_val = int(len(img_list) * val_percent)
    n_train = len(img_list) - n_val
    random.shuffle(img_list)

    train_img_list = img_list[n_val:]
    val_img_list = img_list[:n_val]

    return train_img_list, val_img_list, n_train, n_val


def create_dataloaders(train_dataset, val_dataset, batch_size, workers):
    """Creates dataloaders for the separate train and validation datasets.
    Inputs: train_dataset = training dataset class with augmentation.
            val_dataset = validation dataset class with no augmentation.
            batch_size = dataloader batch size.
            workers = number of parallel workers.
    Outputs: train_loader = training dataset loader.
             val_loader = validation dataset loader.
    """
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    return train_loader, val_loader


class CustomImageDataset(Dataset):
    """GTC Code for a dataset class. The class is instantiated with list of filenames within a directory (created using
    the list_npy_filenames function). The __getitem__ method pairs up corresponding image-label .npy file pairs. This
    dataset can then be input to a dataloader. return_type = "values" or "dict"."""
    
    def __init__(self, paths, is_single_band, return_type):
        self.paths = paths
        self.is_single_band = is_single_band
        self.return_type = return_type
    
    def __getitem__(self, index):
        image = torch.from_numpy(np.vstack(np.load(self.paths[index][0])).astype(float))
        if self.is_single_band:
            image = image[None,:]
        else:
            #image = torch.permute(image, (3, 1, 2, 0))
            image = torch.permute(image, (2, 0, 1))
        mask_raw = (np.load(self.paths[index][1]))
        maskremap100 = np.where(mask_raw == 100, 0, mask_raw)
        maskremap200 = np.where(maskremap100 == 200, 1, maskremap100)
        mask = torch.from_numpy(np.vstack(maskremap200).astype(float))

        #assert image.size == mask.size, \
        #    'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        if self.return_type == "dict":
            return {'image': image, 'mask': mask}
        elif self.return_type == "values":
            mask = mask[None, :]
            return image, mask


    def __len__(self):
        return len(self.paths) 


class CustomImageAugmentDataset(Dataset):
    """GTC Code for an augmented dataset class. The class is instantiated with a list of filenames within a directory
    (created using the list_npy_filenames function). The __getitem__ method pairs up corresponding image-label .npy file
    pairs. If specified, augmentations are also applied to the images with a probability. There is a ~25% chance that
    no augmentations are applied and a 20% chance of each of the following augmentations: horizontal flip, vertical
    flip, 90 degree rotation (anti-clockwise & clockwise), 180 degree rotation, random crop. Multiple augmentations are
    applied in sequence. This dataset can then be input to a dataloader."""

    def __init__(self, paths, is_single_band, return_type, augmentation):
        self.paths = paths
        self.is_single_band = is_single_band
        self.return_type = return_type
        self.augmentation = augmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = torch.from_numpy(np.vstack(np.load(self.paths[index][0])).astype(float))
        if self.is_single_band:
            image = image[None,:]
        else:
            #image = torch.permute(image, (3, 1, 2, 0))
            image = torch.permute(image, (2, 0, 1))
        mask_raw = (np.load(self.paths[index][1]))
        maskremap100 = np.where(mask_raw == 100, 0, mask_raw)
        maskremap200 = np.where(maskremap100 == 200, 1, maskremap100)
        mask = torch.from_numpy(np.vstack(maskremap200).astype(float))

        if self.augmentation:
            image, mask = self.augment_image(image, mask)

        #assert image.size == mask.size, \
        #    'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        if self.return_type == "dict":
            return {'image': image, 'mask': mask}
        elif self.return_type == "values":
            mask = mask[None, :]
            return image, mask

    @staticmethod
    def augment_image(original_image, original_mask):

        augmented_image = original_image
        augmented_mask = original_mask

        aug_probabilities = np.random.choice(a=[0,1], size=6, p=[0.8, 0.2])

        if aug_probabilities[0]: # Horizontal flip
            augment_function = transforms.RandomHorizontalFlip(p=1)
            augmented_image, augmented_mask = augment_function(augmented_image), augment_function(augmented_mask)

        if aug_probabilities[1]: # Horizontal flip
            augment_function = transforms.RandomVerticalFlip(p=1)
            augmented_image, augmented_mask = augment_function(augmented_image), augment_function(augmented_mask)

        if aug_probabilities[2]: # 90 degree rotation anti-clockwise
            augmented_image = torch.rot90(augmented_image, k=1, dims=[1, 2])
            augmented_mask = torch.rot90(augmented_mask, k=1, dims=[0, 1])

        if aug_probabilities[3]: # 180 degree rotation
            augmented_image = torch.rot90(augmented_image, k=2, dims=[1, 2])
            augmented_mask = torch.rot90(augmented_mask, k=2, dims=[0, 1])

        if aug_probabilities[4]: # 90 degree rotation clockwise
            augmented_image = torch.rot90(augmented_image, k=-1, dims=[1, 2])
            augmented_mask = torch.rot90(augmented_mask, k=-1, dims=[0, 1])

        if False:#aug_probabilities[5]: # Random crop (and resize)
            augment_function = transforms.Compose([transforms.RandomCrop(size=256),
                                                   transforms.Resize(512)])
            augmented_image, augmented_mask = augment_function(augmented_image), augment_function(augmented_mask)

        """
        if aug_probabilities[6]: # Gaussian blur
            augment_function = transforms.GaussianBlur(kernel_size=(7,13), sigma=(0.1, 0.2))
            augmented_image, augmented_mask = augment_function(augmented_image), augment_function(augmented_mask)
        """
        return augmented_image, augmented_mask

    
def create_checkpoint_dir(path_checkpoint, img_type, model_type):
    """ A function that checks for a custom directory based on a model type, image type and if
    that directory does not exist, creates it and returns the value for use in checkpoint saving.
    Input img_type can be sar or modis. Includes the datetime in the string name.""" 
    
    
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    dir_list = os.listdir(path_checkpoint)
    FOLDER_EXISTS = True
    x = 1
    
    while FOLDER_EXISTS: 
        dir_checkpoint_name = str('{}_{}_{}_{}/'.format(img_type, model_type, dt_string, str(x)))
        path = os.path.join(path_checkpoint, Path(dir_checkpoint_name))
        if dir_checkpoint_name in dir_list:
           x += 1
        else:
          FOLDER_EXISTS = False
          os.mkdir(path)
          print("Checkpoint directory {} created.".format(str(path)))
          return(path)