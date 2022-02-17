
import glob
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from SeaIce.unet.shared import *
from SeaIce.unet.dataset_preparation import CustomImageDataset

torch.manual_seed(2022)  # Setting random seed so that augmentations can be reproduced.
dir_img = Path(r'G:\Shared drives\2021-gtc-sea-ice\trainingdata\tiled')

def create_npy_list(image_directory, img_string):
    """A function that returns a list of the names of the SAR/MODIS and labelled .npy files in a directory. These lists can
    then be used as an argument for the Dataset class instantiation. The function also checks that the specified directory
    contains matching sar or MODIS/labelled pairs -- specifically, a label.npy file for each image file."""

    img_names = sorted(glob.glob(str(image_directory) + r'/*_' + img_string + r'.npy'))
    label_names = sorted(glob.glob(str(image_directory) + r'/*_labels.npy'))
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


val_percent = 0.1
batch_size = 10

# 1. Create dataset
img_list = create_npy_list(dir_img, r"modis")
dataset = CustomImageDataset(img_list, True)

# 2. Split into train / validation partitions
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
# Can also specify number of UNet steps and channel numbers.
model = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type=None, in_channels=1, classes=1,
                 activation='sigmoid', aux_params={'classes': 1,'pooling': 'max'})
model = model.double()

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

DEVICE = 'cpu'

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    verbose=True,
    device=DEVICE
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    verbose=True,
    device=DEVICE
)

# train model for 40 epochs

max_score = 0
n_epochs = 1

if __name__ == '__main__':
    for i in range(0, n_epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == n_epochs/2:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
