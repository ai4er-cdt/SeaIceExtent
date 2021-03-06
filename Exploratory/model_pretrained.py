from shared import *
from dataset_preparation import *


torch.manual_seed(2022)  # Setting random seed so that augmentations can be reproduced.
img_type = "sar"
val_percent = 0.1
batch_size = 1

# Create dataset
img_list = create_npy_list(training_tiles[1], img_type)

if img_type == "sar":
    single_channel = True
    n_channels = 1
elif img_type == "modis":
    single_channel = False
    n_channels = 3

dataset = CustomImageDataset(img_list, single_channel, "values")

_, _, train_loader, val_loader = split_data(dataset, val_percent, batch_size, 1)

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
# Can also specify number of UNet steps and channel numbers.
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type=None, in_channels=n_channels, classes=1, encoder_depth=5)
model = model.double()

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    verbose=True,
    device=processor
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    verbose=True,
    device=processor
)

# train model for n epochs

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
            dir_checkpoint = create_checkpoint_dir(path_checkpoint, img_type, model_type='pretrained')
            torch.save(model, str(dir_checkpoint + 'best_model.pth'))
            print('Model saved!')

        if i == n_epochs/2:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
