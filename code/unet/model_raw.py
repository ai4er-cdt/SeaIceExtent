""" Implementation of model """
from shared import *
from evaluation import evaluate, dice_loss
from trainpredict.predict import make_predictions
from dataset_preparation import *
from network_structure import UNet
import argparse
import sys
import wandb
from torch import optim


def train_net(net, device, image_type, dir_img,
              epochs: int = 5,
              batch_size: int = 10,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False):
    """Train UNET.
       Parameters:
            net: the model instance.
            device: CPU or CUDA.
            image_type: (string) "sar" or "modis".
            dir_img: (string) path to folder containing tiles.
            epochs: (int) number of epochs to run.
            batch_size: (int) number of tiles to train on at once.
            learning_rate: (float) intitial value before optimisation. 
            val_percent: (float) % of dataset to use for validation.
            save_checkpoint: (boolean) whether to save model versions as the training runs.
            amp: (boolean) enable or disable mixed precision.
        Returns: loss (float) the dice loss of the training run.
    """
    
    # Create dataset
    img_list = create_npy_list(dir_img, image_type)
    # Use this if you want a smaller dataset just to test things with:
    #img_list = small_sample(img_list) # Comment this line out if you want the full set.

    dataset = CustomImageDataset(img_list, False, "dict")
    
    n_val, n_train, train_loader, val_loader = split_data(dataset, val_percent, batch_size, 1)

    # Initialize logging
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    #optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    
    # Begin training
    dir_checkpoint = path_checkpoint
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels of {images.shape}. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                # change number for permute?
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            if epoch == 1:
                dir_checkpoint = create_checkpoint_dir(path_checkpoint, image_type, "raw")
            
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
            print("Using the model for predictions...")
            make_predictions(dir_checkpoint, "raw", image_type, r"{}/test".format(training_tiles[2]), temp_binary, temp_probabilities, metrics = True, save = False)

    return loss

  
def get_args():
    """Default parameters for a typical training run."""
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=10, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--save-checkpoint', '-c', default = True, help='Save a checkpoint file after each validation round')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


def get_mini_args():
    """small, basic implementation of get_args for quicker code tests, not for actual model training."""
    parser = argparse.ArgumentParser(description='small, basic implementation for quicker code tests, not for actual model training')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=1.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--save-checkpoint', '-c', default = False, help='Save a checkpoint file after each validation round')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


def run_training(image_type):
    """Sets up the model and starts the training process.
       Parameter: image_type: (string) "sar" or "modis".
    """
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {processor}')

    if image_type == "sar":
        net = UNet(n_channels=1, n_classes=2, bilinear=True)
    elif image_type == "modis":
        net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    # Train the model with an assortment of different tile sizes. 
    permuted_tile_sizes = permute_tile_sizes()
    net.to(device=processor)
    try:
        train_net(net=net,
                  device=processor,
                  epochs=args.epochs, 
                  image_type="modis",
                  dir_img=permuted_tile_sizes,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  save_checkpoint=args.save_checkpoint,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)


# How to run:
#if __name__ == '__main__':
#    run_training("modis")