"""This script runs best if you login to wandb through the terminal before running the script using: $wandb login"""

try:
    from dataset_preparation import *
    from evaluation import *
    from network_structure import *
    from shared import *
except:
    from unet.dataset_preparation import *
    from unet.evaluation import *
    from unet.network_structure import *
    from unet.shared import *

import wandb
import torch.optim as optim
from pathlib import Path
import os
from datetime import datetime


def create_checkpoint_dir(folder_checkpoint, img_type, model_type, optimizer, lr, batchsize, weightdecay):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    dir_list = os.listdir(folder_checkpoint)
    FOLDER_EXISTS = True
    x = 1
    while FOLDER_EXISTS:
        dir_checkpoint = str('{}_{}_optm{}_batchsize{}_learnrate{}_weightdecay{}_date{}_{}'.format(model_type, img_type,
                                                                                                   str(optimizer),
                                                                                                   str(batchsize),
                                                                                                   str(lr),
                                                                                                   str(weightdecay),
                                                                                                   str(dt_string),
                                                                                                   str(x)))
        path = os.path.join(folder_checkpoint, Path(dir_checkpoint))
        if dir_checkpoint in dir_list:
            x += 1
        else:
            FOLDER_EXISTS = False
            os.mkdir(path)
            # print("Checkpoint directory {} created.".format(str(path)))
            return (path)


def build_optimiser(network, config):
    """Builds the training optimiser based on the parametersd in the config file"""
    if config.optimiser == "sgd":
        optimiser = optim.SGD(network.parameters(),
                              config.learning_rate, config.momentum)
    elif config.optimiser == "adam":
        optimiser = optim.Adam(network.parameters(),
                               lr=config.learning_rate)
    return optimiser


def train_and_validate(config=None, amp=False, device=torch.device('cuda')):

    # SET: Conditions for the run.
    img_dir = '/home/users/jdr53/tiled1024/train/'
    image_type = 'sar'
    return_type = 'dict'
    workers = 4 # MUST BE EVEN
    save_checkpoint = False
    folder_checkpoint = '/home/users/jdr53/hyp_tuning/checkpoints/'
    model_type = 'unet'
    num_output_channels = 1
    loss_function = "BCE"

    # Setting the device to GPU if available or CPU if not.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creating the network
    if image_type == "sar":
        is_single_band = True
        net = UNet(1, num_output_channels, True)
    elif image_type == "modis":
        is_single_band = False
        net = UNet(3, num_output_channels, True)
    else:
        raise Exception(f'Invalid image type detected: {image_type}. Expected "sar" or "modis".')

    # Loading the network to the device (GPU or CPU)
    net.to(device)

    # Initialising a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below, this config will be set by Sweep Controller
        config = wandb.config

        # Getting list of image-mask pairs
        img_list = create_npy_list(img_dir, image_type)

        # Splitting the list into a training and validation set.
        train_img_list, val_img_list, n_train, n_val = split_img_list(img_list, config.validation_percent)

        # Creating training and validation datasets; the training dataset is augmented, the validation set it not.
        train_dataset = CustomImageAugmentDataset(train_img_list, is_single_band, return_type, True)
        validation_dataset = CustomImageAugmentDataset(val_img_list, is_single_band, return_type, False)

        # Creating dataloaders for each dataset.
        train_loader, val_loader = create_dataloaders(train_dataset, validation_dataset, config.batch_size, workers)

        # Configuring the optimiser and learning rate scheduler
        optimiser = build_optimiser(net, config)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', patience=2)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        global_step = 0

        # Begin training
        run_loss_train = 0
        # Selecting form of the cross-entropy loss term
        if loss_function == "BCE":
            criterion = nn.BCELoss()
        elif loss_function == "CE":
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception(f'Invalid loss_function detected: {loss_function}. Expected "BCE" or "CE".')

        # Iterating training and validation over epochs.
        for epoch in range(config.epochs):

            net.train() # Setting the network to train
            epoch_loss = 0
            n_batches = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{config.epochs}', unit='img') as pbar:

                # Iterating over training batches
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
                        # Making predictions
                        mask_pred = net(images)
                        # If 1 channel output is required
                        if net.n_classes == 1:
                            # Applying a sigmoid function to the true and predicted masks.
                            true_masks = (torch.sigmoid(true_masks) > 0.5).float()
                            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                            mask_pred.requires_grad = True
                            # Adding a dummy dimension to match with the shape of the predicted masks
                            true_masks = true_masks[:, None, :, :]
                            # Computing the loss
                            loss = dice_loss(mask_pred, true_masks, multiclass=False,
                                             epsilon=config.weight_decay) + criterion(mask_pred, true_masks)
                        # If 2 channel output is required
                        else:
                            # One-hot encoding the masks
                            true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                            if loss_function == "BCE":
                                mask_pred = F.one_hot(mask_pred.argmax(dim=1),
                                                      net.n_classes).permute(0, 3, 1, 2).float()
                                mask_pred.requires_grad = True
                            else:
                                mask_pred = F.softmax(mask_pred, dim=1).float()
                            # # Computing the loss
                            loss = dice_loss(mask_pred, true_masks, multiclass=True,
                                             epsilon=config.weight_decay) + criterion(mask_pred, true_masks)

                    # Optimisation & Training
                    optimiser.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimiser)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    # Logging batch loss
                    wandb.log({"Batch Loss, Training": batch_loss}, step=global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    n_batches += 1

                # Validation
                val_score = evaluate(net, val_loader, device, config.weight_decay, loss_function)

                print(f'\nVal Score: {val_score}, Epoch: {epoch}')
                scheduler.step(val_score)

            # Calculating epoch training loss
            avg_epoch_training_loss = epoch_loss / n_batches
            run_loss_train += avg_epoch_training_loss
            # run_loss_val += val_score
            print('Logging Epoch Scores')
            wandb.log({"Epoch Loss, Training": avg_epoch_training_loss, "Epoch Loss, Validation": val_score,
                       "Epoch": epoch + 1}, step=global_step)

            # Updating best epoch if loss score is lower than current best
            if epoch == 0:
                best_run_loss_val = val_score
                best_epoch = epoch
            else:
                if val_score < best_run_loss_val:
                    best_run_loss_val = val_score
                    best_epoch = epoch
            # Saving model checkpoint
            if save_checkpoint:
                if epoch == 0:
                    dir_checkpoint = create_checkpoint_dir(folder_checkpoint, image_type, model_type, config.optimiser,
                                                           config.learning_rate, config.batch_size, config.weight_decay)
                torch.save(net.state_dict(), dir_checkpoint + f'/checkpoint_epoch{epoch + 1}.pth')

        # Logging the training and validations losses for the entire run
        wandb.log({"Run Loss, Training": run_loss_train / config.epochs, "Run Loss, Validation": best_run_loss_val,
                   "Run Best Epoch": best_epoch}, step=global_step)


if __name__ == '__main__':

    # Configuring wandb sweep settings
    sweep_config = {
        'method': 'random',  # Random search method -- less computationally expensive yet effective.
    }

    metric = {
        'name': 'Run Loss, Validation',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    # Tuned hyperparameters
    parameters_dict = {
        'optimiser': {
            'values': ['adam', 'sgd']
        },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.00001,
            'max': 0.1
        },
        'batch_size': {
            # Uniformly-distributed between 5-15
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5
        }
    }
    sweep_config['parameters'] = parameters_dict

    # Fixed hyperparamters
    parameters_dict.update({
        'momentum': {
            'value': 0.9},
        'validation_percent': {
            'value': 0.2},
        'img_scale': {
            'value': 0.5},
        'epochs': {
            'value': 20},
        'weight_decay': {
            'value': 1e-8}
    })

    sweep_id = wandb.sweep(sweep_config, project="jasmin_gpu_15_03_tuning")

    # Number of runs
    n_tuning = 1000

    # Initialising the wandb agent
    wandb.agent(sweep_id, function=train_and_validate, count=n_tuning)





