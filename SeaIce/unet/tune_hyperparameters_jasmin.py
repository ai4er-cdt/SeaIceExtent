"""This script runs best if you login to wandb through the terminal before running the script using: $wandb login"""

try:
    from dataset_preparation import *
    from evaluation import *
    from network_structure import *
    from mini_network import *
except:
    from unet.dataset_preparation import *
    from unet.evaluation import *
    from unet.network_structure import *
    from unet.mini_network import *

import wandb
import torch.optim as optim


def build_optimiser(network, config):
    """Builds the training optimiser based on the parametersd in the config file"""
    if config.optimiser == "sgd":
        optimiser = optim.SGD(network.parameters(),
                              config.learning_rate, config.momentum)
    elif config.optimiser == "adam":
        optimiser = optim.Adam(network.parameters(),
                               lr=config.learning_rate)
    return optimiser


def train_and_validate(config=None, amp=False, device='cpu'):

    # Inputs for the helper functions
    img_dir = '/home/users/jdr53/tiled512/'
    image_type = 'sar'
    net = UNet(1, 2, True)
    return_type = 'dict'
    workers = 2

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        if image_type == "sar":
            is_single_band = True
        elif image_type == "modis":
            is_single_band = False

        # Loader
        img_list = create_npy_list(img_dir, image_type)

        train_img_list, val_img_list, n_train, n_val = split_img_list(img_list, config.validation_percent)

        train_dataset = CustomImageAugmentDataset(train_img_list, is_single_band, return_type, True)
        validation_dataset = CustomImageAugmentDataset(val_img_list, is_single_band, return_type, False)

        train_loader, val_loader = create_dataloaders(train_dataset, validation_dataset, config.batch_size, workers)

        # Optimiser
        optimiser = build_optimiser(net, config)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', patience=2)  # CHECK goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        criterion = nn.CrossEntropyLoss()
        global_step = 0
        #epoch_step

        # Begin training
        run_loss_train = 0
        run_loss_val = 0
        for epoch in range(config.epochs):
            net.train()
            epoch_loss = 0
            n_batches = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{config.epochs}', unit='img') as pbar:

                for batch in train_loader:
                    images = batch['image']
                    true_masks = batch['mask']

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels of {images.shape}. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    # Adding based on pytorch documentation
                    #optimiser.zero_grad()

                    # Making prediction and calculating loss
                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                        #loss = criterion(masks_pred, true_masks)

                    # Optimisation
                    optimiser.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimiser)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    wandb.log({"Batch Loss, Training": batch_loss}, step=global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    n_batches += 1

                val_score, _ = evaluate(net, val_loader, device)
                print(f'\nVal Score: {val_score}, Epoch: {epoch}')

                #wandb.log({"Batch Loss, Validation": val_score_list[index]}, step=global_step-(len(val_score_list)+index))
                #wandb.log({"Epoch Validation Loss": val_score}) , step=global_step-(len(val_score_list)+index)
                scheduler.step(val_score)

            avg_epoch_training_loss = epoch_loss / n_batches
            #wandb.log({"Epoch Training Loss": avg_epoch_training_loss})
            run_loss_train += avg_epoch_training_loss
            #run_loss_val += val_score
            print('Logging Epoch Scores')
            wandb.log({"Epoch Loss, Training": avg_epoch_training_loss, "Epoch Loss, Validation": val_score, "Epoch": epoch+1}, step=global_step)

        wandb.log({"Run Loss, Training": run_loss_train / config.epochs, "Run Loss, Validation": val_score}, step=global_step)

        #wandb.log({"Run Training Loss": run_loss_train})
        #wandb.log({"Run Validation Loss": run_loss_val})
        #wandb.log({"loss": run_loss_train})

# Improvements:
# Remove Epoch log -- DONE
# Validation metrics are 0.
# Loss in chart is 0 -- DONE



if __name__ == '__main__':

    #wandb.init(project="test-hyptuning")

    # Model Name
    model_name = 'unet'

    # Configuring wandb settings
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
            'min': 0,
            'max': 0.1
        },
        'batch_size': {
            # Uniformly-distributed between 5-15
            'distribution': 'int_uniform',
            'min': 5,
            'max': 10,
        },
        'weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'int_uniform',
            'min': 1e-8,
            'max': 1e-2
        }
    }
    sweep_config['parameters'] = parameters_dict

    # Fixed hyperparamters
    parameters_dict.update({
        'momentum': {
            'value': 0.9},
        'validation_percent': {
            'value': 0.5},
        'img_scale': {
            'value': 0.5},
        'epochs': {
            'value': 10}
    })

    sweep_id = wandb.sweep(sweep_config, project=model_name + "hyp-sweep-jasmin")

    n_tuning = 30
    wandb.agent(sweep_id, function=train_and_validate, count=n_tuning)
