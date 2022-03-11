"""This script runs best if you login to wandb through the terminal before running the script using: $wandb login"""

try:
    from dataset_preparation import *
    from evaluation import *
    from network_structure import *
except:
    from unet.dataset_preparation import *
    from unet.evaluation import *
    from unet.network_structure import *

import wandb
import torch.optim as optim
from pathlib import Path
import os
from datetime import datetime

def create_checkpoint_dir(parent_folder_checkpoint, img_type, model_type,
                         encoder, optimiser, lr, batchsize, weightdecay):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    dir_list = os.listdir(parent_folder_checkpoint)
    FOLDER_EXISTS = True
    x = 1 
    while FOLDER_EXISTS: 
        dir_checkpoint = str('{}_{}_encode{}_optm{}_batchsize{}_learnrate{}_weightdecay{}_date{}_{}'.format(model_type, img_type, 
                                                   str(encoder), str(optimiser), str(batchsize), str(lr), 
                                                   str(weightdecay), str(dt_string), str(x)))
        path = os.path.join(parent_folder_checkpoint, Path(dir_checkpoint))
        if dir_checkpoint in dir_list:
            x += 1
        else:
            FOLDER_EXISTS = False
            os.mkdir(path)
            return(path)
        

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

    # Inputs for the helper functions
    # Image and model params
    img_dir = '/home/users/mcl66/SeaIce/tiled512/'
    #img_dir = '/mnt/g/Shared drives/2021-gtc-sea-ice/trainingdata/tiled512/train/'
    image_type = 'sar'
    model_type = "pretrained"
    workers = 10
    
    # for checkpoint saving
    checkpoint_dir = '/home/users/mcl66/SeaIce/checkpoints/'
    save_checkpoint = True
    
    # unlikely to need modification
    return_type = 'dict'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if image_type == "sar":
        is_single_band = True
        n_channels = 1
    elif image_type == "modis":
        is_single_band = False
        n_channels = 3

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        
        net = smp.Unet(encoder_name=config.encoder_name, encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type=None, in_channels=n_channels, classes=1, encoder_depth=5)
        net = net.double()

        # Loader
        img_list = create_npy_list(img_dir, image_type)

        train_img_list, val_img_list, n_train, n_val = split_img_list(img_list, config.validation_percent)

        train_dataset = CustomImageAugmentDataset(train_img_list, is_single_band, return_type, True)
        validation_dataset = CustomImageAugmentDataset(val_img_list, is_single_band, return_type, False)

        train_loader, val_loader = create_dataloaders(train_dataset, validation_dataset, config.batch_size, workers)

        # Optimiser
        optimiser = build_optimiser(net, config)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max',
                                                         patience=2)  # CHECK goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        criterion = nn.BCELoss()
        global_step = 0
        
        loss = smp.utils.losses.DiceLoss()
        metrics = [smp.utils.metrics.IoU(threshold=0.5),]
        
        train_epoch = smp.utils.train.TrainEpoch(
            net,
            loss=loss,
            metrics=metrics,
            optimizer=optimiser,
            #verbose=True,
            device=processor
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            net,
            loss=loss,
            metrics=metrics,
            #verbose=True,
            device=processor
        )


        # Begin training
        run_loss_train = 0
        run_loss_val = 0
        dir_checkpoint = ''
        for epoch in range(config.epochs):
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                dir_checkpoint = create_checkpoint_dir(folder_checkpoint, image_type, model_type, config.enconder_name, config.optimiser, 
                                                           config.learning_rate, config.batch_size, config.weight_decay)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            
            if i == n_epochs/2:
                optimizer.param_groups[0]['lr'] = 1e-5
            
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

            val_score = evaluate(net, val_loader, device, epsilon=config.weight_decay)
            #print('6')
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
            if save_checkpoint:
                if epoch == 0:
                    dir_checkpoint = create_checkpoint_dir(folder_checkpoint, image_type, model_type, config.optimiser, 
                                                           config.learning_rate, config.batch_size, config.weight_decay)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        wandb.log({"Run Loss, Training": run_loss_train / config.epochs, "Run Loss, Validation": val_score}, step=global_step)




if __name__ == '__main__':

    #wandb.init(project="test-hyptuning")

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
            'min': 0.00001,
            'max': 0.1
        },
        'batch_size': {
            # Uniformly-distributed between 5-15
            'distribution': 'int_uniform',
            'min': 5,
            'max': 25
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 1e-8,
            'max': 1e-2
        },
        'encoder_name': {
            'values': [ 'resnet34', 'vgg19', 'dpn68']
        },
        
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
            'value': 10}
    })

    sweep_id = wandb.sweep(sweep_config, project="hyp-sweep-jasmin-mcl")

    n_tuning = 30
    wandb.agent(sweep_id, function=train_and_validate, count=n_tuning)
