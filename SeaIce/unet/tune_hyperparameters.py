from unet.dataset_preparation import *
from unet.evaluation import *
from unet.mini_network import *
import wandb
import os
import torch.optim as optim


def build_optimiser(network, config):
    if config.optimiser == "sgd":
        optimiser = optim.SGD(network.parameters(),
                              config.learning_rate, config.momentum)
    elif config.optimiser == "adam":
        optimiser = optim.Adam(network.parameters(),
                               lr=config.learning_rate)
    return optimiser


def train_and_validate(config=None, amp=False, device='cpu'):

    img_dir = '../tiled/'
    image_type = 'sar'
    net = MiniUNet(1, 2, True)
    return_type = 'dict'

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
        # Use this if you want a smaller dataset just to test things with
        img_list = small_sample(img_list)
        dataset = CustomImageDataset(img_list, is_single_band, return_type)
        n_val, n_train, train_loader, val_loader = split_data(dataset, config.validation_percent, config.batch_size, 2)

        # Optimiser
        optimiser = build_optimiser(net, config)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', patience=2)  # CHECK goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        criterion = nn.CrossEntropyLoss()
        global_step = 0

        # Begin training

        for epoch in range(config.epochs):
            net.train()
            epoch_loss = 0
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

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                    # change number for permute?

                    optimiser.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimiser)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    wandb.log({"batch loss": loss.item()})

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    val_score = evaluate(net, val_loader, device)
                    scheduler.step(val_score)


            wandb.log({"loss": epoch_loss, "epoch": epoch})



if __name__ == '__main__':

    os.environ["WANDB_API_KEY"] = 'ENTER API KEY'

    wandb.init(project="test-hyptuning")

    # Model Name
    model_name = 'mini-unet'

    # Configuring wandb settings
    sweep_config = {
        'method': 'random',  # Random search method -- less computationally expensive yet effective.
    }

    metric = {
        'name': 'loss',
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
            'max': 15,
        },
        'epochs': {
            # a flat distribution between 0 and 0.1
            'distribution': 'int_uniform',
            'min': 1,
            'max': 10
        }
    }
    sweep_config['parameters'] = parameters_dict

    # Fixed hyperparamters
    parameters_dict.update({
        'weight_decay': {
            'value': 1e-8},
        'momentum': {
            'value': 0.9},
        'validation_percent': {
            'value': 0.1},
        'img_scale': {
            'value': 0.5}
    })

    sweep_id = wandb.sweep(sweep_config, project=model_name + "hyp-sweep")

    n_tuning = 5
    wandb.agent(sweep_id, function=train_and_validate, count=n_tuning)
