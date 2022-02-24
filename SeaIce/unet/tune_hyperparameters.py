from SeaIce.unet.dataset_preparation import *
import wandb

# Model Name
model_name = 'unet-original'

# Configuring wandb settings
sweep_config = {
    'method': 'random'  # Random search method -- less computationally expensive yet effective.
    'metric': {
        'name': 'loss',
        'minimize'}
}

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
        'distribution': 'uniform',
        'min': 5,
        'max': 15,
    },
    'epochs': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
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

sweep_id = wandb.sweep(sweep_config, project=model_name + "pytorch-sweeps-demo")


import torch.optim as optim


def build_optimiser(network, config):
    if config.optimiser == "sgd":
        optimiser = optim.SGD(network.parameters(),
                              config.learning_rate, config.momentum)
    elif config.optimiser == "adam":
        optimiser = optim.Adam(network.parameters(),
                               lr=config.learning_rate)
    return optimiser


def train_and_validate(img_dir, image_type, return_type, config=None):

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

        # Network
        network = 10000

        # Optimiser
        optimiser = build_optimiser(network, config)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', patience=2)  # CHECK goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        criterion = nn.CrossEntropyLoss()
        global_step = 0

        # Begin training

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
                                    'pred': wandb.Image(
                                        torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })

        for epoch in range(config.epochs):
            avg_loss = train_and_validate_epoch(network, train_loader, val_loader, optimiser)
            wandb.log({"loss": avg_loss, "epoch": epoch})


n_tuning = 5
wandb.agent(sweep_id, function=train_and_validate, count=n_tuning)

"""
def optimise():
    learning_rate = round(random.uniform(0.000001, 0.01), 6)
    learning_rate_best = learning_rate
    batch_size = random.randint(1, 10)
    batch_size_best = batch_size
    epochs = random.randint(1, 5)
    epochs_best = epochs
    val_percent = random.randint(1, 40)
    val_percent_best = val_percent
    img_scale = round(random.uniform(0.1, 1.0), 2)
    img_scale_best = img_scale
    receptor_field = 0
    weight_decay = 0
    bias = 0
    tile_sizes = [128, 256, 512, 1024]
    tile_index = random.randint(0, 3)
    tile_index_best = tile_index
    tile_size = tile_sizes[tile_index]

    for _ in range(3):

        #### loop the NN ####
        learning_rate += random.uniform(-0.001, 0.001)
        learning_rate = round(learning_rate, 6)
        if learning_rate < 0.000001 or learning_rate > 0.01:
            learning_rate = learning_rate_best

        batch_size += random.randint(-1, 1)
        if batch_size < 1 or batch_size > 10:
            batch_size = batch_size_best

        epochs += random.randint(-1, 1)
        if epochs < 1 or epochs > 5:
            epochs = epochs_best

        val_percent += random.randint(-5, 5)
        if val_percent < 1 or val_percent > 40:
            val_percent = val_percent_best

        img_scale += random.uniform(-0.1, 0.1)
        img_scale = round(img_scale, 2)
        if img_scale < 0.1 or img_scale > 1.0:
            img_scale = img_scale_best

        tile_index += random.randint(-1, 1)
        if tile_index < 0 or tile_index > 3:
            tile_index = tile_index_best
        tile_size = tile_sizes[tile_index]

        print(learning_rate, batch_size, epochs, val_percent, img_scale, tile_size)

    ### Get fitness function ###

    # if improved:
    learning_rate_best = learning_rate
    batch_size_best = batch_size
    epochs_best = epochs
    val_percent_best = val_percent
    img_scale_best = img_scale
    tile_index_best = tile_index

    # if fitness score is worse than a certain threshold:
    learning_rate = learning_rate_best
    batch_size = batch_size_best
    epochs = epochs_best
    val_percent = val_percent_best
    img_scale = img_scale_best
    tile_index = tile_index_best


optimise()

"""