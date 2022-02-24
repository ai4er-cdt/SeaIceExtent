from SeaIce.unet.shared import *
import wandb

# Model Name
model_name = 'unet-original'

# Configuring wandb settings
sweep_config = {
    'method': 'random' # Random search method -- less computationally expensive yet effective.
    }

# Tuned hyperparameters
parameters_dict = {
    'optimizer': {
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

sweep_id = wandb.sweep(sweep_config, project= model_name + "pytorch-sweeps-demo")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Loader
        # Network
        # Optimiser
        # Calculate loss for each epoch
        
wandb.agent(sweep_id, train, count=5)

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