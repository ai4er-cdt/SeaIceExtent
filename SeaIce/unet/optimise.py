from SeaIce.unet.shared import *
from SeaIce.unet.model_raw import *
from SeaIce.unet.mini_network import MiniUNet
 
#epochs/early stopping criterion

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



if __name__ == '__main__':
    args = get_mini_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {processor}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = MiniUNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device=processor)
    val_score = train_net(net=net,
                  device=processor,
                  epochs=args.epochs, 
                  image_type="modis",
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  img_scale=args.scale,
                  save_checkpoint=args.save_checkpoint,
                  val_percent=args.val / 100,
                  amp=args.amp)
