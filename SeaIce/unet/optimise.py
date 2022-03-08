from shared import *
from model_raw import *
from mini_network import MiniUNet
 

if __name__ == '__main__':
    args = get_mini_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {processor}')
    net = MiniUNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    model = net.to(device=processor)
    #print(receptive_field(model, input_size=(3, 256, 256)))
    for _ in range(3):
        # Train the model with an assortment of different tile sizes. 
        permuted_tile_sizes = permute_tile_sizes()
        # Replace these parameters later with Jonnycode.
        loss = train_net(net=net,
                  device=processor,
                  image_type="modis",
                  dir_img=permuted_tile_sizes,
                  epochs=args.epochs, 
                  batch_size=args.batch_size,
                  learning_rate=0.0001,
                  save_checkpoint=args.save_checkpoint,
                  val_percent=0.1,
                  amp=args.amp)
        loss = loss.item()
        print(loss)