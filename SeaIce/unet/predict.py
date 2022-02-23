# Use an existing, trained and tested model to estimate the sea and ice from one image.
from SeaIce.unet.shared import *
from SeaIce.unet.network_structure import UNet


def load_model(model_path, unet_type, image_type): 
    if image_type == "modis":
        channels = 3
    elif image_type == "sar":
        channels = 1
    if unet_type == "raw":
        model = UNet(n_channels=channels, n_classes=2, bilinear=True)
    elif unet_type == "pretrained":
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type=None, in_channels=channels, classes=1, encoder_depth=5)
        model = model.double()
    state = torch.load(model_path)
    
    #model.load_state_dict(state)
    #model.eval()

#load_model(r"./best_model.pth", "pretrained", "sar")
    
