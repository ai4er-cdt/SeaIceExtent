from unet.shared import *
from unet.network_structure import UNet
from unet.dataset_preparation import create_npy_list
from torchvision import transforms
import segmentation_models_pytorch as smp


def load_model(model_path, unet_type, image_type):
    """Load a previously trained model from .pth file.
       Parameters: 
            model_path: (string) file path of .pth model.
            unet_type: (string) "raw" or "pretrained".
            image_type: (string) "sar" or "modis".
       Returns: the model instance loaded into PyTorch, ready for eval().
    """
    if image_type == "modis":
        classes = 1
        channels = 3
    elif image_type == "sar":
        classes = 2
        channels = 1
    if unet_type == "raw":
        model = UNet(n_channels=channels, n_classes=classes, bilinear=True)
    elif unet_type == "pretrained":
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type=None, in_channels=channels, classes=1, encoder_depth=5)
        model = model.double()
    state = torch.load(model_path)
    model.load_state_dict(state)
    return(model)


def plot_img_and_mask(img, mask, image_type):
    """Show visualisation of prediction.
       Parameters: 
            img: the image passed into the network.
            mask: (numpy array) the predicted classes.
            image_type: (string) "sar" or "modis".
       Output: pyplot plot of predicted classifications.
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    _, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    if image_type == "sar":
            ax[0].imshow(img, cmap='gray')
    elif image_type == "modis":
            ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
    

def unlog_img(img):
    img_e = np.exp(np.array(img))
    return(img_e)
    

def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    """Predicts classes from an image.
       Parameters:
            net: the model instance.
            full_img: the image to predict.
            device: CPU or CUDA.
            out_threshold: (float) the predicted probability of a pixel's class required to assert and display that class.
       Outputs:
            (numpy array) a binary classification of ice or water per pixel.
            full_mask (numpy array) the predicted probabilities for classifications.
    """
    net.eval()
    img = full_img 
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy(), full_mask
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(), full_mask
    

def make_predictions(model_path, unet_type, image_type, dir_in, dir_out_bin=None, dir_out_prob=None, log = False, metrics = False, viz = False, save = False):
    if viz:
        import matplotlib.pyplot as plt
    if metrics:
        img_list = create_npy_list(dir_in, image_type)
        # Build up all the metrics to get an average for the whole image.
        all_precision, all_recall, all_accuracy = [], [], []
    else:
        _, img_list = get_contents(dir_in, ".npy", "suffix")
    
    net = load_model(model_path, unet_type, image_type)

    for filename in img_list:
        if metrics: 
            filename = filename[0]
            groundtruth_filename = filename.replace(image_type,'labels')
            gt_npy = np.vstack(np.load(groundtruth_filename))
            gt100 = np.where(gt_npy == 100, 0, gt_npy)
            gt200 = np.where(gt100 == 200, 1, gt100)
            gt_remap = np.vstack(gt200).astype(int)
            #print(gt_remap)
        logging.info(f'\nPredicting image {filename} ...')
        if log: npimg = unlog_img(np.load(filename))
        else: npimg = np.load(filename)
        img = torch.from_numpy(np.vstack(npimg).astype(int))
        if image_type == "sar":
            img = img[None,:]
            out_threshold = 0.5
        elif image_type == "modis":
            img = torch.permute(img, (2, 0, 1))
            out_threshold = 0.8
        
        mask_bin, mask_prob = predict_img(net=net, full_img=img, out_threshold=out_threshold, device=processor)

        if metrics:
            #print(gt_remap.shape, mask[0].shape)
            #print(np.unique(gt_remap), np.unique(mask[0]))
            #print(accuracy_score(gt_remap, mask[0], average = 'micro'), precision_score(gt_remap, mask[0]))
            N = gt_remap.shape[0] * gt_remap.shape[1]
            #accuracy = (gt_remap == mask).sum() / N
            TP = ((mask_bin[0] == 1) & (gt_remap == 1)).sum()
            TN = ((mask_bin[0] == 0) & (gt_remap == 0)).sum()
            FP = ((mask_bin[0] == 1) & (gt_remap == 0)).sum()
            FN = ((mask_bin[0] == 0) & (gt_remap == 1)).sum()
            precision = TP / (TP+FP)
            recall = TP / (TP+FN)
            accuracy = (TP+TN) / N
            print("Precision: {}, Recall: {}, Accuracy: {}".format(precision, recall, accuracy))
            # Build up all the metrics to get an average for the whole image.
            all_precision.append(precision) if not np.isnan(precision) else all_precision.append(0)
            all_recall.append(recall) if not np.isnan(recall) else all_recall.append(0)
            all_accuracy.append(accuracy) if not np.isnan(accuracy) else all_accuracy.append(0)
        if viz:
             logging.info(f'Visualizing results for image {filename}, close to continue...')
             plot_img_and_mask(img.squeeze(), mask_bin, image_type)

        if save:
             filename = filename[::-1]
             filename = filename.split("\\", 1)[0]
             filename = filename[::-1]
             filename = filename.split(".")[0]
             out_filename = name_file(filename, ".npy", dir_out_bin)
             np.save(out_filename, mask_bin)
             out_filename = name_file(filename, ".npy", dir_out_prob)
             print("predicted", filename)
             np.save(out_filename, mask_prob)
             logging.info(f'Mask saved to {out_filename}')
    if metrics:
        num_tiles = len(img_list)
        total_precision = sum(all_precision) / num_tiles
        total_recall = sum(all_recall) / num_tiles
        total_accuracy = sum(all_accuracy) / num_tiles
        return total_precision, total_recall, total_accuracy


