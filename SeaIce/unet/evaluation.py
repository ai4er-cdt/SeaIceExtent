""" Model evaluation functions and Dice Loss scoring functions """
from shared import *


def evaluate(net, dataloader, device, epsilon):
    """Evaluate the model during training.
       Parameters:
            net: the unet model instance.
            dataloader: the validation set dataloader object.
            device: CPU or CUDA.
            epsilon: weight decay
       Returns: dice score over the validation set.
    """

    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False, epsilon=epsilon) + criterion(mask_pred, mask_true)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False, epsilon=epsilon) + criterion(mask_pred, mask_true)
    net.train()
    print(num_val_batches)
    print(dice_score.item() / num_val_batches)


    # Fixes a potential division by zero error
    if num_val_batches == 0:
        output = dice_score.item()
    else:
        output = dice_score.item() / num_val_batches

    return output


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    """Average of Dice coefficient for all batches, or for a single mask."""
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...], epsilon=epsilon)
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, epsilon=1e-6):
    """Dice loss (objective to minimize) between 0 and 1"""
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, True, epsilon)


def view_model(model_path):
    # Load and view a previously trained model from a .pth or .pt file.
    model = torch.load(model_path)
    model.eval()
    print(model)
    return model


