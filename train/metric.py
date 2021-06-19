
def convert_prediction_to_mask(prediction, thr: float=0):
    mask = prediction > thr
    return mask

# INTERSECTION OVER UNION
def get_iou(prediction, target):

    if target.shape != prediction.shape:
        raise Exception('A target shape doesn`t match with a prediction shape')

    if target.dim() != 3:
        raise Exception(f'A target dim is {target.dim()}. Must be 3.')

    pred_copy = prediction.clone()
    pred_copy = convert_prediction_to_mask(pred_copy)
    
    target_copy = target.clone()
    target_copy = convert_prediction_to_mask(target_copy)

    intersection = torch.bitwise_and(target_copy, pred_copy).sum().item()
    union = torch.bitwise_or(target_copy, pred_copy).sum().item()
    
    if (target_copy.sum().item() == 0) and (pred_copy.sum().item() == 0):
        return 1
    elif union == 0:
        return 0

    return intersection / union

def get_mean_iou(predictions, targets):

    with torch.no_grad():
        if targets.shape != predictions.shape:
            raise Exception('A targets shape doesn`t match with a predictions shape')

        if targets.dim() != 4:
            raise Exception(f'A target dim is {targets.dim()}. Must be 4.')

        iou_sum = 0
        for i in range(targets.shape[0]):
            iou = get_iou(targets[i], predictions[i])
            iou_sum += iou
        mean_iou = iou_sum / targets.shape[0]
        return mean_iou

# PIXEL ACCURACY
def get_pixel_acc(prediction, target):

    if target.shape != prediction.shape:
        raise Exception('A target shape doesn`t match with a prediction shape')

    if target.dim() != 3:
        raise Exception(f'A target dim is {target.dim()}. Must be 3.')

    pred_copy = prediction.clone()
    pred_copy = convert_prediction_to_mask(pred_copy)

    target_copy = target.clone()
    target_copy = convert_prediction_to_mask(target_copy)

    same = (target_copy == pred_copy).sum().item()
    channels, height, width = target.shape
    area = height * width * channels
    acc = same / area
    return acc

def get_mean_pixel_acc(predictions, targets):

    with torch.no_grad():
        if targets.shape != predictions.shape:
            raise Exception('A targets shape doesn`t match with a predictions shape')

        if targets.dim() != 4:
            raise Exception(f'A target dim is {targets.dim()}. Must be 4.')

        acc_sum = 0
        for i in range(targets.shape[0]):
            acc = get_pixel_acc(targets[i], predictions[i])
            acc_sum += acc
        mean_acc = acc_sum / targets.shape[0]
        return mean_acc
