from torchvision import transforms


class AugPreprocessor:
    def __init__(self, resize: tp.Tuple[int, int]):
        self.resize = resize

        self.to_gray = transforms.Grayscale(3)
        self.jitter = transforms.ColorJitter(brightness=.5, hue=.3)
        
        self.x_preprocessor = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

        self.y_preprocessor = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
        ])

    def preprocess(self, collection, aug_params, bitwise: bool=False):
        
        color = aug_params[0]
        angle = aug_params[1]
        hflip = aug_params[2]
        vflip = aug_params[3]

        image = self.x_preprocessor(collection[0])
        image = image.unsqueeze(0)
        mask = self.y_preprocessor(collection[1])
        mask = mask.unsqueeze(0).type(torch.BoolTensor)

        if color==1:
            image = self.to_gray(image)
        elif color==2:
            image = self.jitter(image)
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
        if hflip:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if vflip:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        if bitwise is True:
            reverse_mask = torch.bitwise_not(mask)
            mask = torch.cat([mask, reverse_mask], dim=1)
        return image, mask

    def __call__(self, data, bitwise: bool=False):
        images = []
        masks = []
        for item in data:
            aug_param, collection = item
            image, mask = self.preprocess(collection, aug_param, bitwise)
            images.append(image)
            masks.append(mask)
        x = torch.cat(images, dim=0)
        y = torch.cat(masks, dim=0)
        return x, y

class Preprocessor:
    def __init__(self, resize: tp.Tuple[int, int]):
        self.resize = resize
        
        self.x_preprocessor = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

        self.y_preprocessor = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
        ])

    def preprocess(self, collection, bitwise: bool=False):
        image = self.x_preprocessor(collection[0])
        image = image.unsqueeze(0)
        mask = self.y_preprocessor(collection[1])
        mask = mask.unsqueeze(0).type(torch.BoolTensor)
        if bitwise is True:
            reverse_mask = torch.bitwise_not(mask)
            mask = torch.cat([mask, reverse_mask], dim=1)
        return image, mask

    def __call__(self, data, bitwise: bool=False):
        images = []
        masks = []
        for collection in data:
            image, mask = self.preprocess(collection, bitwise)
            images.append(image)
            masks.append(mask)
        x = torch.cat(images, dim=0)
        y = torch.cat(masks, dim=0)
        return x, y
