from PIL import Image


class Loader:
    def __init__(self, data: tp.List[ tp.Union[tp.Tuple[str, str, str], tp.Tuple[str, str]]]):

        """
        Convert pathes to PIL Images
        """

        self.aug_params = []
        self.images = []
        self.masks = []
        for item in data:
            collection = item[0]
            if len(collection) == 2:
                image = Image.open(collection[0])
                mask = Image.open(collection[1])
                self.images.append(image)
                self.masks.append(mask)
            elif len(collection) == 3:
                image = Image.open(collection[0])
                mask = Image.open(collection[1])
                background = Image.open(collection[2])
                image = self.paste(image, background)
                self.images.append(image)
                self.masks.append(mask)
            self.aug_params.append(item[1])

    def paste(self, image, background):
        background = background.resize(image.size)        
        background.paste(image, (0,0), image)
        return background

    def __getitem__(self, index: tp.Union[int, slice]):
        items = []
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            for i in range(start, stop, step):
                if i < len(self):
                    items.append(self(i))
        else:
            items.append(self(index))
        return items

    def __call__(self, index: int):
        return (self.aug_params[index], (self.images[index], self.masks[index]))

    def __len__(self):
        return len(self.images)
