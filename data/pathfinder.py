from pathlib import Path
import typing as tp
import random


class GTEAPaths:
    def __init__(self, path_to_folder: str, shuffle: bool=True,
        seed: tp.Optional[int]=None):
        
        self.path_to_image = "/".join([path_to_folder, "image"])
        self.path_to_mask = "/".join([path_to_folder, "mask"])
        self.folder = Path(self.path_to_image)
        self.paths = self._load_paths()

        if seed is not None:
            random.seed(seed)

        if shuffle is True:
            random.shuffle(self.paths)

    def _load_paths(self) -> tp.List[tp.Tuple[str, str]]:
        paths = []
        for file in self.folder.glob("*"):
            filename = file.name
            if ".mat" in filename:
                continue
            image_path = str(file)
            mask_name = filename[:filename.find(".")] + ".png"
            mask_path = "/".join([self.path_to_mask, mask_name])
            paths.append((image_path, mask_path))
        return paths

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

    def __len__(self):
        return len(self.paths)

    def __call__(self, index: int):
        return self.paths[index]

class EgoHandPaths:
    def __init__(self, path_to_folder: str, shuffle: bool=True,
        seed: tp.Optional[int]=None):
        
        self.path_to_image = "/".join([path_to_folder, "image"])
        self.path_to_mask = "/".join([path_to_folder, "mask"])
        self.folder = Path(self.path_to_image)
        self.paths = self._load_paths()

        if seed is not None:
            random.seed(seed)

        if shuffle is True:
            random.shuffle(self.paths)

    def _load_paths(self) -> tp.List[tp.Tuple[str, str]]:
        paths = []
        for subfolder in self.folder.glob("*"):
            subname = subfolder.name
            for file in subfolder.glob("*"):
                filename = file.name
                if ".mat" in filename:
                    continue
                image_path = str(file)
                mask_name = filename[6:]
                mask_path = "/".join([self.path_to_mask, subname, mask_name])
                paths.append((image_path, mask_path))
        return paths

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

    def __len__(self):
        return len(self.paths)

    def __call__(self, index: int):
        return self.paths[index]

class Ego2HandPaths:
    def __init__(self, path_to_folder: str, path_to_background: str, 
        shuffle: bool=True, seed: tp.Optional[int]=None):
        self.bg_folder = Path(path_to_background)
        self.bg_paths = self._load_bg_paths()
        self.folder = Path(path_to_folder)
        self.paths = self._load_paths()

        if seed is not None:
            random.seed(seed)

        if shuffle is True:
            random.shuffle(self.paths)

    def _load_bg_paths(self) -> tp.List[str]:
        files = self.bg_folder.glob('*')
        bg_paths = []
        for file in files:
            bg_paths.append(str(file))
        return bg_paths

    def _get_random_background(self):
        return random.choice(self.bg_paths)

    def _load_paths(self) -> tp.List[tp.Tuple[str, str]]:
        paths = []
        for action in self.folder.glob('*'):
            for sequence in action.glob('*'):
                for files in sequence.glob('*'):
                    images = list(files.glob('*'))
                    if len(images[0].name) == 9:
                        image = str(images[0])
                        mask = str(images[1])
                    else:
                        image = str(images[1])
                        mask = str(images[0])
                    paths.append((image, mask, self._get_random_background()))
        return paths

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

    def __len__(self):
        return len(self.paths)

    def __call__(self, index: int):
        return self.paths[index]

class UnionPaths:
    def __init__(self, paths: tp.List[tp.Tuple], 
        shuffle: bool=True, seed: tp.Optional[int]=None):

        self.paths = paths

        if seed is not None:
            random.seed(seed)

        if shuffle is True:
            random.shuffle(self.paths)

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

    def __len__(self):
        return len(self.paths)

    def __call__(self, index: int):
        return self.paths[index]

class AugmentationPaths:
    def __init__(self, paths: tp.List[tp.Tuple], angles: tp.Tuple[int, int]=(-180, 180),
        seed: tp.Optional[int]=None, shuffle: bool=True, augmentate: bool=True):
        
        # [color, rotation, hflip, vflip]
        self.changes = []
        self.paths = []

        if seed is not None:
            random.seed(seed)

        for path in paths:
            self.paths.append(path)
            self.changes.append([0, 0, False, False])
            
            if augmentate is False:
                continue
                
            angle = random.randint(angles[0], angles[1])
            color = random.choice([0, 1, 2])
            hflip = random.choice([True, False])
            vflip = random.choice([True, False])
            self.paths.append(path)
            self.changes.append([color, angle, hflip, vflip])

        if shuffle is True:
            random.shuffle(self.paths)

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

    def __len__(self):
        return len(self.paths)

    def __call__(self, index: int):
        return (self.paths[index], self.changes[index])

