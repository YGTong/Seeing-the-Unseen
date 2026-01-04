import os
import random

import numpy as np

class DeepGlobe(object):
    """
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    random.seed(0)
    def __init__(self, root, split_ration=0.8):
        # self.root = os.path.expanduser(root)
        self.root = os.path.join(root,'DeepGlobe')
        self.imgs_dir = os.path.join(self.root, 'train')
        self.masks_dir = os.path.join(self.root, 'train_mask')
        self.split_ration = split_ration

        if not os.path.isdir(self.imgs_dir) or not os.path.isdir(self.masks_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        self.train_files, self.val_files = self.get_files()
        print(
            f"training image num: {len(self.train_files)}, validation image num: {len(self.val_files)}"
        )

    def get_files(self):
        datasets = []
        for imgs in os.listdir(self.imgs_dir):
            if imgs.endswith('.jpg'):
                img_path = os.path.join(self.imgs_dir, imgs)
                target_path = os.path.join(self.masks_dir, '{}_{}'.format(imgs.split('_')[0], 'mask.png'))
                datasets.append((img_path, target_path))
        #打乱list
        random.shuffle(datasets)

        train = datasets[:int(len(datasets)*self.split_ration)]
        train_indices = np.arange(len(train))[:]

        test = datasets[int(len(datasets)*self.split_ration):]
        test_indices = np.arange(len(test))[:]

        train_files = [
            {"img": train[i][0], "label": train[i][1]}
            for i in train_indices
        ]
        test_files = [
            {"img": test[i][0], "label": test[i][1]}
            for i in test_indices
        ]

        return train_files, test_files


# DeepGlobe = DeepGlobe(root='../../../dataset', split_ration=0.8)


