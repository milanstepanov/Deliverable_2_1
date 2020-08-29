import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2


def lfread(path, ang=7):
    """
    Load lenslet image and reshape in light field format VUCHW.
    """
    lenslet = cv2.imread(path)
    h, w, c = lenslet.shape
    lf = np.zeros((ang, ang, c, h // ang, w // ang), dtype=np.uint8)

    for idx in range(ang ** 2):
        q = idx // ang
        p = idx % ang
        lf[q, p, :, :, :] = lenslet[q::ang, p::ang, :].transpose(2, 0, 1)

    del lenslet
    return lf


class LightFieldDataset(Dataset):
    """Light Field dataset."""

    def __init__(self, root_dirs, angular, spatial_size=(372, 540), transform=None, ext=".png"):
        """
        Args:
            root_dir (string): A list with path to directory with images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dirs = root_dirs
        self.ext = ext
        
        self.img_list = []
        for root_dir in self.root_dirs: 
            
            for img_name in os.listdir(root_dir):
                if img_name.endswith(self.ext):
                    self.img_list.append(os.path.join(root_dir, img_name))

        self.transform = transform

        self.angular = angular
        self.min_view = 0
        self.max_view = self.angular - 1

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'filename': self.img_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class LoadLF(object):
    """The transform class which loads input image."""

    def __call__(self, sample):
        return {'lf': lfread(sample['filename']), 'filename': sample['filename']}


class RandomView(object):
    """The transform class selects random view from LF image and adds corner views and the select view to the
    dataset."""

    def __init__(self, angular):
        self.angular = angular

    def __call__(self, sample):
        lf = sample['lf']
        filename = sample['filename']

        # corner views
        tl = lf[0, 0, :, :, :]  # top left
        bl = lf[self.angular - 1, 0, :, :, :]  # bottom left
        tr = lf[0, self.angular - 1, :, :, :]  # top right
        br = lf[self.angular - 1, self.angular - 1, :, :, :]  # top left

        # position of view to reconstruct
        size = (1,)
        p = torch.randint(0, self.angular, size=size)  # index of the horizontal position of the selected view
        if p == 0 or p == self.angular - 1:
            q = torch.randint(1, self.angular - 1, size=size)  # index of the vertical position of the selected view
        else:
            q = torch.randint(0, self.angular, size=size)

        # view to reconstruct
        ground_truth = lf[q, p, :, :, :]

        return {'tl': tl, 'bl': bl, 'tr': tr, 'br': br,
                'p': p, 'q': q, 'ground_truth': ground_truth, 'filename': filename}


class SelectView(object):
    """The transform class selects random view from LF image and adds corner views and the select view to the
    dataset."""

    def __init__(self, angular, q, p):
        self.angular = angular
        self.q = q
        self.p = p

    def __call__(self, sample):
        lf = sample['lf']
        filename = sample['filename']

        # corner views
        tl = lf[0, 0, :, :, :]  # top left
        bl = lf[self.angular - 1, 0, :, :, :]  # bottom left
        tr = lf[0, self.angular - 1, :, :, :]  # top right
        br = lf[self.angular - 1, self.angular - 1, :, :, :]  # top left

        # view to reconstruct
        ground_truth = lf[self.q, self.p, :, :, :]

        return {'tl': tl, 'bl': bl, 'tr': tr, 'br': br,
                'p': self.p, 'q': self.q, 'ground_truth': ground_truth, 'filename': filename}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        tl, bl, tr, br = sample['tl'], sample['bl'], sample['tr'], sample['br']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']

        h, w = ground_truth.shape[1:]

        size = (1,)
        top = torch.randint(0, h - self.output_size, size=size)
        left = torch.randint(0, w - self.output_size, size=size)

        tl = tl[:, top: top + self.output_size, left: left + self.output_size]
        bl = bl[:, top: top + self.output_size, left: left + self.output_size]
        tr = tr[:, top: top + self.output_size, left: left + self.output_size]
        br = br[:, top: top + self.output_size, left: left + self.output_size]
        ground_truth = ground_truth[:, top: top + self.output_size, left: left + self.output_size]

        return {'tl': tl, 'bl': bl, 'tr': tr, 'br': br,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}


class RandomGamma(object):
    """Apply random gamma to all views."""

    def __init__(self, gamma_min, gamma_max):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, sample):
        tl, bl, tr, br = sample['tl'], sample['bl'], sample['tr'], sample['br']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']

        size = (1,)
        gamma = torch.randint(self.gamma_min, self.gamma_max, size=size)

        tl = (np.power(tl.astype('float32') / 255., gamma) * 255).astype('uint8')
        bl = (np.power(bl.astype('float32') / 255., gamma) * 255).astype('uint8')
        tr = (np.power(tr.astype('float32') / 255., gamma) * 255).astype('uint8')
        br = (np.power(br.astype('float32') / 255., gamma) * 255).astype('uint8')

        ground_truth = (np.power(ground_truth.astype('float32') / 255., gamma) * 255).astype('uint8')

        return {'tl': tl, 'bl': bl, 'tr': tr, 'br': br,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}


class Gamma(object):
    "Apply gamma to all views."

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, sample):
        tl, bl, tr, br = sample['tl'], sample['bl'], sample['tr'], sample['br']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']
        
        tl = (np.power(tl.astype('float32')/255., self.gamma) * 255).astype('uint8')
        bl = (np.power(bl.astype('float32')/255., self.gamma) * 255).astype('uint8')
        tr = (np.power(tr.astype('float32')/255., self.gamma) * 255).astype('uint8')
        br = (np.power(br.astype('float32')/255., self.gamma) * 255).astype('uint8')
        
        ground_truth = (np.power(ground_truth.astype('float32')/255., self.gamma) * 255).astype('uint8')

        return {'tl': tl, 'bl': bl, 'tr': tr, 'br': br,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}


class ToTensor(object):
    """Convert nd arrays in sample to Tensors."""

    def __init__(self, gamma=False, gamma_min=0.4, gamma_max=1., normalize=False):
        self.gamma = gamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.normalize = normalize

    def __call__(self, sample):
        tl, bl, tr, br = sample['tl'], sample['bl'], sample['tr'], sample['br']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']

        tl = torch.as_tensor(tl).float() / 255.
        bl = torch.as_tensor(bl).float() / 255.
        tr = torch.as_tensor(tr).float() / 255.
        br = torch.as_tensor(br).float() / 255.
        ground_truth = torch.as_tensor(ground_truth).float() / 255.
        p = torch.as_tensor(p).float()
        q = torch.as_tensor(q).float()

        if self.gamma:
            gamma = (self.gamma_max-self.gamma_min)*torch.rand(size=(1,))+self.gamma_min

            tl = torch.pow(tl, gamma)
            bl = torch.pow(bl, gamma)
            tr = torch.pow(tr, gamma)
            br = torch.pow(br, gamma)

            ground_truth = torch.pow(ground_truth, gamma)

        if self.normalize:
            scale = 2.
            offset = .5

            tl = tl * scale - offset
            bl = bl * scale - offset
            tr = tr * scale - offset
            br = br * scale - offset

            ground_truth = ground_truth * scale - offset

        return {'c1': tl, 'c2': bl, 'c3': tr, 'c4': br,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}
