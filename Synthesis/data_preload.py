import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def lfread(path, ang=7):
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

    def __init__(self, root_dirs, angular, spatial_size = (372,540), transform=None, ext=".png"):
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

        
        self.all_data = np.zeros((len(self.img_list), 7, 7, 3, spatial_size[0], spatial_size[1]), dtype=np.uint8)
        for img_id, img_name in enumerate(self.img_list):            
            lf = lfread(img_name)
            self.all_data[img_id, ...] = lf[:, :, :, 0:spatial_size[0], 0:spatial_size[1]]
            
    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lf = self.all_data[idx]
        sample = {'lf': lf, 'filename': self.img_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SelectView(object):
    def __init__(self, angular, p, q):
        self.angular = angular
        self.p = p
        self.q = q
        
    def __call__(self, sample):
        lf = sample['lf']
        filename = sample['filename']

        # corner views
        c1 = lf[0, 0, :, :, :]
        c2 = lf[self.angular - 1, 0, :, :, :]
        c3 = lf[0, self.angular - 1, :, :, :]
        c4 = lf[self.angular - 1, self.angular - 1, :, :, :]
        
        ground_truth = lf[self.q, self.p, :, :, :]

        return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                'p': self.p, 'q': self.q,
                'ground_truth': ground_truth,
                'filename': filename}
        

class RandomView(object):
    """Randomly select a view to reconstruct."""

    def __init__(self, angular):
        self.angular = angular

    def __call__(self, sample):
        lf = sample['lf']
        filename = sample['filename']

        # corner views
        c1 = lf[0, 0, :, :, :]
        c2 = lf[self.angular - 1, 0, :, :, :]
        c3 = lf[0, self.angular - 1, :, :, :]
        c4 = lf[self.angular - 1, self.angular - 1, :, :, :]

        # position of view to reconstruct            
#         p = np.random.randint(0, self.angular)
#         q = np.random.randint(0, self.angular)

        size = (1,)
        p = torch.randint(0, self.angular, size=size)
        if p==0 or p == self.angular-1:
            q = torch.randint(1, self.angular-1,size=size)
        else:
            q = torch.randint(0, self.angular, size=size)

        # view to reconstruct
        ground_truth = lf[q, p, :, :, :]
        
        return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                'p': p, 'q': q, 'ground_truth': ground_truth, 'filename': filename}


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
        c1, c2, c3, c4 = sample['c1'], sample['c2'], sample['c3'], sample['c4']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']

        h, w = ground_truth.shape[1:]

        size = (1,)
        top = torch.randint(0, h - self.output_size, size=size)
        left = torch.randint(0, w - self.output_size, size=size)

        c1 = c1[:, top: top + self.output_size, left: left + self.output_size]
        c2 = c2[:, top: top + self.output_size, left: left + self.output_size]
        c3 = c3[:, top: top + self.output_size, left: left + self.output_size]
        c4 = c4[:, top: top + self.output_size, left: left + self.output_size]
        ground_truth = ground_truth[:, top: top + self.output_size, left: left + self.output_size]

        return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}


class Gamma(object):
    "Apply gamma to all views."

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, sample):
        c1, c2, c3, c4 = sample['c1'], sample['c2'], sample['c3'], sample['c4']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']
        
        c1 = (np.power(c1.astype('float32')/255., self.gamma) * 255).astype('uint8')
        c2 = (np.power(c2.astype('float32')/255., self.gamma) * 255).astype('uint8')
        c3 = (np.power(c3.astype('float32')/255., self.gamma) * 255).astype('uint8')
        c4 = (np.power(c4.astype('float32')/255., self.gamma) * 255).astype('uint8')
        
        ground_truth = (np.power(ground_truth.astype('float32')/255., self.gamma) * 255).astype('uint8')

        return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        c1, c2, c3, c4 = sample['c1'], sample['c2'], sample['c3'], sample['c4']
        p, q, ground_truth = sample['p'], sample['q'], sample['ground_truth']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        c1 = torch.as_tensor(c1).float() / 255.
        c2 = torch.as_tensor(c2).float() / 255.
        c3 = torch.as_tensor(c3).float() / 255.
        c4 = torch.as_tensor(c4).float() / 255.
        ground_truth = torch.as_tensor(ground_truth).float() / 255.
        p = torch.as_tensor(p).float()
        q = torch.as_tensor(q).float()

        return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                'p': p, 'q': q, 'ground_truth': ground_truth,
                'filename': sample['filename']}


class ObjToTensor(object):
    
    def __call__(self, sample):
                
        for key, val in sample.items():
            
            sample[key] = torch.as_tensor(val).float() / 255.
            
            
        return sample
