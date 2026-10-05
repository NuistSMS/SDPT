from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import os
import torch
import random
import numpy as np


def pkload(fname):

    with open(fname, 'rb') as f:
        return pickle.load(f)


class Random_Flip(object):


    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=2)

        return {'image': image, 'label': label}


class Random_Rot90(object):

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        k = np.random.randint(0, 4)

        image = np.rot90(image, k, axes=(0, 1))
        label = np.rot90(label, k, axes=(0, 1))

        return {'image': image.copy(), 'label': label.copy()}


class Random_Gamma(object):


    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.7, 1.5)  


            min_val = image.min()
            rng = image.max() - min_val
            if rng > 0:
                image = np.power(((image - min_val) / rng), gamma) * rng + min_val

        return {'image': image, 'label': label}





class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        axial_image = sample['axial_image']
        axial_image = np.ascontiguousarray(axial_image.transpose(2, 3, 0, 1))

        coronal_image = sample['coronal_image']
        coronal_image = np.ascontiguousarray(coronal_image.transpose(1, 3, 0, 2))

        sagittal_image = sample['sagittal_image']
        sagittal_image = np.ascontiguousarray(sagittal_image.transpose(0, 3, 1, 2))

        axial_label = sample['axial_label']
        axial_label = np.ascontiguousarray(axial_label.transpose(2, 0, 1))
        coronal_label = sample['coronal_label']
        coronal_label = np.ascontiguousarray(coronal_label.transpose(1, 0, 2))
        sagittal_label = sample['sagittal_label']
        sagittal_label = np.ascontiguousarray(sagittal_label.transpose(0, 1, 2))

        image = torch.from_numpy(image).float()
        axial_image = torch.from_numpy(axial_image).float()
        coronal_image = torch.from_numpy(coronal_image).float()
        sagittal_image = torch.from_numpy(sagittal_image).float()
        axial_label = torch.from_numpy(axial_label).float()
        coronal_label = torch.from_numpy(coronal_label).float()
        sagittal_label = torch.from_numpy(sagittal_label).float()

        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label,
                'axial_image': axial_image, 'coronal_image': coronal_image, 'sagittal_image': sagittal_image,
                'axial_label': axial_label, 'coronal_label': coronal_label, 'sagittal_label': sagittal_label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (20, 20), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (20, 20)), mode='constant')
        return {'image': image, 'label': label}


class Center_Crop(object):
    output_size = (192, 192, 192)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (w, h, d, c) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2], :]

        return {'image': image, 'label': label}


class Random_Slice(object):

    def __init__(self, num_slices=8):
        self.num_slices = num_slices

    def __call__(self, sample):
        image = sample['image']  
        label = sample['label']  
        w, h, d, c = image.shape  


        axial_indices = random.sample(range(d), self.num_slices)
        coronal_indices = random.sample(range(h), self.num_slices)
        sagittal_indices = random.sample(range(w), self.num_slices)


        sample['axial_image'] = image[:, :, axial_indices, :]  
        sample['coronal_image'] = image[:, coronal_indices, :, :]  
        sample['sagittal_image'] = image[sagittal_indices, :, :, :]  


        sample['axial_label'] = label[:, :, axial_indices]  
        sample['coronal_label'] = label[:, coronal_indices, :]  
        sample['sagittal_label'] = label[sagittal_indices, :, :]  

        return sample


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']
        scale_factor = np.random.uniform(1.0 - factor, 1.0 + factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image * scale_factor + shift_factor

        return {'image': image, 'label': label}


def BraTS_processing_train(sample, batchsize):

    trans = transforms.Compose([
        Pad(),
        Center_Crop(),
        Random_Flip(),  
        Random_Rot90(),  
        Random_Gamma(),
        Random_intencity_shift(),  
        Random_Slice(batchsize),
        ToTensor()
    ])
    sample = trans(sample)
    return sample




def BraTS_processing_valid(sample, batchsize):

    trans = transforms.Compose([
        Pad(),
        Center_Crop(),
        Random_Slice(batchsize),
        ToTensor()
    ])
    sample = trans(sample)
    return sample


class Brats19_dataset(Dataset):


    def __init__(self, list_dir, plant='all', batchsize=36, root='', mode='', list_file_name=''):
        self.plant = plant
        self.batchsize = batchsize
        self.lines = []
        paths, names = [], []


        list_file = os.path.join(list_dir, list_file_name)
        list_file = os.path.normpath(list_file)
        root = os.path.normpath(root) if root else ''

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                name = os.path.basename(line)
                names.append(name)

                if "HGG" in line or "LGG" in line:
                    path = os.path.join(root, line + '_')
                else:
                    path = os.path.join(root, name + '_')

                path = os.path.normpath(path)
                paths.append(path)
                self.lines.append(line)

        self.names = names
        self.paths = paths
        self.mode = mode


        if len(paths) > 0:
            path_prefix = paths[0]
            case_folder = path_prefix.rstrip('_')
            pkl_name = f"{os.path.basename(case_folder)}_pkl_ui8f32b0.pkl"
            full_pkl_path = os.path.normpath(os.path.join(case_folder, pkl_name))
            print(f"First case prefix: {path_prefix}")
            print(f"First serialized case: {full_pkl_path}")
            print(f"Serialized case exists: {os.path.exists(full_pkl_path)}")

    def __getitem__(self, item):
        path_prefix = self.paths[item]
        case_folder = path_prefix.rstrip('_')
        pkl_name = f"{os.path.basename(case_folder)}_pkl_ui8f32b0.pkl"
        full_pkl_path = os.path.normpath(os.path.join(case_folder, pkl_name))

        if self.mode == 'train':
            image, label = pkload(full_pkl_path)
            sample = {'image': image, 'label': label}

            sample = BraTS_processing_train(sample, self.batchsize)
        elif self.mode == 'valid':
            image, label = pkload(full_pkl_path)
            sample = {'image': image, 'label': label}

            sample = BraTS_processing_valid(sample, self.batchsize)
        else:
            raise ValueError("mode not set (should be 'train' or 'valid')")

        if self.plant == 'axial':
            return sample['axial_image'], sample['axial_label']
        elif self.plant == 'coronal':
            return sample['coronal_image'], sample['coronal_label']
        elif self.plant == 'sagittal':
            return sample['sagittal_image'], sample['sagittal_label']
        elif self.plant == 'all':
            all_image = torch.concatenate([sample['axial_image'][:self.batchsize // 3],
                                           sample['coronal_image'][:self.batchsize // 3],
                                           sample['sagittal_image'][:self.batchsize // 3]], dim=0)
            all_label = torch.concatenate([sample['axial_label'][:self.batchsize // 3],
                                           sample['coronal_label'][:self.batchsize // 3],
                                           sample['sagittal_label'][:self.batchsize // 3]], dim=0)
            return all_image, all_label

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
