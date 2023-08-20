import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, acc_type = 0): # acc_type = 1: acc_4, acc_type = 2: acc_8
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            #print('image files are: ', image_files)
            for fname in sorted(image_files):
                #print('fname is: {}'.format(fname))
                num_slices = self._get_metadata(fname)
                
                is_right_acc = ((acc_type == 0) 
                                or (acc_type == 1 and 'acc4' in str(fname)) 
                                or (acc_type == 2 and 'acc8' in str(fname))
                               )

                if is_right_acc:
                    self.image_examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
                
            is_right_acc = ((acc_type == 0) 
                            or (acc_type == 1 and 'acc4' in str(fname)) 
                            or (acc_type == 2 and 'acc8' in str(fname))
                           )

            if is_right_acc:
                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            #print('keys: ', hf.keys())
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        #print('kspace_fname is: ', kspace_fname.name)
        is_acc4 = 'acc4' in kspace_fname.name
        #print('mask: ', mask[0])#.shape)
        return self.transform(mask, input, target, attrs, is_acc4, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False, acc_type = 0):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        acc_type = acc_type
    )
    #print("shape is:", data_path, data_storage[0][1].shape)
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
