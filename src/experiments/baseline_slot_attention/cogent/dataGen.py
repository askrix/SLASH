import torch
import torchvision

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import datasets as datasets


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    '''
    Returns and iterable dataset with specified batchsize and shuffling.
    '''
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
)


def get_encoding(color, shape, shade, size):
    
    if color == 'red':
        col_enc = [1,0,0,0,0,0,0,0]
    elif color == 'blue':
        col_enc = [0,1,0,0,0,0,0,0]
    elif color == 'green':
        col_enc = [0,0,1,0,0,0,0,0]
    elif color == 'gray':
        col_enc = [0,0,0,1,0,0,0,0]
    elif color == 'brown':
        col_enc = [0,0,0,0,1,0,0,0]
    elif color == 'magenta':
        col_enc = [0,0,0,0,0,1,0,0]
    elif color == 'cyan':
        col_enc = [0,0,0,0,0,0,1,0]
    elif color == 'yellow':
        col_enc = [0,0,0,0,0,0,0,1]

    if shape == 'circle':
        shape_enc = [1,0,0]
    elif shape == 'triangle':
        shape_enc = [0,1,0]
    elif shape == 'square':
        shape_enc = [0,0,1]    
   
    if shade == 'bright':
        shade_enc = [1,0]
    elif shade =='dark':
        shade_enc = [0,1]

             
    if size == 'small':
        size_enc = [1,0]
    elif size == 'big':
        size_enc = [0,1]
    
    return np.array([1]+ col_enc + shape_enc + shade_enc + size_enc)
    
    
class SHAPEWORLD_COGENT(Dataset):
    def __init__(self, root, mode, learn_concept='default', bg_encoded=True):
        
        datasets.maybe_download_shapeworld_cogent()

        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        #dictionary of the form {'image_idx':'img_path'}
        self.img_paths = {}
        
        
        for file in os.scandir(os.path.join(root, 'images', mode)):
            img_path = file.path
            
            img_path_idx =   img_path.split("/")
            img_path_idx = img_path_idx[-1]
            img_path_idx = img_path_idx[:-4][6:]
            try:
                img_path_idx =  int(img_path_idx)
                self.img_paths[img_path_idx] = img_path
            except:
                print("path:",img_path_idx)
                

        
        count = 0
        
        #target maps of the form {'target:idx': observation string} or {'target:idx': obj encoding}
        self.obj_map = {}
                
        with open(os.path.join(root, 'labels', mode,"world_model.json")) as f:
            worlds = json.load(f)
            
            
            
            #iterate over all objects
            for world in worlds:
                num_objects = 0
                target_obs = ""
                obj_enc = []
                for entity in world['entities']:
                    
                    color = entity['color']['name']
                    shape = entity['shape']['name']
                    
                    shade_val = entity['color']['shade']
                    if shade_val == 0.0:
                        shade = 'bright'
                    else:
                        shade = 'dark'
                    
                    size_val = entity['shape']['size']['x']
                    if size_val == 0.075:
                        size = 'small'
                    elif size_val == 0.15:
                        size = 'big'
                    
                    name = 'o' + str(num_objects+1)
                    obj_enc.append(get_encoding(color, shape, shade, size))
                    num_objects += 1
                    
                #bg encodings
                for i in range(num_objects, 4):
                    name = 'o' + str(num_objects+1)
                    obj_enc.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
                    num_objects += 1

                self.obj_map[count] = torch.Tensor(obj_enc)
                count+=1

    def __getitem__(self, index):
        
        #get the image
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        return img, self.obj_map[index]#, mask
        
    def __len__(self):
        return len(self.img_paths)

