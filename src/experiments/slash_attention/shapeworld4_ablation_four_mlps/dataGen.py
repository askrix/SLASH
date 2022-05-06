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
import datasets

from tqdm import tqdm

    

def get_data_and_object_list(dataset, batch_size, shuffle=True):
    """
    Returns a list of hashmaps containing data input for the NeurAsp program, the queries and the oberservation attributes as an array.
    NeurASP expects the data in form of lists of a hashmap. Thats because we need to map the input for example two images to its corresponding atoms in the logic program.
    Example MNIST digit addition: [{'im1':[...], 'im2':[...] }, {'im1':[...], 'im2':[...] }, ...]
    
    """

    
    loader = get_loader(dataset, 1000, 8, shuffle)
    
    data_list = []
    query_list = []
    obj_list = []

    for im, query, obj in tqdm(loader):
        for i in range(0, len(query)):
            data_list.append({'im': im[i][None,:,:,:]})
            query_list.append(query[i])
            obj_list.append(obj[i])

    data_list = np.array(data_list)
    query_list= np.array(query_list)
    obj_list = torch.stack(obj_list)
    
    return data_list, query_list, obj_list
        


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    '''
    Returns and iterable dataset with specified batchsize and shuffling.
    '''
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
)


    
       
def get_encoding(color, shape, shade, size):

    if color == 'red':
        col_enc = [1,0,0,0,0,0,0,0,0]
    elif color == 'blue':
        col_enc = [0,1,0,0,0,0,0,0,0]
    elif color == 'green':
        col_enc = [0,0,1,0,0,0,0,0,0]
    elif color == 'gray':
        col_enc = [0,0,0,1,0,0,0,0,0]
    elif color == 'brown':
        col_enc = [0,0,0,0,1,0,0,0,0]
    elif color == 'magenta':
        col_enc = [0,0,0,0,0,1,0,0,0]
    elif color == 'cyan':
        col_enc = [0,0,0,0,0,0,1,0,0]
    elif color == 'yellow':
        col_enc = [0,0,0,0,0,0,0,1,0]
    elif color == 'black':
        col_enc = [0,0,0,0,0,0,0,0,1]
        

    if shape == 'circle':
        shape_enc = [1,0,0,0]
    elif shape == 'triangle':
        shape_enc = [0,1,0,0]
    elif shape == 'square':
        shape_enc = [0,0,1,0]    
    elif shape == 'bg':
        shape_enc = [0,0,0,1]
        
    if shade == 'bright':
        shade_enc = [1,0,0]
    elif shade =='dark':
        shade_enc = [0,1,0]
    elif shade == 'bg':
        shade_enc = [0,0,1]
        
        
    if size == 'small':
        size_enc = [1,0,0]
    elif size == 'big':
        size_enc = [0,1,0]
    elif size == 'bg':
        size_enc = [0,0,1]
    
    return col_enc + shape_enc + shade_enc + size_enc + [1]
    
    
class SHAPEWORLD4(Dataset):
    def __init__(self, root, mode, learn_concept='default', bg_encoded=True):
        
        datasets.maybe_download_shapeworld4()

        
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
        
        #target maps of the form {'target:idx': query string} or {'target:idx': obj encoding}
        self.query_map = {}
        self.obj_map = {}
                
        with open(os.path.join(root, 'labels', mode,"world_model.json")) as f:
            worlds = json.load(f)
            
            
            
            #iterate over all objects
            for world in worlds:
                num_objects = 0
                target_query = ""
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
                    target_query = target_query+ ":- not object({},{},{},{},{}). ".format(name, color, shape, shade, size)
                    obj_enc.append(get_encoding(color, shape, shade, size))
                    num_objects += 1
                    
                #bg encodings
                for i in range(num_objects, 4):
                    name = 'o' + str(num_objects+1)
                    target_query = target_query+ ":- not object({},black,bg, bg, bg). ".format(name)
                    obj_enc.append(get_encoding("black","bg","bg","bg"))
                    num_objects += 1


                self.query_map[count] = target_query
                self.obj_map[count] = np.array(obj_enc)
                count+=1
            
            
                    
    def __getitem__(self, index):
        
        #get the image
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.CenterCrop(250),
            #transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        return img, self.query_map[index] ,self.obj_map[index]#, mask
        
    def __len__(self):
        return len(self.img_paths)

