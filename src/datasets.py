import os
import tempfile
import  urllib.request
import shutil

from zipfile import ZipFile
import gzip
import utils

def maybe_download(directory, url_base, filename, suffix='.zip'):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base  +filename
    
    _, zipped_filepath = tempfile.mkstemp(suffix=suffix)
        
    print('Downloading {} to {}'.format(url, zipped_filepath))
    
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True


def extract_dataset(directory, filepath, filepath_extracted):
    if not os.path.isdir(filepath_extracted):
        print('unzip ',filepath, " to", filepath_extracted)
        with ZipFile(filepath, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall(directory)



def maybe_download_shapeworld2():
        
    directory = "../../data/"
    file_name= "shapeworld2.zip"
    maybe_download(directory, "https://hessenbox.tu-darmstadt.de/dl/fiULMv9mDzR6p39gdx9n2mgj/", file_name)
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"shapeworld2")
    
    extract_dataset(directory, filepath, filepath_extracted)
            
            
def maybe_download_shapeworld4():
        
    directory = "../../data/"
    file_name= "shapeworld4.zip"
    maybe_download(directory, "https://hessenbox.tu-darmstadt.de/dl/fiEE3hftM4n1gBGn4HJLKUkU/", file_name)
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"shapeworld4")
    
    extract_dataset(directory, filepath, filepath_extracted)

    
def maybe_download_shapeworld_cogent():
        
    directory = "../../data/"
    file_name= "shapeworld_cogent.zip"
    maybe_download(directory, "https://hessenbox.tu-darmstadt.de/dl/fi3CDjPRsYgAvotHcC8GPaWj/", file_name)
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"shapeworld_cogent")
    
    extract_dataset(directory, filepath, filepath_extracted)
    
def maybe_download_shapeworld_ood():
        
    directory = "../../data/"
    file_name= "shapeworld_ood.zip"
    maybe_download(directory, "https://hessenbox.tu-darmstadt.de/dl/fiEt6BYhgFiSBYBnG4MKr2Jg/", file_name)
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"shapeworld_ood")
    
    extract_dataset(directory, filepath, filepath_extracted)
    
    

def maybe_download_covtype():
    directory = "../../data/"
    file_name= "covtype.data.gz"
    maybe_download(directory, "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/", file_name, suffix='.gz')
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"covtype/covtype.data")
    
    if not os.path.isdir(os.path.join(directory,"covtype")):
        utils.mkdir_p(os.path.join(directory,"covtype"))
        
    with gzip.open(filepath, 'rb') as s_file, open(filepath_extracted, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, 65536)


