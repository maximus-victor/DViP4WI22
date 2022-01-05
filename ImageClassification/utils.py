import numpy as np
import torch
import random
from PIL import Image

def seed_libraries(SEED=42):
    # Python seeds
    random.seed(SEED)
    np.random.seed(SEED)
    # Torch seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def load_image(infilename, infilegrayscale):
    """
    This function loads an image into memory when you give it the path of the image
    """
    if infilegrayscale:
      img = Image.open(infilename).convert('L')
    else:
      img = Image.open(infilename)
    #data = np.asarray(img)
    return img

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()