from utils import load_image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Evaluate for DataCentric Approach
def get_transform(train):
   transform = []
   transform.append(transforms.ToTensor())
   transform.append(transforms.Resize((256,256)))
   transform.append(transforms.Normalize((0.5), (0.5)))
   transform.append(transforms.RandomInvert(p=1))
   if train:
      transform.append(transforms.RandomHorizontalFlip(p=0.3))
      transform.append(transforms.RandomVerticalFlip(p=0.3))
      transform.append(transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(90, 90))],p=0.3))
      transform.append(transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(180, 180))],p=0.3))
   return transforms.Compose(transform)


class DVPDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn=None):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:
            x = self.map(self.dataset[index][0])
        else:
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]   # label
        return x, y

    def __len__(self):
        return len(self.dataset)

class WaferDataset(Dataset):
    def __init__(self, path, labels, model=None):
        self.X = path
        self.y = labels
        self.model = model
        if self.model in ["BaseNet", "RegNet", "VicNet"]:
          self.infilegrayscale = True
        else:
          self.infilegrayscale = False

        # preprocessing
        if model in ["ResNet18", "VGG16"]: # validation
            self.preprocess = transforms.Compose([
              transforms.Resize(244),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # duplicate channel as Resnet18 based on RGB
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              # Data augmentation
              transforms.RandomHorizontalFlip(p=0.5),
              transforms.RandomRotation(30),
              transforms.RandomVerticalFlip(p=0.5)
            ])
        elif model in ["BaseNet", "RegNet", "VicNet", "MaxNet"]:
          self.preprocess = transforms.Compose([
              transforms.Resize(244),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              # Data augmentation
              transforms.RandomHorizontalFlip(p=0.5),
              transforms.RandomRotation(30),
              transforms.RandomVerticalFlip(p=0.5)
            ])
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = load_image(self.X.iloc[i], self.infilegrayscale)
        label = self.y.iloc[i]
        if self.model is not None:
          image = self.preprocess(image)
        # print(image.shape)
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)

class WaferBaseDataset(Dataset):
    def __init__(self, path, labels, model=None):
        self.X = path
        self.y = labels
        self.model = model
        if self.model in ["BaseNet", "RegNet", "VicNet","OctaNet"]:
          self.infilegrayscale = True
        else:
          self.infilegrayscale = False

        # preprocessing
        if model in ["ResNet18", "VGG16"]: 
            self.preprocess = transforms.Compose([
              transforms.Resize(244),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # duplicate channel as Resnet18 based on RGB
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model in ["BaseNet", "RegNet", "VicNet", "MaxNet","OctaNet"]:
            self.preprocess = transforms.Compose([
              transforms.Resize(256),
              transforms.ToTensor()
            ])
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = load_image(self.X.iloc[i], self.infilegrayscale)
        label = self.y.iloc[i]
        if self.model is not None:
          image = self.preprocess(image)
        # print(image.shape)
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
