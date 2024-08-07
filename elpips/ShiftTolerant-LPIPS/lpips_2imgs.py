import torch
import stlpips
from torch.utils.data import Dataset
from PIL import Image


class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

img0_path = '/liascratch/cabodi/sdxl/images_2/A painting of The Starry Night by Vincent van Gogh_044f1861cdb2481395f63bba05bff112.png'
img1_path = '/liascratch/cabodi/sdxl/query_images/totally_different.png'

## Initializing the model

##### LPIPS (v0.1)
# stlpips_metric = stlpips.LPIPS(net='alex')

##### LPIPS trained from scratch
# stlpips_metric = stlpips.LPIPS(net='alex', variant="vanilla")

##### ST-LPIPS (STv0.0)
# stlpips_metric = stlpips.LPIPS(net='alex', variant="antialiased")

stlpips_metric = stlpips.LPIPS(net="vgg", variant="shift_tolerant") # change as needed (e.g. vgg vanilla works well too)

if torch.cuda.is_available():
	stlpips_metric.cuda()

import torchvision.transforms as transforms
resize_transform = transforms.Resize((256, 256))

# # Load images
img0 = stlpips.im2tensor(resize_transform(stlpips.load_image(img0_path))) # RGB image from [-1,1]
img1 = stlpips.im2tensor(resize_transform(stlpips.load_image(img1_path)))

# Stack the images to create a batch
img0_batch = img0.repeat(4, 1, 1, 1)
img1_batch = img1.repeat(4, 1, 1, 1)

if torch.cuda.is_available():
	img0 = img0_batch.cuda()
	img1 = img1_batch.cuda()

# Compute distance
dist01 = stlpips_metric.forward(img0,img1)
print('Distance: ', dist01)


##### Load images in the same way as used for computing 2AFC score
# import torchvision.transforms as transforms
# from PIL import Image
# transform_list = []
# # transform_list.append(transforms.Scale(load_size)) # deprecated
# transform_list.append(transforms.Resize(64, Image.BICUBIC))
# transform_list += [transforms.ToTensor(),
# 	transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

# transform = transforms.Compose(transform_list)

# img0 = Image.open(opt.path0).convert('RGB')
# img0 = transform(img0).unsqueeze(0)

# img1 = Image.open(opt.path1).convert('RGB')
# img1 = transform(img1).unsqueeze(0)

# dist01 = stlpips_metric.forward(img0,img1)
# print('Distance: %.3f'%dist01)