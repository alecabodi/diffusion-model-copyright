from lpips_pytorch import LPIPS, lpips
import PIL.Image
import torchvision.transforms as transforms

import pyiqa
import torch

# list all available metrics
print(pyiqa.list_models())


# define as a criterion module (recommended)
criterion = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

x_path = '/liascratch/cabodi/elpips/query_images/almost_replica.png'
y_path = '/liascratch/cabodi/sdxl/images_2/A painting of The Starry Night by Vincent van Gogh_044f1861cdb2481395f63bba05bff112.png'
x = PIL.Image.open(x_path)
y = PIL.Image.open(y_path)
transform = transforms.ToTensor()
x = transform(x).unsqueeze(0)
y = transform(y).unsqueeze(0)

loss = criterion(x, y)
print('LPIPS: {:.4f}'.format(loss.item()))