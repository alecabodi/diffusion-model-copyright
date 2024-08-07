import torch
import stlpips

img0_path = ''
img1_path = ''

stlpips_metric = stlpips.Custom_LPIPS() # change as needed

if torch.cuda.is_available():
	stlpips_metric.cuda()

# # Load images
img0 = stlpips.im2tensor(stlpips.load_image(img0_path)) # RGB image from [-1,1]
img1 = stlpips.im2tensor(stlpips.load_image(img1_path))

if torch.cuda.is_available():
	img0 = img0.cuda()
	img1 = img1.cuda()

# Compute distance
dist01 = stlpips_metric.forward(img0,img1)
print('Distance: %.3f'%dist01)


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