#测试 interpolate函数
from PIl import Image
import torch.nn.functional as f
from torchvision import transform

file = "./or.png"
img = Image.open(file)
print(img.size)   #(2200, 1200)

t = transforms.Compose([
		transforms.CenterCrop([512,512]),
		transforms.ToTensor(),
	])

img_tensor = t(img)
print(img_tensor.shape)  #torch.Size([3, 512, 512])

img_tensor = img_tensor.reshape([1,3,512,512])  #由于interpolate函数只能接受4D输入
img_bilinear = f.interpolate( img_tensor, mode = "bilinear", scale_factor = 0.25 )  #图像缩小为原来四分之一
img_bilinear.reshape([3,128,128])
print(img_bilinear.shape) #torch.Size([3, 128, 128])

img_pil_img = transforms.ToPILImage()(img_bilinear).convert('RGB')
img_pil_img.show()
img_pil_img.shape('./bilinear_from_torch.png')
