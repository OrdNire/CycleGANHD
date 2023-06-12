import time
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

from models import Generator, Discriminator, SRResNet
from utils import ReplayBuffer, LambdaLR, weights_init_normal, convert_image,calculate_psnr,calculate_ssim
from datasets import ImageDataset
import itertools
import tensorboardX
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-usecheckpoint", default=None, help="use your trained model, please input model path")
parser.add_argument("-defaultmodel", default="vangogh", help="use default model, please inpuut model name")
parser.add_argument("-inputpath", default="./default_models/images/inputs", help="input images path")
parser.add_argument("-outputpath", default="./default_models/images/outputs", help="output images path")
parser.add_argument("-size", default=256, type=int, help="image size")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超分模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

netG_B2A = Generator().to(device)
srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor).to(device)
# 加载权重
srresnet.load_state_dict(torch.load("./default_models/srrestnet/checkpoint_srresnet.pth")['model'])

input_B = torch.ones([1, 3, args.size, args.size], dtype=torch.float).to(device)

def SRTransfer(img):
    img = convert_image(img, source="[0, 1]", target="imagenet-norm")
    # img.unsqueeze_(0)

    img = img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = srresnet(img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='[0, 1]')

    return sr_img

if __name__ == '__main__':
    data_root = args.inputpath
    output_root = args.outputpath
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 使用默认模型
    if args.usecheckpoint is None:
        if args.defaultmodel == "vangogh":
            netG_B2A.load_state_dict(torch.load("./default_models/photo2vangogh/netG_B2A13.pth"))
        else:
            raise ValueError("model not exist!")
    else:
        netG_B2A.load_state_dict(torch.load(args.usecheckpoint))

    list_imgs = os.listdir(data_root)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    start_time = time.time()
    t = tqdm(list_imgs, leave=False, total=len(list_imgs))
    for i, img_name in enumerate(t):
        img = transform(Image.open(os.path.join(data_root, img_name)))
        real_B = torch.tensor(input_B.copy_(img), dtype=torch.float).to(device)

        # 风格迁移
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # 超分重构
        fake_A = SRTransfer(fake_A)

        save_image(fake_A, os.path.join(args.outputpath, "{}.png".format(i)))

    end_time = time.time()
    print("Time: {:2f}".format(end_time - start_time))