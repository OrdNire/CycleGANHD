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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = 256

epoch = 13

# 超分模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
srresnet = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor).to(device)

vangogh_weights = "./final_outputs/vangogh/checkpoints"
ukiyoe_weights = "checkpoints"

# 加载权重
netG_A2B.load_state_dict(torch.load(os.path.join(vangogh_weights, "netG_A2B{}.pth".format(epoch))))
netG_B2A.load_state_dict(torch.load(os.path.join(vangogh_weights, "netG_B2A{}.pth".format(epoch))))
srresnet.load_state_dict(torch.load("checkpoints/checkpoint_srresnet.pth")['model'])


input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)



def test_all_test():
    data_root = "datasets/vangogh2photo"

    # 数据增强
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataloader = DataLoader(ImageDataset(data_root, transforms_, mode="test"), batch_size=1,
                            shuffle=False, num_workers=4)
    if not os.path.exists("outputs/A"):
        os.makedirs("outputs/A")
    if not os.path.exists("outputs/B"):
        os.makedirs("outputs/B")

    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        fake_A_hd = SRTransfer(fake_A)
        fake_B_hd = SRTransfer(fake_B)

        save_image(real_B, "outputs/origin/{}.png".format(i))
        save_image(fake_A, "outputs/A/CycleGAN/{}.png".format(i))
        save_image(fake_A_hd, "outputs/A/CycleGANHD/{}.png".format(i))
        # save_image(fake_B, "outputs/B/CycleGAN/{}.png".format(i))
        # save_image(fake_B_hd, "outputs/B/CycleGANHD/{}.png".format(i))

def test_dif_epoch():
    data_root = "datasets/imagesB"
    if not os.path.exists("outputs/epochs_ukiyoe"):
        os.makedirs("outputs/epochs_ukiyoe")

    list_imgs = os.listdir(data_root)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    n_epochs = 16
    for t_epoch in range(n_epochs + 1):
        netG_A2B.load_state_dict(torch.load(os.path.join(ukiyoe_weights, "netG_A2B{}.pth".format(t_epoch))))
        netG_B2A.load_state_dict(torch.load(os.path.join(ukiyoe_weights, "netG_B2A{}.pth".format(t_epoch))))
        for i, img_name in enumerate(["4.jpg", "7.jpg"]):
            img = transform(Image.open(os.path.join(data_root, img_name)))
            print(img.shape)
            real_B = torch.tensor(input_B.copy_(img), dtype=torch.float).to(device)

            time1 = time.time()
            # 风格迁移
            fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
            time2 = time.time()

            # 超分重构
            fake_A = SRTransfer(fake_A)
            time3 = time.time()

            print("CycleGAN用时：{}， CycleGANHD用时：{}".format(time2 - time1, time3 - time1))

            # print(fake_A.shape)
            save_image(fake_A, "outputs/epochs_ukiyoe/{}_{}.png".format(t_epoch, i))

# 风景 -> 风格
def test_image_B2A(model_weights = vangogh_weights, model_epoch=13):
    data_root = "datasets/imagesB"
    if not os.path.exists("outputs/imagesA"):
        os.makedirs("outputs/imagesA")

    if not os.path.exists("outputs/imagesA/vangogh"):
        os.makedirs("outputs/imagesA/vangogh")

    if not os.path.exists("outputs/imagesA/ukiyoe"):
        os.makedirs("outputs/imagesA/ukiyoe")

    if not os.path.exists("outputs/imagesA/vangogh/cycleGAN"):
        os.makedirs("outputs/imagesA/vangogh/cycleGAN")
    if not os.path.exists("outputs/imagesA/vangogh/cycleGANHD"):
        os.makedirs("outputs/imagesA/vangogh/cycleGANHD")

    if not os.path.exists("outputs/imagesA/ukiyoe/cycleGAN"):
        os.makedirs("outputs/imagesA/ukiyoe/cycleGAN")
    if not os.path.exists("outputs/imagesA/ukiyoe/cycleGANHD"):
        os.makedirs("outputs/imagesA/ukiyoe/cycleGANHD")

    netG_A2B.load_state_dict(torch.load(os.path.join(model_weights, "netG_A2B{}.pth".format(model_epoch))))
    netG_B2A.load_state_dict(torch.load(os.path.join(model_weights, "netG_B2A{}.pth".format(model_epoch))))

    list_imgs = os.listdir(data_root)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for i, img_name in enumerate(list_imgs):
        img = transform(Image.open(os.path.join(data_root, img_name)))
        real_B = torch.tensor(input_B.copy_(img), dtype=torch.float).to(device)

        time1 = time.time()
        # 风格迁移
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
        time2 = time.time()
        if model_weights == vangogh_weights:
            save_image(fake_A, "outputs/imagesA/vangogh/cycleGAN/{}.png".format(i))
        else:
            save_image(fake_A, "outputs/imagesA/ukiyoe/cycleGAN/{}.png".format(i))

        # 超分重构
        fake_A = SRTransfer(fake_A)
        time3 = time.time()

        print("图像：{} CycleGAN用时：{}， CycleGANHD用时：{}".format(img_name,time2 - time1, time3 - time1))

        # print(fake_A.shape)
        if model_weights == vangogh_weights:
            save_image(fake_A, "outputs/imagesA/vangogh/cycleGANHD/{}.png".format(i))
        else:
            save_image(fake_A, "outputs/imagesA/ukiyoe/cycleGANHD/{}.png".format(i))

def test_image_A2B():
    data_root = "datasets/imagesA"
    if not os.path.exists("outputs/imagesB"):
        os.makedirs("outputs/imagesB")

    list_imgs = os.listdir(data_root)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for i, img_name in enumerate(list_imgs):
        img = transform(Image.open(os.path.join(data_root, img_name)))
        print(img.shape)
        real_A = torch.tensor(input_A.copy_(img), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)

        fake_B = transforms.Resize((2000, 2000), interpolation=InterpolationMode.BILINEAR)(fake_B)

        save_image(fake_B, "outputs/imagesB/{}.png".format(i))

def SRTransfer(img):
    img = convert_image(img, source="[0, 1]", target="imagenet-norm")
    # img.unsqueeze_(0)

    img = img.to(device)  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = srresnet(img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='[0, 1]')

    return sr_img

def metrix_img():
    origin_root = "./outputs/origin"
    img1_root = "./outputs/A/CycleGAN"
    img2_root = "./outputs/A/CycleGANHD"
    transform = transforms.Compose([
        transforms.Resize(1024),
    ])
    for i in range(30):
        origin_name = os.path.join(origin_root, "{}.png".format(i))
        img1_name = os.path.join(img1_root, "{}.png".format(i))
        img2_name = os.path.join(img2_root, "{}.png".format(i))
        origin = Image.open(origin_name)
        img1 = Image.open(img1_name)
        img2 = Image.open(img2_name)
        origin = transform(origin)
        img1 = transform(img1)
        img2 = transform(img2)
        origin = np.array(origin)
        img1 = np.array(img1)
        img2 = np.array(img2)
        print("CycleGAN psnr: {:4f} CycleGANHD psnr: {:4f}".format(calculate_psnr(origin, img1), calculate_psnr(origin, img2)))
        print("CycleGAN ssim: {:4f} CycleGANHD ssim: {:4f}".format(calculate_ssim(origin, img1), calculate_ssim(origin, img2)))

if __name__ == '__main__':
    # test_all_test()

    # test_image_A2B()

    test_image_B2A(ukiyoe_weights, model_epoch=15)
    # metrix_img()
    # test_dif_epoch()