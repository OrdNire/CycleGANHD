import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
import time
from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from datasets import ImageDataset
import itertools
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-trainmodel", default="vangogh2photo", help="your training datasets")
parser.add_argument("-epoch", default=23, type=int, help="training epoch")
parser.add_argument("-learningrate", default=0.0002, type=float, help="learning rate")
parser.add_argument("-size", default=256, type=int, help="image size")
parser.add_argument("-fromepoch", default=0, type=int, help="continue training as your epoch")
args = parser.parse_args()

TIMESTAMP = time.time()
save_dir = "./checkpoints"
run_dir = "./runs/" + str(TIMESTAMP)
data_dir = "./datasets/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 1
size = args.size
lr = args.learningrate
n_epoch = args.epoch
epoch = args.fromepoch
decay_epoch = n_epoch // 2

if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # model
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    if epoch > 0:
        # 加载权重
        netG_A2B.load_state_dict(torch.load("checkpoints/netG_A2B{}.pth".format(epoch - 1)))
        netG_B2A.load_state_dict(torch.load("checkpoints/netG_B2A{}.pth".format(epoch - 1)))
        netD_A.load_state_dict(torch.load("checkpoints/netD_A{}.pth".format(epoch - 1)))
        netD_B.load_state_dict(torch.load("checkpoints/netD_B{}.pth".format(epoch - 1)))


    # loss
    # loss参考论文论文
    loss_GAN = torch.nn.MSELoss()
    loss_cycle = torch.nn.L1Loss()
    # 论文中保持生成图像的颜色一直利用的identity loss
    loss_identity = torch.nn.L1Loss()

    # optimizer & LR
    opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                             lr=lr, betas=(0.5, 0.999))
    opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G,
                                                       lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA,
                                                       lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
    lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB,
                                                       lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)

    data_root = os.path.join(data_dir, args.trainmodel)
    input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    label_real = torch.ones([batchsize, 1], requires_grad=False, dtype=torch.float).to(device)
    label_fake = torch.zeros([batchsize, 1], requires_grad=False, dtype=torch.float).to(device)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    writer_log = SummaryWriter(run_dir)

    # 数据增强
    transforms_ = [
        transforms.Resize(int(256 * 1.12), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataloader = DataLoader(ImageDataset(data_root, transforms_), batch_size=batchsize,
                            shuffle=True, num_workers=4)
    datalen = len(dataloader)
    step = 0
    while epoch < n_epoch:
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        epoch_loss_G = 0
        epoch_loss_G_identity = 0
        epoch_loss_G_GAN = 0
        epoch_loss_G_cycle = 0
        epoch_loss_D_A = 0
        epoch_loss_D_B = 0
        t = tqdm(dataloader, leave=False, total=datalen)
        for i, batch in enumerate(t):
            real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
            real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

            # ---- generator loss----
            opt_G.zero_grad()
            # identity_loss，参考论文公式
            same_B = netG_A2B(real_B)
            loss_identity_B = loss_identity(same_B, real_B) * 5.0

            same_A = netG_B2A(real_A)
            loss_identity_A = loss_identity(same_A, real_A) * 5.0

            # gan_loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = loss_GAN(pred_fake, label_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = loss_GAN(pred_fake, label_real)

            # cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0
            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0

            loss_G = loss_identity_A + loss_identity_B \
                    +loss_GAN_A2B + loss_GAN_B2A \
                    +loss_cycle_ABA + loss_cycle_BAB

            loss_G.backward()
            opt_G.step()

            epoch_loss_G += loss_G
            epoch_loss_G_identity += loss_identity_A + loss_identity_B
            epoch_loss_G_GAN += loss_GAN_A2B + loss_GAN_B2A
            epoch_loss_G_cycle += loss_cycle_ABA + loss_cycle_BAB

            # ---- discriminator loss----
            # DA
            opt_DA.zero_grad()

            pred_real = netD_A(real_A)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach()) # 防止优化D时候改变了G的参数，将梯度截断
            loss_D_fake = loss_GAN(pred_fake, label_fake)

            # total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            opt_DA.step()

            # DB
            opt_DB.zero_grad()

            pred_real = netD_B(real_B)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_fake, label_fake)

            # total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_DB.step()

            epoch_loss_D_A += loss_D_A
            epoch_loss_D_B += loss_D_B

            t.set_postfix({
                "loss_G": "{:6f}".format(loss_G / batchsize),
                "loss_G_identity" : "{:6f}".format((loss_identity_A + loss_identity_B) / batchsize),
                "loss_G_GAN" : "{:6f}".format((loss_GAN_B2A + loss_GAN_A2B) / batchsize),
                "loss_G_cycle" : "{:6f}".format((loss_cycle_BAB + loss_cycle_ABA) / batchsize),
                "loss_D_A" : "{:6f}".format(loss_D_A / batchsize),
                "loss_D_B" : "{:6f}".format(loss_D_B / batchsize),
                "epoch": "{:02d}".format(epoch)
            })

            writer_log.add_scalar("loss_G", loss_G / batchsize, global_step=step+1)
            writer_log.add_scalar("loss_G_identity", (loss_identity_A + loss_identity_B) / batchsize, global_step=step + 1)
            writer_log.add_scalar("loss_G_GAN", (loss_GAN_B2A + loss_GAN_A2B) / batchsize, global_step=step + 1)
            writer_log.add_scalar("loss_G_cycle", (loss_cycle_BAB + loss_cycle_ABA) / batchsize, global_step=step + 1)
            writer_log.add_scalar("loss_D_A", loss_D_A / batchsize, global_step=step + 1)
            writer_log.add_scalar("loss_D_B", loss_D_B / batchsize, global_step=step + 1)

            step += 1

        # 学习率衰减
        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()

        # print("loss_G:{}, loss_G_identity:{}, loss_G_GAN, "
        #       "loss_G_cycle:{}, loss_D_A:{}, loss_D_B:{}".format(
        #     epoch_loss_G,
        #     epoch_loss_G_identity,
        #     epoch_loss_G_GAN,
        #     epoch_loss_G_cycle,
        #     epoch_loss_D_A,
        #     epoch_loss_D_B
        # ))

        writer_log.add_scalar("epoch_loss_G", epoch_loss_G / datalen, global_step=epoch)
        writer_log.add_scalar("epoch_loss_G_identity", epoch_loss_G_identity / datalen, global_step=epoch)
        writer_log.add_scalar("epoch_loss_G_GAN", epoch_loss_G_GAN / datalen, global_step=epoch)
        writer_log.add_scalar("epoch_loss_G_cycle", epoch_loss_G_cycle / datalen, global_step=epoch)
        writer_log.add_scalar("epoch_loss_D_A", epoch_loss_D_A / datalen, global_step=epoch)
        writer_log.add_scalar("epoch_loss_D_B", epoch_loss_D_B / datalen, global_step=epoch)


        torch.save(netG_A2B.state_dict(), os.path.join(save_dir, "netG_A2B{}.pth".format(epoch)))
        torch.save(netG_B2A.state_dict(), os.path.join(save_dir, "netG_B2A{}.pth".format(epoch)))
        torch.save(netD_A.state_dict(), os.path.join(save_dir, "netD_A{}.pth".format(epoch)))
        torch.save(netD_B.state_dict(), os.path.join(save_dir, "netD_B{}.pth".format(epoch)))
        epoch += 1