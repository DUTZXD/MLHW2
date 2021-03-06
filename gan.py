import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

batch_size = 100
epochs = 50
z_dim = 100
ngf = 64
ndf = 64

g_loss_list = []
d_loss_list = []


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.generator(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out


class MyData(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgs.append(line)
        self.imgs = imgs
        self.transform = transform
        fh.close()

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.load_state_dict(torch.load('g_net_params.pkl'))
    discriminator.load_state_dict(torch.load('d_net_params.pkl'))

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))

    real_label = torch.ones(size=(batch_size, 1, 1, 1), requires_grad=False).to(device)
    fake_label = torch.zeros(size=(batch_size, 1, 1, 1), requires_grad=False).to(device)

    dataset = MyData('./data.txt', transform=transforms.Compose([transforms.Resize(96), transforms.ToTensor()]))
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs+1):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        for batch_index, images in enumerate(data_loader):
            optimizer_D.zero_grad()
            images = images.to(device)
            d_out_real = discriminator(images)
            real_loss = criterion(d_out_real, real_label)
            real_loss.backward()
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = generator(noise)
            d_out_fake = discriminator(fake.detach())
            fake_loss = criterion(d_out_fake, fake_label)
            fake_loss.backward()
            d_loss = real_loss + fake_loss
            optimizer_D.step()

            optimizer_G.zero_grad()
            d_out_fake = discriminator(fake)
            g_loss = criterion(d_out_fake, real_label)
            g_loss.backward()
            optimizer_G.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            print("[%d/%d], [%d/%d], Loss_D: %.3f, Loss_G: %.3f" % (epoch, epochs, batch_index, len(data_loader), d_loss.item(), g_loss.item()))
            if epoch % 5 == 0:
                vutils.save_image(fake.data, './result/fake_result_%03d.png' % epoch, normalize=True)
        g_loss_list.append(g_loss_sum)
        d_loss_list.append(d_loss_sum)
        torch.save(generator.state_dict(), 'g_net_params.pkl')
        torch.save(discriminator.state_dict(), 'd_net_params.pkl')

