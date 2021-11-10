import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        def unet_encoder(in_channels, out_channels, normalize=True, drop_prob=0.):
            layer = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize: layer.append(nn.InstanceNorm2d(out_channels))
            layer.append(nn.LeakyReLU(),)
            layer.append(nn.Dropout(drop_prob))
            block = nn.Sequential(*layer)
            return block

        def unet_decoder(in_channels, out_channels, normalize=True, drop_prob=0.):
            layer = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize: layer.append(nn.InstanceNorm2d(out_channels))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(drop_prob))
            block = nn.Sequential(*layer)
            return block

        self.enc1 = unet_encoder(self.input_channels, 64, normalize=False)
        self.enc2 = unet_encoder(64, 128)
        self.enc3 = unet_encoder(128, 256)
        self.enc4 = unet_encoder(256, 512, drop_prob=0.5)
        self.enc5 = unet_encoder(512, 512, drop_prob=0.5)
        self.enc6 = unet_encoder(512, 512, drop_prob=0.5)
        self.enc7 = unet_encoder(512, 512, drop_prob=0.5)
        self.neck = unet_encoder(512, 512, normalize=False)
        self.dec1 = unet_decoder(512, 512, drop_prob=0.5)
        self.dec2 = unet_decoder(1024, 512, drop_prob=0.5)
        self.dec3 = unet_decoder(1024, 512, drop_prob=0.5)
        self.dec4 = unet_decoder(1024, 512, drop_prob=0.5)
        self.dec5 = unet_decoder(1024, 256)
        self.dec6 = unet_decoder(512, 128)
        self.dec7 = unet_decoder(256, 64)
        self.gen = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, self.output_channels, kernel_size=4, padding=1),
            nn.Tanh(),
        )

    def forward(self, condition_img):
        o1 = self.enc1(condition_img)
        o2 = self.enc2(o1)
        o3 = self.enc3(o2)
        o4 = self.enc4(o3)
        o5 = self.enc5(o4)
        o6 = self.enc6(o5)
        o7 = self.enc7(o6)
        o8 = self.neck(o7)
        o9 = torch.cat((self.dec1(o8), o7), 1)
        o10 = torch.cat((self.dec2(o9), o6), 1)
        o11 = torch.cat((self.dec3(o10), o5), 1)
        o12 = torch.cat((self.dec4(o11), o4), 1)
        o13 = torch.cat((self.dec5(o12), o3), 1)
        o14 = torch.cat((self.dec6(o13), o2), 1)
        o15 = torch.cat((self.dec7(o14), o1), 1)
        out = self.gen(o15)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels

        def cnn_block(in_channels, out_channels, normalize=True, drop_prob=0.):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize: block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU())
            block.append(nn.Dropout(drop_prob))
            return block
        self.model = nn.Sequential(
            *cnn_block(self.input_channels*2, 64, normalize=False),
            *cnn_block(64, 128),
            *cnn_block(128, 256),
            *cnn_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img, condition_img):
        input = torch.cat((img, condition_img), 1)
        out = self.model(input)
        return out


class Model(nn.Module):
    def __init__(self, input_channels, output_channels, batch_size, lamda=1):
        super(Model, self).__init__()
        self.generator = Generator(input_channels, output_channels)
        self.discriminator = Discriminator(input_channels)
        self.batch_size = batch_size
        self.lamda = lamda
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()

    def loss_gen(self, img, condition_img):
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = self.MSE_loss(torch.ones_like(fake_img_discriminant), fake_img_discriminant)
        L1_loss = self.L1_loss(img, generated_img) * self.lamda
        gen_loss = fake_disc_loss + L1_loss
        return gen_loss

    def loss_dis(self, img, condition_img):
        real_img_discriminant = self.discriminator(img, condition_img)
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = self.MSE_loss(torch.zeros_like(fake_img_discriminant), fake_img_discriminant)
        real_disc_loss = self.MSE_loss(torch.ones_like(real_img_discriminant), real_img_discriminant)
        disc_loss = (fake_disc_loss + real_disc_loss) / 2
        return disc_loss

    '''
    def loss_gen(self, img, condition_img):
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = torch.sum(
            torch.log(torch.ones_like(fake_img_discriminant) - fake_img_discriminant)) / self.batch_size
        L1_loss = self.L1_loss(img, generated_img) * self.lamda
        gen_loss = fake_disc_loss + self.lamda * L1_loss
        return gen_loss

    def loss_dis(self, img, condition_img):
        real_img_discriminant = self.discriminator(img, condition_img)
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = torch.sum(torch.log(torch.ones_like(fake_img_discriminant) - fake_img_discriminant)) / self.batch_size
        disc_loss = -(torch.sum(torch.log(real_img_discriminant))/self.batch_size + fake_disc_loss)
        return disc_loss

    def loss(self, img, condition_img):
        real_img_discriminant = self.discriminator(img, condition_img)
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = torch.sum(torch.log(torch.ones_like(fake_img_discriminant) - fake_img_discriminant)) / self.batch_size
        L1_loss = self.L1_loss(img, generated_img) * self.lamda
        disc_loss = -(torch.sum(torch.log(real_img_discriminant))/self.batch_size + fake_disc_loss)
        gen_loss = fake_disc_loss + self.lamda * L1_loss
        return disc_loss, gen_loss
        
        
    def loss_gen(self, img, condition_img):
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = self.MSE_loss(torch.ones_like(fake_img_discriminant), fake_img_discriminant)
        L1_loss = self.L1_loss(img, generated_img) * self.lamda
        gen_loss = fake_disc_loss + L1_loss
        return gen_loss

    def loss_dis(self, img, condition_img):
        real_img_discriminant = self.discriminator(img, condition_img)
        generated_img = self.generator(condition_img)
        fake_img_discriminant = self.discriminator(generated_img, condition_img)
        fake_disc_loss = self.MSE_loss(torch.zeros_like(fake_img_discriminant), fake_img_discriminant)
        real_disc_loss = self.MSE_loss(torch.ones_like(real_img_discriminant), real_img_discriminant)
        disc_loss = (fake_disc_loss + real_disc_loss) / 2
        return disc_loss
    '''





