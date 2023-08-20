import torch
from torch import nn
from torch.nn import functional as F

import utils.model.fastmri as fastmri
from utils.model.fastmri.data import transforms

def kspace2image(masked_kspace, attrs = None):
    image = fastmri.ifft2c(masked_kspace)

    # crop input to correct size
    #crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

    # check for FLAIR 203
    """
    if image.shape[-2] < crop_size[1]:
        crop_size = (image.shape[-2], image.shape[-2])
    """
    
    crop_size = (384, 384)
    image = transforms.complex_center_crop(image, crop_size)

    # absolute value
    image = fastmri.complex_abs(image)

    # apply Root-Sum-of-Squares if multicoil data

    image = fastmri.rss(image)

    # normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)


    return image, mean, std

class Unet_classifier(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.sens = SensitivityModel(1, 1)

        self.first_block = ConvBlock(in_chans, 2)
        self.down1 = Down(2, 4)
       
        #self.last_block = nn.Conv2d(2, out_chans, kernel_size=1)
        
        self.conv2 = ConvBlock(4, 8)
        self.down2 = Down(8, 16)
        
        self.fc = nn.Linear(16 * 96 * 96, 2, bias = True)

    def norm(self, x):
        b, h, w = x.shape
        x = x.reshape(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, kspace, mask):
        #print('input shape: ', .shape)
        input = torch.mean(kspace2image(kspace)[0], axis=0)
        input = input.unsqueeze(0)
        #print('input shape: ', input.shape)
        #input, mean, std = self.norm(input)
        input = input.unsqueeze(1)
        #print('squeezed input: ', input.shape)
        d1 = self.first_block(input)
        #print('d1 shape: {}'.format(d1.shape))
        m0 = self.down1(d1)
        #print('m0 shape: {}'.format(m0.shape))
        """
        output = self.last_block(u1)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)
        """
        conv2 = self.conv2(m0)
        output = self.down2(conv2)
        #print('output shape: {}'.format(output.shape))
        out = output.view(output.size(0), -1)
        output = self.fc(out)
        
        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)
    
    
class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

      

    def chans_to_batch_dim(self, x: torch.Tensor):
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        print(x.shape)
        # convert to image space
        x = fastmri.ifft2c(x)
        return x