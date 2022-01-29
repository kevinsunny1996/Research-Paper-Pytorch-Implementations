import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Every contraction block on U-Net has two conv layers with 3X3 kernels, so squeezing both layers into a sequential one.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    '''
    1. out_channels is kept as 1 as the final o/p will contain b/w car segment , can be kept as 2 referring to the paper.
    2. Create a features list containing the channels that will be used per layer as part of the contraction, same for expansion.
    '''
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        '''
        Create two ModuleList objects - ups and downs
        1. downs - To store the layers in the contraction part.
        2. ups - To store the layers in the expansion part.
        3. ModuleList is used as we would need to access these layers later on.
        '''
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            # Update the in_channel with the feature as it will be used in the next part of DoubleConv
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # Upsample the expansion part of UNEt
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                    )
            )
            # Perform the double conv to complete one step
            self.ups.append(DoubleConv(feature*2, feature))

        # Middle part of UNET (512,1024)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Output part of the UNET , using 1X1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # List to store skip connections
        skip_connections = []

        # Down -> Bottleneck -> Up -> 1X1 conv
        for down in self.downs:
            x = down(x)
            # Add the skip connection before moving to next down step.
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        # Reverse the skip connections list to connect to up part in reverse.
        skip_connections = skip_connections[::-1]

        # Iterate through the up part in steps of 2 (Upsample -> Double Conv).
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # Skip connections are added from every down layer to every up layer , hence //2 to nullify the step of 2 in for loop.
            skip_connection = skip_connections[idx//2]

            # Keeping the upsampled size similar to the down layer size of that level
            if x.shape != skip_connection.shape:
                # Resize using the height and the width only
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concat along channel dim
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Running the output through double conv (ups[0] = Transpose Conv Layer, ups[1] = Double Conv layer)
            x = self.ups[idx+1](concat_skip)
        

        return self.final_conv(x)

def test():
    # kernel_size, stride, input height and width
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)

    preds = model(x)
    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape

if __name__ == "__main__":
    test()