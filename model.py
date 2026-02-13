# Import Dependencies
import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, num_channels, scaling_factor):
        """ ESPCN Model - IMPROVED VERSION
        
        Improvements over original:
        1. ReLU instead of Tanh (better gradient flow)
        2. More filters: 128 → 64 → 32 (vs original 64 → 32)
        3. Deeper: 3 conv layers (vs original 2)
        4. Better capacity for 4x upsampling

        :param num_channels (int): Number of channels in input image
        :param scaling_factor (int): Factor to scale-up the input image by
        """

        super(ESPCN, self).__init__()

        # IMPROVED: More filters, deeper network, ReLU activation
        self.feature_map_layer = nn.Sequential(
            # Layer 1: 5x5 conv, 128 filters (doubled from 64)
            nn.Conv2d(in_channels=num_channels, kernel_size=(5, 5), 
                     out_channels=128, padding=(2, 2)),
            nn.ReLU(inplace=True),  # Changed from Tanh
            
            # Layer 2: 3x3 conv, 64 filters (doubled from 32)
            nn.Conv2d(in_channels=128, kernel_size=(3, 3), 
                     out_channels=64, padding=(1, 1)),
            nn.ReLU(inplace=True),  # Changed from Tanh
            
            # Layer 3: 3x3 conv, 32 filters (NEW LAYER)
            nn.Conv2d(in_channels=64, kernel_size=(3, 3), 
                     out_channels=32, padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.sub_pixel_layer = nn.Sequential(
            # f3 = 3, output shape: H x W x (C x r**2)
            nn.Conv2d(in_channels=32, kernel_size=(3, 3), 
                     out_channels=num_channels * (scaling_factor ** 2), 
                     padding=(1, 1)),
            # Sub-Pixel Convolution Layer - PixelShuffle
            # rearranges: H x W x (C x r**2) => rH x rW x C
            nn.PixelShuffle(upscale_factor=scaling_factor)
        )

    def forward(self, x):
        """
        :param x: input image
        :return: model output
        """
        # inputs: H x W x C
        x = self.feature_map_layer(x)
        # output: rH x rW x C (r: scale_factor)
        out = self.sub_pixel_layer(x)
        return out


if __name__ == '__main__':
    # Print and Test model outputs with a random input Tensor
    sample_input = torch.rand(size=(1, 1, 224, 224))
    print("Input shape:", sample_input.shape)

    model = ESPCN(num_channels=1, scaling_factor=4)
    print(f"\n{model}\n")

    # Forward pass with sample input
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024:.2f} KB (float32)")
