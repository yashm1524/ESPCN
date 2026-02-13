# Import Dependencies
import os
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model import ESPCN
from utils import calculate_psnr


def prepare_image(lr_image_path, hr_image_path, device, scaling_factor):
    """ Function to prepare LR/HR images for inference
    
    :param lr_image_path: path to LR image
    :param hr_image_path: path to HR image (optional, for PSNR calculation)
    :param device: 'cpu', 'cuda'
    :param scaling_factor: upscaling factor
    :return: lr Y channel tensor, hr Y channel tensor, bicubic YCbCr, bicubic RGB
    """
    
    # Load LR image
    lr_image = Image.open(lr_image_path).convert('RGB')
    
    # Generate Bicubic upscaled image for comparison
    bicubic_image = lr_image.resize(
        (lr_image.width * scaling_factor, lr_image.height * scaling_factor),
        resample=Image.BICUBIC
    )
    
    # Convert to numpy
    lr_image_np = np.array(lr_image).astype(np.float32)
    bicubic_image_np = np.array(bicubic_image).astype(np.float32)
    
    # Convert RGB to YCbCr
    lr_ycrcb = cv2.cvtColor(lr_image_np, cv2.COLOR_RGB2YCrCb)
    bicubic_ycrcb = cv2.cvtColor(bicubic_image_np, cv2.COLOR_RGB2YCrCb)
    
    # Extract Y channel
    lr_y = lr_ycrcb[:, :, 0]
    
    # Normalize
    lr_y = lr_y / 255.0
    bicubic_image_np = bicubic_image_np / 255.0
    
    # Convert to torch tensor
    lr_y = torch.from_numpy(lr_y).to(device)
    lr_y = lr_y.unsqueeze(0).unsqueeze(0)
    
    # Load HR image if provided (for PSNR calculation)
    hr_y = None
    hr_image_rgb = None
    if hr_image_path and os.path.exists(hr_image_path):
        hr_image = Image.open(hr_image_path).convert('RGB')
        hr_image_rgb = np.array(hr_image).astype(np.float32) / 255.0
        hr_image_np = np.array(hr_image).astype(np.float32)
        hr_ycrcb = cv2.cvtColor(hr_image_np, cv2.COLOR_RGB2YCrCb)
        hr_y = hr_ycrcb[:, :, 0] / 255.0
        hr_y = torch.from_numpy(hr_y).to(device)
        hr_y = hr_y.unsqueeze(0).unsqueeze(0)
    
    return lr_y, hr_y, bicubic_ycrcb, bicubic_image_np, hr_image_rgb


def infer(args):
    """ Function to perform inference on test images
    
    :param args: command line arguments
    """
    
    # Select Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    print(f"Using device: {device}")
    
    # Load Model
    model = ESPCN(num_channels=1, scaling_factor=args.scaling_factor)
    model.load_state_dict(torch.load(args.fpath_weights, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {args.fpath_weights}")
    
    # Prepare images
    lr_y, hr_y, bicubic_ycrcb, bicubic_rgb, hr_rgb = prepare_image(
        args.fpath_lr_image,
        args.fpath_hr_image,
        device,
        args.scaling_factor
    )
    
    # Run inference
    with torch.no_grad():
        preds = model(lr_y).clamp(0.0, 1.0)
    
    # Calculate PSNR if HR image is provided
    if hr_y is not None:
        psnr_hr_sr = calculate_psnr(hr_y, preds)
        print(f'PSNR (HR vs SR): {psnr_hr_sr:.2f} dB')
    else:
        psnr_hr_sr = None
        print("No HR image provided, skipping PSNR calculation")
    
    # Convert prediction back to RGB
    preds_y = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds_y, bicubic_ycrcb[..., 1], bicubic_ycrcb[..., 2]]).transpose([1, 2, 0])
    output_rgb = np.clip(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB), 0.0, 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_rgb)
    
    # Save output
    output_filename = os.path.basename(args.fpath_lr_image).replace(
        ".png", f"_espcn_x{args.scaling_factor}.png"
    )
    output_path = os.path.join(args.dirpath_out, output_filename)
    output_image.save(output_path)
    print(f"SR image saved to: {output_path}")
    
    # Save bicubic for comparison
    bicubic_filename = os.path.basename(args.fpath_lr_image).replace(
        ".png", f"_bicubic_x{args.scaling_factor}.png"
    )
    bicubic_path = os.path.join(args.dirpath_out, bicubic_filename)
    bicubic_image_pil = Image.fromarray((bicubic_rgb * 255).astype(np.uint8))
    bicubic_image_pil.save(bicubic_path)
    print(f"Bicubic image saved to: {bicubic_path}")
    
    # Plot comparison
    if hr_rgb is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(hr_rgb)
        axes[0].set_title("Original HR Image")
        axes[0].axis('off')
        
        axes[1].imshow(bicubic_rgb)
        axes[1].set_title(f"Bicubic Upscaling (x{args.scaling_factor})")
        axes[1].axis('off')
        
        axes[2].imshow(output_rgb / 255.0)
        axes[2].set_title(f"ESPCN SR (x{args.scaling_factor})" + 
                         (f"\nPSNR: {psnr_hr_sr:.2f} dB" if psnr_hr_sr else ""))
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(bicubic_rgb)
        axes[0].set_title(f"Bicubic Upscaling (x{args.scaling_factor})")
        axes[0].axis('off')
        
        axes[1].imshow(output_rgb / 255.0)
        axes[1].set_title(f"ESPCN SR (x{args.scaling_factor})")
        axes[1].axis('off')
    
    plt.tight_layout()
    result_path = os.path.join(args.dirpath_out, "comparison_result.png")
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {result_path}")
    
    if args.show_plot:
        plt.show()
    plt.close()


def build_parser():
    parser = ArgumentParser(prog="ESPCN Inference for GOPRO x4")
    parser.add_argument("-w", "--fpath_weights", required=True, type=str,
                        help="Path to trained model weights (.pth file)")
    parser.add_argument("--fpath_lr_image", required=True, type=str,
                        help="Path to LR image for inference")
    parser.add_argument("--fpath_hr_image", required=False, type=str, default=None,
                        help="Path to HR image (optional, for PSNR calculation)")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Path to output directory")
    parser.add_argument("-sf", "--scaling_factor", default=4, type=int,
                        help="Image upscaling factor (default: 4)")
    parser.add_argument("--show_plot", action="store_true",
                        help="Display matplotlib plot")
    
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    
    # Create output directory
    os.makedirs(args.dirpath_out, exist_ok=True)
    
    # Run inference
    infer(args)