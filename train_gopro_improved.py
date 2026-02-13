# Import Dependencies
import os
import copy
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import AverageMeter, calculate_psnr
from model_improved import ESPCN  # Using improved model
from dataloader_gopro_augmented import get_data_loader  # Using augmented dataloader


def train(model, train_loader, device, criterion, optimizer):
    """ Function to train the model

    :param model: instance of model
    :param train_loader: training data loader
    :param device: training device, 'cpu', 'cuda'
    :param criterion: loss criterion, MSE
    :param optimizer: model optimizer, Adam
    :return: running training loss
    """

    model.train()
    running_loss = AverageMeter()

    for data in train_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        prediction = model(inputs)
        loss = criterion(prediction, labels)
        running_loss.update(loss.item(), len(inputs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss


def evaluate(model, val_loader, device, criterion):
    """ Function to evaluate the model

    :param model: instance of the model
    :param val_loader: validation data loader
    :param device: training device, 'cpu', 'cuda'
    :param criterion: loss criterion
    :return: model predictions, running PSNR, running validation loss
    """

    # Evaluate the Model
    model.eval()
    running_psnr = AverageMeter()
    running_loss = AverageMeter()

    for data in val_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
            loss = criterion(preds, labels)
            running_loss.update(loss.item(), len(inputs))

        running_psnr.update(calculate_psnr(preds, labels), len(inputs))

    print('eval psnr: {:.2f}'.format(running_psnr.avg))

    return preds, running_psnr, running_loss


def main(args):
    """ Main function to train/evaluate the model

    :param args: model input arguments
    :return: best trained model
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(f"Device: {device}")

    # Set seed for reproducability
    torch.manual_seed(args.seed)

    print("\n========================================")
    print("ESPCN IMPROVED TRAINING")
    print("========================================")
    print("Model improvements:")
    print("  ✓ ReLU activation (better gradient flow)")
    print("  ✓ Deeper network (3 conv layers)")
    print("  ✓ More filters (128→64→32)")
    print("  ✓ Data augmentation (flips + rotations)")
    print("========================================\n")

    # Get dataloaders for GOPRO dataset structure with augmentation
    print("--- Loading GOPRO Dataset ---")
    train_loader, val_loader = get_data_loader(
        dirpath_train_lr=args.dirpath_train_lr,
        dirpath_train_hr=args.dirpath_train_hr,
        dirpath_val_lr=args.dirpath_val_lr,
        dirpath_val_hr=args.dirpath_val_hr,
        scaling_factor=args.scaling_factor,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        augment=True  # Enable data augmentation
    )

    # Verify data loading
    print("\n--- Verifying Data Loading ---")
    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"Training - LR: {lr_image.shape}, HR: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"Validation - LR: {lr_image.shape}, HR: {hr_image.shape}")
        break

    # Get the IMPROVED Model
    model = ESPCN(num_channels=1, scaling_factor=args.scaling_factor)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024:.2f} KB\n")

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': model.feature_map_layer.parameters()},
        # As per paper, Sec 3.2, The final layer learns 10 times slower
        {'params': model.sub_pixel_layer.parameters(), 'lr': args.learning_rate * 0.1}
    ], lr=args.learning_rate)

    # Train the Model
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    print(f"--- Starting Training for {args.epochs} epochs ---")
    print(f"Target: Beat previous PSNR of 30.36 dB")
    print(f"Expected: 32-33 dB with improvements\n")
    
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        # Learning rate decay (gentler decay at 0.9 instead of 0.8)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate * (0.1 ** (epoch // int(args.epochs * 0.9)))

        # Train
        training_loss = train(model, train_loader, device, criterion, optimizer)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.dirpath_out, f'epoch_{epoch}.pth'))

        # Evaluate
        preds, running_psnr, validation_loss = evaluate(model, val_loader, device, criterion)

        # Save best model
        if running_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = running_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.dirpath_out, 'best.pth'))
            print(f"  ⭐ New best PSNR: {best_psnr:.2f} dB (epoch {epoch})")

        # Logging
        improvement = running_psnr.avg - 30.36  # Compare to previous result
        print(f"Epoch: {epoch}, Train Loss: {training_loss.avg:.4f}, "
              f"Val Loss: {validation_loss.avg:.4f}, PSNR: {running_psnr.avg:.2f} dB "
              f"(+{improvement:.2f} dB vs baseline)")

    print(f'\n========================================')
    print(f'TRAINING COMPLETE!')
    print(f'========================================')
    print(f'Best Epoch: {best_epoch}')
    print(f'Best PSNR: {best_psnr:.2f} dB')
    print(f'Improvement: +{best_psnr - 30.36:.2f} dB vs baseline (30.36 dB)')
    print(f'========================================\n')
    
    # Save final best model
    torch.save(best_weights, os.path.join(args.dirpath_out, 'best.pth'))
    print(f'Best model saved to: {os.path.join(args.dirpath_out, "best.pth")}')

    return model.load_state_dict(best_weights)


def build_parser():
    parser = ArgumentParser(prog="ESPCN GOPRO Training - IMPROVED VERSION")
    
    # Dataset paths
    parser.add_argument("--dirpath_train_lr", required=True, type=str,
                        help="Path to training LR images (e.g., GOPRO_SR/train/LR_x4)")
    parser.add_argument("--dirpath_train_hr", required=True, type=str,
                        help="Path to training HR images (e.g., GOPRO_SR/train/HR)")
    parser.add_argument("--dirpath_val_lr", required=True, type=str,
                        help="Path to validation LR images (e.g., GOPRO_SR/test/LR_x4)")
    parser.add_argument("--dirpath_val_hr", required=True, type=str,
                        help="Path to validation HR images (e.g., GOPRO_SR/test/HR)")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Path to directory to save model weights")
    
    # Model parameters
    parser.add_argument("-sf", "--scaling_factor", default=4, type=int,
                        help="Image upscaling factor (default: 4 for GOPRO)")
    parser.add_argument("-ps", "--patch_size", default=17, type=int,
                        help="Sub-images patch size for LR images (default: 17)")
    parser.add_argument("-s", "--stride", default=13, type=int,
                        help="Sub-image extraction stride (default: 13)")
    
    # Training parameters
    parser.add_argument("--epochs", default=300, type=int,
                        help="Number of training epochs (default: 300, increased from 200)")
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("-bs", "--batch_size", default=16, type=int,
                        help="Training batch size (default: 16)")
    parser.add_argument("--seed", default=100, type=int,
                        help="Random seed for reproducibility")
    
    # Logging and saving
    parser.add_argument("--save_interval", default=25, type=int,
                        help="Save model checkpoint every n epochs")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.dirpath_out, exist_ok=True)
    
    # Train the model
    espcn_model = main(args)
