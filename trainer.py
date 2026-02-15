import torch
import torch.nn as nn
from torch.nn.functional import silu
from lazy_loader_thing import TiffDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def calculate_iou(outputs, masks, threshold=0.5, epsilon=1e-6):
    # Apply sigmoid to convert logits to probabilities [0, 1]
    preds = torch.sigmoid(outputs)
    
    # Convert probabilities to binary mask (0 or 1)
    preds = (preds > threshold).float()
    
    # Flatten the tensors to 1D arrays for easier calculation
    # (Batch, Channel, Height, Width) -> (N,)
    preds = preds.view(-1)
    masks = masks.view(-1)
    
    # Calculate Intersection and Union
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    
    # Calculate IoU (add epsilon to avoid division by zero)
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou.item()


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # contracting/downsampling/encoder
        # 1024x1024x3
        self.en1c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.en1n1 = nn.GroupNorm(8, 32)
        self.en1c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.en1n2 = nn.GroupNorm(8, 32)
        self.en1p = nn.MaxPool2d(2, 2)

        # 512x512x32
        self.en2c1 = nn.Conv2d(32, 64, 3, padding=1)
        self.en2n1 = nn.GroupNorm(8, 64)
        self.en2c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.en2n2 = nn.GroupNorm(8, 64)
        self.en2p = nn.MaxPool2d(2, 2)

        # 256x256x64
        self.en3c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.en3n1 = nn.GroupNorm(8, 128)
        self.en3c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.en3n2 = nn.GroupNorm(8, 128)
        self.en3p = nn.MaxPool2d(2, 2)

        # 128x128x128
        self.en4c1 = nn.Conv2d(128, 256, 3, padding=1)
        self.en4n1 = nn.GroupNorm(8, 256)
        self.en4c2 = nn.Conv2d(256, 256, 3, padding=1)
        self.en4n2 = nn.GroupNorm(8, 256)
        self.en4p = nn.MaxPool2d(2, 2)

        # 64x64x256
        self.en5c1 = nn.Conv2d(256, 512, 3, padding=1)
        self.en5n1 = nn.GroupNorm(8, 512)
        self.en5c2 = nn.Conv2d(512, 512, 3, padding=1)
        self.en5n2 = nn.GroupNorm(8, 512)

        # expanding/upsamplimg/decoder
        # 64x64x512
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.de4c1 = nn.Conv2d(512, 256, 3, padding=1) # 512 again because de4c1 receives up4 + skip connection (256 + 256 = 512)
        self.de4n1 = nn.GroupNorm(8, 256)
        self.de4c2 = nn.Conv2d(256, 256, 3, padding=1)
        self.de4n2 = nn.GroupNorm(8, 256)

        # 128x128x256
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.de3c1 = nn.Conv2d(256, 128, 3, padding=1)
        self.de3n1 = nn.GroupNorm(8, 128)
        self.de3c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.de3n2 = nn.GroupNorm(8, 128)

        # 256x256x128
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.de2c1 = nn.Conv2d(128, 64, 3, padding=1)
        self.de2n1 = nn.GroupNorm(8, 64)
        self.de2c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.de2n2 = nn.GroupNorm(8, 64)

        # 512x512x64
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.de1c1 = nn.Conv2d(64, 32, 3, padding=1)
        self.de1n1 = nn.GroupNorm(8, 32)
        self.de1c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.de1n2 = nn.GroupNorm(8, 32)

        # Output layer (no normalization here)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        # -------- Encoder --------
        xe1c1 = silu(self.en1n1(self.en1c1(x)))
        xe1c2 = silu(self.en1n2(self.en1c2(xe1c1)))
        xp1 = self.en1p(xe1c2)

        xe2c1 = silu(self.en2n1(self.en2c1(xp1)))
        xe2c2 = silu(self.en2n2(self.en2c2(xe2c1)))
        xp2 = self.en2p(xe2c2)

        xe3c1 = silu(self.en3n1(self.en3c1(xp2)))
        xe3c2 = silu(self.en3n2(self.en3c2(xe3c1)))
        xp3 = self.en3p(xe3c2)

        xe4c1 = silu(self.en4n1(self.en4c1(xp3)))
        xe4c2 = silu(self.en4n2(self.en4c2(xe4c1)))
        xp4 = self.en4p(xe4c2)

        xe5c1 = silu(self.en5n1(self.en5c1(xp4)))
        xe5c2 = silu(self.en5n2(self.en5c2(xe5c1)))

        # -------- Decoder --------
        xup4 = self.up4(xe5c2)
        xup4_con = torch.cat([xup4, xe4c2], dim=1)
        xd41 = silu(self.de4n1(self.de4c1(xup4_con)))
        xd42 = silu(self.de4n2(self.de4c2(xd41)))

        xup3 = self.up3(xd42)
        xup3_con = torch.cat([xup3, xe3c2], dim=1)
        xd31 = silu(self.de3n1(self.de3c1(xup3_con)))
        xd32 = silu(self.de3n2(self.de3c2(xd31)))

        xup2 = self.up2(xd32)
        xup2_con = torch.cat([xup2, xe2c2], dim=1)
        xd21 = silu(self.de2n1(self.de2c1(xup2_con)))
        xd22 = silu(self.de2n2(self.de2c2(xd21)))

        xup1 = self.up1(xd22)
        xup1_con = torch.cat([xup1, xe1c2], dim=1)
        xd11 = silu(self.de1n1(self.de1c1(xup1_con)))
        xd12 = silu(self.de1n2(self.de1c2(xd11)))

        out = self.out(xd12)

        return out


## run space
if __name__ == "__main__":
    dataset = TiffDataset()

    dataset_size = len(dataset)

    # TiffDataset() is not set to create splits and i cba to change all that now. im going to manually create the splits - 70%, 15%, 15%
    train_size = int(0.7 * dataset_size)
    val_size   = int(0.15 * dataset_size)
    test_size  = dataset_size - train_size - val_size # im not usign int(0.15*dataset_size) bc i want the remainder. if int(0.15*dataset_size) were used, val_zise and test_size would be same which is not possible as int conversion will perform int division and thus val_size cannot = test_size. the offset must be there

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda")

    #checkpointing 
    best_val = float("inf")

    for epoch in range(100):

        # training
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        prog_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for images, masks in prog_bar:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
                iou = calculate_iou(outputs, masks)
                train_iou += iou

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            prog_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for images, masks in prog_bar:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_iou += calculate_iou(outputs, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        if val_iou < best_val:
            best_val = val_iou
            # save best model
            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
            }, "best_model.pth")


        print(f"Epoch {epoch+1} | Train (IoU): {train_iou:.4f} | Val (IoU): {val_iou:.4f}")


    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        
        for images, masks in tqdm(test_loader):

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")

    model.eval()

    images, masks = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

    img = images[0].cpu().permute(1,2,0)
    mask = masks[0][0]
    pred = preds[0][0].cpu()

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(mask, cmap="gray")

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred, cmap="gray")

    plt.show()

