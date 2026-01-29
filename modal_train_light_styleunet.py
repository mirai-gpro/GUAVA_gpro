# modal_train_light_styleunet.py
# Knowledge Distillation: 35ch → 96ch 軽量 StyleUNet
# Teacher: 元の StyleUNet (118MB)
# Student: RFDN ベース軽量モデル (~5-10MB)

import modal
import os

# Modal app definition
app = modal.App("light-styleunet-distillation")

# Create volume for checkpoints
volume = modal.Volume.from_name("guava-training-vol", create_if_missing=True)

# Docker image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "pillow",
        "tqdm",
        "tensorboard",
    )
)

VOLUME_PATH = "/vol"


@app.function(
    image=image,
    gpu="L4",  # or "A10G", "T4"
    timeout=6 * 3600,  # 6 hours
    volumes={VOLUME_PATH: volume},
)
def train_light_styleunet(
    num_epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    checkpoint_path: str = None,
):
    """
    Knowledge Distillation で軽量 StyleUNet を訓練

    Teacher: 元の StyleUNet (35ch → 96ch, ~118MB)
    Student: RFDN ベース (35ch → 96ch, ~5-10MB)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    from tqdm import tqdm
    import json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # 1. Teacher Model: Original StyleUNet architecture
    # ============================================================

    class ModulatedConv2d(nn.Module):
        """StyleGAN2 style modulated convolution"""
        def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True):
            super().__init__()
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.kernel_size = kernel_size
            self.demodulate = demodulate

            self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
            self.style_linear = nn.Linear(style_dim, in_ch)
            self.bias = nn.Parameter(torch.zeros(out_ch))

            nn.init.kaiming_normal_(self.weight)
            nn.init.ones_(self.style_linear.weight)
            nn.init.zeros_(self.style_linear.bias)

        def forward(self, x, style):
            B, C, H, W = x.shape

            # Style modulation
            style_mod = self.style_linear(style).view(B, 1, C, 1, 1) + 1
            weight = self.weight.unsqueeze(0) * style_mod

            # Demodulation
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(B, self.out_ch, 1, 1, 1)

            # Apply convolution per sample
            x = x.view(1, B * C, H, W)
            weight = weight.view(B * self.out_ch, C, self.kernel_size, self.kernel_size)

            padding = self.kernel_size // 2
            out = nn.functional.conv2d(x, weight, padding=padding, groups=B)
            out = out.view(B, self.out_ch, H, W) + self.bias.view(1, -1, 1, 1)

            return out

    class StyleBlock(nn.Module):
        def __init__(self, in_ch, out_ch, style_dim):
            super().__init__()
            self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, style_dim)
            self.conv2 = ModulatedConv2d(out_ch, out_ch, 3, style_dim)
            self.act = nn.LeakyReLU(0.2, True)

            if in_ch != out_ch:
                self.skip = nn.Conv2d(in_ch, out_ch, 1)
            else:
                self.skip = nn.Identity()

        def forward(self, x, style):
            skip = self.skip(x)
            x = self.act(self.conv1(x, style))
            x = self.act(self.conv2(x, style))
            return x + skip

    class TeacherStyleUNet(nn.Module):
        """Original StyleUNet (Teacher) - 35ch → 96ch"""
        def __init__(self, in_ch=35, out_ch=96, style_dim=512):
            super().__init__()

            # Encoder
            self.enc1 = StyleBlock(in_ch, 64, style_dim)
            self.enc2 = StyleBlock(64, 128, style_dim)
            self.enc3 = StyleBlock(128, 256, style_dim)
            self.enc4 = StyleBlock(256, 512, style_dim)

            # Bottleneck
            self.bottleneck = StyleBlock(512, 512, style_dim)

            # Decoder
            self.dec4 = StyleBlock(512 + 512, 256, style_dim)
            self.dec3 = StyleBlock(256 + 256, 128, style_dim)
            self.dec2 = StyleBlock(128 + 128, 64, style_dim)
            self.dec1 = StyleBlock(64 + 64, 64, style_dim)

            # Output
            self.out_conv = nn.Conv2d(64, out_ch, 1)

            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        def forward(self, x, style):
            # Encoder
            e1 = self.enc1(x, style)
            e2 = self.enc2(self.pool(e1), style)
            e3 = self.enc3(self.pool(e2), style)
            e4 = self.enc4(self.pool(e3), style)

            # Bottleneck
            b = self.bottleneck(self.pool(e4), style)

            # Decoder with skip connections
            d4 = self.dec4(torch.cat([self.up(b), e4], dim=1), style)
            d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1), style)
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), style)
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), style)

            return self.out_conv(d1)

    # ============================================================
    # 2. Student Model: Lightweight RFDN-based architecture
    # ============================================================

    class ESA(nn.Module):
        """Enhanced Spatial Attention"""
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
            self.conv2 = nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1, groups=channels // reduction)
            self.conv3 = nn.Conv2d(channels // reduction, channels, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # Global pooling
            avg = x.mean(dim=[2, 3], keepdim=True)

            # Channel attention
            att = self.conv1(avg)
            att = nn.functional.relu(att)
            att = self.conv2(att)
            att = nn.functional.relu(att)
            att = self.conv3(att)

            return x * self.sigmoid(att)

    class RFDB(nn.Module):
        """Residual Feature Distillation Block"""
        def __init__(self, channels, distill_rate=0.25):
            super().__init__()
            distill_ch = int(channels * distill_rate)
            remain_ch = channels - distill_ch

            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

            self.distill_conv = nn.Conv2d(channels, distill_ch, 1)
            self.remain_conv = nn.Conv2d(channels, remain_ch, 1)

            self.esa = ESA(channels)
            self.act = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            out1 = self.act(self.conv1(x))
            out2 = self.act(self.conv2(out1))
            out3 = self.act(self.conv3(out2))

            # Feature distillation
            distill = self.distill_conv(out3)
            remain = self.remain_conv(out3)

            out = torch.cat([distill, remain], dim=1)
            out = self.esa(out)

            return out + x

    class LightStyleUNet(nn.Module):
        """
        Lightweight StyleUNet (Student)
        35ch → 96ch with ~5-10MB parameters
        Uses RFDN blocks instead of heavy StyleBlocks
        """
        def __init__(self, in_ch=35, out_ch=96, base_ch=48, style_dim=512):
            super().__init__()

            # Style projection (simplified)
            self.style_proj = nn.Sequential(
                nn.Linear(style_dim, base_ch * 4),
                nn.LeakyReLU(0.2, True),
                nn.Linear(base_ch * 4, base_ch * 4),
            )

            # Input conv
            self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

            # Encoder (lightweight)
            self.enc1 = RFDB(base_ch)
            self.enc2 = RFDB(base_ch)
            self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

            self.enc3 = RFDB(base_ch * 2)
            self.enc4 = RFDB(base_ch * 2)
            self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

            # Bottleneck
            self.bottleneck1 = RFDB(base_ch * 4)
            self.bottleneck2 = RFDB(base_ch * 4)

            # Style modulation at bottleneck
            self.style_mod = nn.Conv2d(base_ch * 4, base_ch * 4, 1)

            # Decoder
            self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
            self.dec1 = RFDB(base_ch * 2)
            self.dec2 = RFDB(base_ch * 2)

            self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
            self.dec3 = RFDB(base_ch)
            self.dec4 = RFDB(base_ch)

            # Output
            self.output_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

            self.act = nn.LeakyReLU(0.2, True)

        def forward(self, x, style):
            # Style embedding
            style_emb = self.style_proj(style)  # [B, base_ch*4]

            # Input
            f = self.act(self.input_conv(x))

            # Encoder
            f = self.enc1(f)
            f1 = self.enc2(f)  # Skip connection

            f = self.act(self.down1(f1))
            f = self.enc3(f)
            f2 = self.enc4(f)  # Skip connection

            f = self.act(self.down2(f2))

            # Bottleneck with style modulation
            f = self.bottleneck1(f)

            # Apply style modulation
            B, C, H, W = f.shape
            style_scale = style_emb.view(B, C, 1, 1)
            f = f * (1 + self.style_mod(style_scale.expand(-1, -1, H, W)))

            f = self.bottleneck2(f)

            # Decoder
            f = self.act(self.up1(f))
            f = f + f2  # Skip connection
            f = self.dec1(f)
            f = self.dec2(f)

            f = self.act(self.up2(f))
            f = f + f1  # Skip connection
            f = self.dec3(f)
            f = self.dec4(f)

            # Output
            out = self.output_conv(f)

            return out

    # ============================================================
    # 3. Synthetic Dataset for Distillation
    # ============================================================

    class SyntheticDistillationDataset(Dataset):
        """
        合成データセット
        実際のデータがない場合、ランダムな入力で Teacher の出力を学習
        """
        def __init__(self, num_samples=10000, resolution=512):
            self.num_samples = num_samples
            self.resolution = resolution

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Random UV features (32ch) + RGB (3ch) = 35ch
            uv_features = torch.randn(32, self.resolution, self.resolution) * 0.5
            rgb = torch.rand(3, self.resolution, self.resolution)

            x = torch.cat([uv_features, rgb], dim=0)  # 35ch

            # Random style vector
            style = torch.randn(512)

            return x, style

    # ============================================================
    # 4. Training Loop
    # ============================================================

    print("=" * 60)
    print("Knowledge Distillation: Light StyleUNet")
    print("=" * 60)

    # Create models
    print("\nCreating Teacher model (will be frozen)...")
    teacher = TeacherStyleUNet(in_ch=35, out_ch=96, style_dim=512).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher parameters: {teacher_params:,} ({teacher_params * 4 / 1024 / 1024:.2f} MB)")

    print("\nCreating Student model...")
    student = LightStyleUNet(in_ch=35, out_ch=96, base_ch=48, style_dim=512).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {student_params:,} ({student_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"  Compression ratio: {teacher_params / student_params:.1f}x")

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(ckpt['student_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resuming from epoch {start_epoch}")

    # Dataset and DataLoader
    print(f"\nCreating synthetic dataset...")
    dataset = SyntheticDistillationDataset(num_samples=10000, resolution=256)  # Lower res for speed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")

    # Optimizer and Loss
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    # TensorBoard
    writer = SummaryWriter(f"{VOLUME_PATH}/logs/light_styleunet")

    # Training
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}")

    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        student.train()
        epoch_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x, style) in enumerate(pbar):
            x = x.to(device)
            style = style.to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = teacher(x, style)

            # Student forward
            student_out = student(x, style)

            # Loss
            loss = criterion(student_out, teacher_out)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        scheduler.step()

        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
            save_path = f"{VOLUME_PATH}/checkpoints/light_styleunet_epoch{epoch+1}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save({
                'epoch': epoch + 1,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)

            print(f"  Saved checkpoint: {save_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = f"{VOLUME_PATH}/checkpoints/light_styleunet_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'student_state_dict': student.state_dict(),
                    'loss': avg_loss,
                }, best_path)
                print(f"  New best model saved!")

    writer.close()

    # Export to ONNX
    print(f"\n{'='*60}")
    print("Exporting best model to ONNX...")
    print(f"{'='*60}")

    best_ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/light_styleunet_best.pt", map_location=device)
    student.load_state_dict(best_ckpt['student_state_dict'])
    student.eval()

    # Dummy inputs
    dummy_x = torch.randn(1, 35, 512, 512, device=device)
    dummy_style = torch.randn(1, 512, device=device)

    onnx_path = f"{VOLUME_PATH}/light_styleunet.onnx"
    torch.onnx.export(
        student,
        (dummy_x, dummy_style),
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=['uv_features', 'extra_style'],
        output_names=['output'],
        dynamic_axes={
            'uv_features': {0: 'batch'},
            'extra_style': {0: 'batch'},
            'output': {0: 'batch'},
        },
    )

    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"Exported: {onnx_path}")
    print(f"  Size: {onnx_size:.2f} MB")
    print(f"  Compression: {118 / onnx_size:.1f}x smaller than original")

    # Commit volume
    volume.commit()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    return {
        'best_loss': best_loss,
        'onnx_path': onnx_path,
        'onnx_size_mb': onnx_size,
        'compression_ratio': 118 / onnx_size,
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    resume: str = None,
):
    """
    使用方法:

    # 新規訓練
    modal run modal_train_light_styleunet.py --epochs 100 --batch-size 4

    # チェックポイントから再開
    modal run modal_train_light_styleunet.py --resume /vol/checkpoints/light_styleunet_epoch50.pt
    """
    print("Starting Light StyleUNet distillation training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    result = train_light_styleunet.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        checkpoint_path=resume,
    )

    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"  Best loss: {result['best_loss']:.6f}")
    print(f"  ONNX size: {result['onnx_size_mb']:.2f} MB")
    print(f"  Compression: {result['compression_ratio']:.1f}x")
    print(f"\nONNX model saved to: {result['onnx_path']}")
    print("\nTo download the model:")
    print("  modal volume get guava-training-vol light_styleunet.onnx")
