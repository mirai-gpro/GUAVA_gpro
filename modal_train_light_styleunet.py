# modal_train_light_styleunet.py
# Knowledge Distillation: 35ch → 96ch 軽量 StyleUNet
#
# 実データを使用したKnowledge Distillation
# - 入力: 35ch (3ch RGB + 32ch DINOv2) @ UV空間
# - 出力: 96ch UV features
# - Extra Style: 512-dim global feature
#
# 使用方法:
#   # データ抽出後に学習
#   modal run modal_train_light_styleunet.py --epochs 100 --batch-size 2
#
#   # データ抽出 (別スクリプト)
#   modal run extract_uv_styleunet_data.py --action extract --num-frames 500

import modal
import os

# Modal app definition
app = modal.App("light-styleunet-distillation")

# Volumes
training_volume = modal.Volume.from_name("guava-training-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("uv-styleunet-distill-data", create_if_missing=True)

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
DATA_PATH = "/data"


@app.function(
    image=image,
    gpu="L4",
    timeout=6 * 3600,
    volumes={
        VOLUME_PATH: training_volume,
        DATA_PATH: data_volume,
    },
)
def train_light_styleunet(
    num_epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    checkpoint_path: str = None,
    use_synthetic: bool = False,  # True: 合成データ, False: 実データ
):
    """
    Knowledge Distillation で軽量 StyleUNet を訓練

    実データモード (推奨):
        extract_uv_styleunet_data.py で抽出したデータを使用

    合成データモード:
        ランダム入力でTeacher出力を学習
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # 1. Student Model: Lightweight RFDN-based architecture
    # ============================================================

    class ESA(nn.Module):
        """Enhanced Spatial Attention"""
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
            self.conv2 = nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1, groups=max(1, channels // reduction))
            self.conv3 = nn.Conv2d(channels // reduction, channels, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg = x.mean(dim=[2, 3], keepdim=True)
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

            distill = self.distill_conv(out3)
            remain = self.remain_conv(out3)

            out = torch.cat([distill, remain], dim=1)
            out = self.esa(out)

            return out + x

    class LightStyleUNet(nn.Module):
        """
        Lightweight StyleUNet (Student)
        35ch → 96ch with ~5-10MB parameters
        Uses RFDN blocks + Style modulation
        """
        def __init__(self, in_ch=35, out_ch=96, base_ch=48, style_dim=512):
            super().__init__()

            # Style projection
            self.style_proj = nn.Sequential(
                nn.Linear(style_dim, base_ch * 4),
                nn.LeakyReLU(0.2, True),
                nn.Linear(base_ch * 4, base_ch * 4),
            )

            # Input conv
            self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

            # Encoder
            self.enc1 = RFDB(base_ch)
            self.enc2 = RFDB(base_ch)
            self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

            self.enc3 = RFDB(base_ch * 2)
            self.enc4 = RFDB(base_ch * 2)
            self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

            # Bottleneck
            self.bottleneck1 = RFDB(base_ch * 4)
            self.bottleneck2 = RFDB(base_ch * 4)

            # Style modulation
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
            f1 = self.enc2(f)

            f = self.act(self.down1(f1))
            f = self.enc3(f)
            f2 = self.enc4(f)

            f = self.act(self.down2(f2))

            # Bottleneck with style modulation
            f = self.bottleneck1(f)

            B, C, H, W = f.shape
            style_scale = style_emb.view(B, C, 1, 1)
            f = f * (1 + self.style_mod(style_scale.expand(-1, -1, H, W)))

            f = self.bottleneck2(f)

            # Decoder
            f = self.act(self.up1(f))
            f = f + f2
            f = self.dec1(f)
            f = self.dec2(f)

            f = self.act(self.up2(f))
            f = f + f1
            f = self.dec3(f)
            f = self.dec4(f)

            # Output
            out = self.output_conv(f)

            return out

    # ============================================================
    # 2. Dataset: Real or Synthetic
    # ============================================================

    class RealDistillationDataset(Dataset):
        """
        抽出した実データを使用
        extract_uv_styleunet_data.py で生成されたデータを読み込む
        """
        def __init__(self, data_dir: str, resolution: int = 512):
            self.data_dir = Path(data_dir)
            self.resolution = resolution

            # Find all input files
            self.input_files = sorted((self.data_dir / "input_35ch").glob("*.pt"))
            self.output_files = sorted((self.data_dir / "output_96ch").glob("*.pt"))
            self.style_files = sorted((self.data_dir / "extra_style").glob("*.pt"))

            assert len(self.input_files) == len(self.output_files), \
                f"Input/Output mismatch: {len(self.input_files)} vs {len(self.output_files)}"

            print(f"Loaded {len(self.input_files)} real data pairs")

        def __len__(self):
            return len(self.input_files)

        def __getitem__(self, idx):
            # Load input (35ch)
            x = torch.load(self.input_files[idx])
            if x.dim() == 4:
                x = x.squeeze(0)

            # Load target output (96ch)
            y = torch.load(self.output_files[idx])
            if y.dim() == 4:
                y = y.squeeze(0)

            # Load style
            if idx < len(self.style_files):
                style = torch.load(self.style_files[idx])
                if style.dim() == 2:
                    style = style.squeeze(0)
            else:
                style = torch.randn(512)

            # Resize if needed
            if x.shape[-1] != self.resolution:
                x = nn.functional.interpolate(
                    x.unsqueeze(0), size=(self.resolution, self.resolution), mode='bilinear'
                ).squeeze(0)
                y = nn.functional.interpolate(
                    y.unsqueeze(0), size=(self.resolution, self.resolution), mode='bilinear'
                ).squeeze(0)

            return x, y, style

    class SyntheticDistillationDataset(Dataset):
        """
        合成データセット (フォールバック用)
        実データがない場合に使用
        """
        def __init__(self, num_samples=10000, resolution=256):
            self.num_samples = num_samples
            self.resolution = resolution

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Random UV features (32ch) + RGB (3ch) = 35ch
            uv_features = torch.randn(32, self.resolution, self.resolution) * 0.5
            rgb = torch.rand(3, self.resolution, self.resolution)
            x = torch.cat([rgb, uv_features], dim=0)  # 35ch (3+32)

            # Random style vector
            style = torch.randn(512)

            # Target will be generated by Teacher during training
            # Return dummy target
            y = torch.zeros(96, self.resolution, self.resolution)

            return x, y, style

    # ============================================================
    # 3. Teacher Model (for synthetic mode only)
    # ============================================================

    class TeacherStyleUNet(nn.Module):
        """Simplified Teacher model for synthetic data mode"""
        def __init__(self, in_ch=35, out_ch=96, style_dim=512):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
            )

            self.style_linear = nn.Linear(style_dim, 256)

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, out_ch, 3, padding=1),
            )

        def forward(self, x, style):
            B = x.shape[0]
            feat = self.encoder(x)
            style_feat = self.style_linear(style).view(B, -1, 1, 1)
            feat = feat * (1 + style_feat)
            out = self.decoder(feat)
            return out

    # ============================================================
    # 4. Training Setup
    # ============================================================

    print("=" * 60)
    print("Light StyleUNet Knowledge Distillation")
    print("=" * 60)

    # Check for real data
    real_data_dir = Path(DATA_PATH) / "uv_styleunet_dataset"
    has_real_data = (
        (real_data_dir / "input_35ch").exists() and
        len(list((real_data_dir / "input_35ch").glob("*.pt"))) > 0
    )

    if not use_synthetic and has_real_data:
        print("\n✅ Using REAL extracted data")
        dataset = RealDistillationDataset(real_data_dir, resolution=256)
        teacher = None  # Not needed for real data
    else:
        print("\n⚠️ Using SYNTHETIC data (real data not found or --use-synthetic specified)")
        print("  Run extract_uv_styleunet_data.py first for better results!")
        dataset = SyntheticDistillationDataset(num_samples=5000, resolution=256)
        teacher = TeacherStyleUNet(in_ch=35, out_ch=96, style_dim=512).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    # Create Student model
    print("\nCreating Student model...")
    student = LightStyleUNet(in_ch=35, out_ch=96, base_ch=48, style_dim=512).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {student_params:,} ({student_params * 4 / 1024 / 1024:.2f} MB)")

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(ckpt['student_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resuming from epoch {start_epoch}")

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"\nDataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Optimizer and Loss
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss function: MSE + L1
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # TensorBoard
    writer = SummaryWriter(f"{VOLUME_PATH}/logs/light_styleunet")

    # ============================================================
    # 5. Training Loop
    # ============================================================

    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}")

    best_loss = float('inf')
    os.makedirs(f"{VOLUME_PATH}/checkpoints", exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        student.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_l1 = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x, target, style) in enumerate(pbar):
            x = x.to(device)
            style = style.to(device)

            # Get target
            if teacher is not None:
                # Synthetic mode: generate target from Teacher
                with torch.no_grad():
                    target = teacher(x, style)
            else:
                target = target.to(device)

            # Student forward
            student_out = student(x, style)

            # Loss
            loss_mse = mse_loss(student_out, target)
            loss_l1 = l1_loss(student_out, target)
            loss = loss_mse + 0.1 * loss_l1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_l1 += loss_l1.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_mse.item():.4f}',
            })

        scheduler.step()

        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_l1 = epoch_l1 / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, MSE={avg_mse:.6f}, L1={avg_l1:.6f}, LR={current_lr:.6f}")

        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/mse', avg_mse, epoch)
        writer.add_scalar('Loss/l1', avg_l1, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
            save_path = f"{VOLUME_PATH}/checkpoints/light_styleunet_epoch{epoch+1}.pt"

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
                print(f"  ✅ New best model!")

    writer.close()

    # ============================================================
    # 6. Export to ONNX
    # ============================================================

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
            'uv_features': {0: 'batch', 2: 'height', 3: 'width'},
            'extra_style': {0: 'batch'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
    )

    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"Exported: {onnx_path}")
    print(f"  Size: {onnx_size:.2f} MB")
    print(f"  Compression: {118 / onnx_size:.1f}x smaller than original")

    # Commit volume
    training_volume.commit()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    return {
        'best_loss': best_loss,
        'onnx_path': onnx_path,
        'onnx_size_mb': onnx_size,
        'compression_ratio': 118 / onnx_size,
        'data_mode': 'synthetic' if teacher is not None else 'real',
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    resume: str = None,
    use_synthetic: bool = False,
):
    """
    使用方法:

    # 実データで学習 (推奨)
    modal run modal_train_light_styleunet.py --epochs 100 --batch-size 2

    # 合成データで学習 (データ抽出前のテスト用)
    modal run modal_train_light_styleunet.py --epochs 50 --use-synthetic

    # チェックポイントから再開
    modal run modal_train_light_styleunet.py --resume /vol/checkpoints/light_styleunet_epoch50.pt

    # 学習後のモデルダウンロード
    modal volume get guava-training-vol light_styleunet.onnx
    """
    print("Starting Light StyleUNet distillation training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Use synthetic: {use_synthetic}")

    result = train_light_styleunet.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        checkpoint_path=resume,
        use_synthetic=use_synthetic,
    )

    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"  Data mode: {result['data_mode']}")
    print(f"  Best loss: {result['best_loss']:.6f}")
    print(f"  ONNX size: {result['onnx_size_mb']:.2f} MB")
    print(f"  Compression: {result['compression_ratio']:.1f}x")
    print(f"\nONNX model saved to: {result['onnx_path']}")
    print("\nTo download the model:")
    print("  modal volume get guava-training-vol light_styleunet.onnx")
