"""
modal_train.py - Modal上でKnowledge Distillation学習を実行

Modalは serverless GPU プラットフォーム
- 従量課金でGPUを利用可能
- A100/H100/T4等が利用可能
- セットアップが簡単

使い方:
1. pip install modal
2. modal setup  (認証)
3. modal run modal_train.py

データ:
- Modal Volumeに学習データをアップロード
- または、S3/GCSからダウンロード
"""

import modal
from pathlib import Path

# Modal app定義
app = modal.App("guava-distillation")

# イメージ定義 (依存パッケージ)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy",
    "tqdm",
    "onnx",
    "onnxruntime",
    "onnxscript",  # ONNX export用
)

# Volumeでデータを永続化
volume = modal.Volume.from_name("guava-distillation-data", create_if_missing=True)

# モデル定義をコピー
rfdn_student_code = '''
"""RFDN Student Model - copied inline for Modal"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESA(nn.Module):
    """Enhanced Spatial Attention"""
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RFDB(nn.Module):
    """Residual Feature Distillation Block"""
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = int(in_channels * distillation_rate)
        self.rc = self.remaining_channels = in_channels
        
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = nn.Conv2d(in_channels, self.rc, 3, padding=1)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = nn.Conv2d(self.remaining_channels, self.rc, 3, padding=1)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = nn.Conv2d(self.remaining_channels, self.rc, 3, padding=1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        
        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.act(self.c1_r(input))
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1))
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2))
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        return out_fused + input


class RFDN_Refiner(nn.Module):
    def __init__(self, in_channels=32, out_channels=3, n_feats=48, n_blocks=4):
        super(RFDN_Refiner, self).__init__()
        self.fea_conv = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.rfdb_blocks = nn.ModuleList([RFDB(n_feats) for _ in range(n_blocks)])
        self.lr_conv = nn.Conv2d(n_feats * n_blocks, n_feats, 1)
        self.out_conv = nn.Conv2d(n_feats, out_channels, 3, padding=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        fea = self.fea_conv(x)
        block_outputs = []
        out = fea
        for block in self.rfdb_blocks:
            out = block(out)
            block_outputs.append(out)
        out = torch.cat(block_outputs, dim=1)
        out = self.lr_conv(out)
        out = out + fea
        out = self.out_conv(out)
        out = self.final_act(out)
        return out
'''


@app.function(
    image=image,
    gpu="L4",  # L4は24GB VRAM
    timeout=7200,  # 2時間に延長
    volumes={"/data": volume},
)
def train_distillation(
    data_path: str = "/data/distill_dataset",
    output_path: str = "/data/output",
    num_epochs: int = 30,
    batch_size: int = 2,   # メモリ節約のため2に
    learning_rate: float = 1e-4,
):
    """
    Modal上で蒸留学習を実行
    
    データはModal Volumeに配置:
    /data/pairs/
        features/  - 32ch特徴マップ
        rgb/       - RGB出力
    """
    
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from pathlib import Path
    from tqdm import tqdm
    import os
    
    # RFDNモデルを動的にimport
    exec(rfdn_student_code, globals())
    
    # Dataset
    class DistillationDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            self.features_dir = self.data_dir / 'features'
            self.rgb_dir = self.data_dir / 'rgb'
            
            # 全ファイルリストを取得
            all_files = sorted(list(self.features_dir.glob('*.pt')))
            
            # 不正なサンプルをフィルタリング
            self.files = []
            skipped = 0
            for f in all_files:
                try:
                    feat = torch.load(f)
                    # 分散が小さすぎる場合はスキップ (背景のみ)
                    if feat.std() < 0.01:
                        skipped += 1
                        continue
                    # NaNチェック
                    if torch.isnan(feat).any():
                        skipped += 1
                        continue
                    self.files.append(f)
                except Exception as e:
                    skipped += 1
                    continue
            
            print(f"Dataset: {len(self.files)} valid samples (skipped {skipped} invalid)")
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            feat_path = self.files[idx]
            rgb_path = self.rgb_dir / feat_path.name
            features = torch.load(feat_path)
            rgb = torch.load(rgb_path)
            return features, rgb
    
    # VGG Perceptual Loss
    class VGGPerceptualLoss(nn.Module):
        def __init__(self):
            super().__init__()
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
            for block in self.blocks:
                for param in block.parameters():
                    param.requires_grad = False
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        def forward(self, pred, target):
            self.mean = self.mean.to(pred.device)
            self.std = self.std.to(pred.device)
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
            pred = torch.nn.functional.interpolate(pred, size=(224, 224), mode='bilinear')
            target = torch.nn.functional.interpolate(target, size=(224, 224), mode='bilinear')
            loss = 0.0
            for block in self.blocks:
                pred = block(pred)
                target = block(target)
                loss += torch.nn.functional.l1_loss(pred, target)
            return loss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Check data
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        print("Please upload training data to Modal Volume first")
        return None
    
    # Model
    model = RFDN_Refiner().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    # Data
    dataset = DistillationDataset(data_path)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Loss (L1のみ - メモリ節約)
    l1_loss = nn.L1Loss()
    # perceptual_loss = VGGPerceptualLoss().to(device)  # 無効化
    # perceptual_weight = 0.1
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Output dir
    os.makedirs(output_path, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for features, rgb_target in tqdm(train_loader, desc=f'Epoch {epoch}'):
            features = features.to(device)
            rgb_target = rgb_target.to(device)
            
            rgb_pred = model(features)
            
            loss = l1_loss(rgb_pred, rgb_target)
            # loss_perceptual = perceptual_loss(rgb_pred, rgb_target)
            # loss = loss_l1 + perceptual_weight * loss_perceptual
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, rgb_target in val_loader:
                features = features.to(device)
                rgb_target = rgb_target.to(device)
                rgb_pred = model(features)
                loss = l1_loss(rgb_pred, rgb_target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Save best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_path}/student_best.pt')
        
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}' + 
              (' (best)' if is_best else ''))
    
    # Final save
    torch.save(model.state_dict(), f'{output_path}/student_final.pt')
    
    # Export ONNX (外部データなし、1ファイルに統合)
    model.eval()
    model_cpu = model.cpu()
    dummy_input = torch.randn(1, 32, 512, 512)  # CPU
    onnx_path = f'{output_path}/rfdn_refiner.onnx'
    
    torch.onnx.export(
        model_cpu, 
        dummy_input, 
        onnx_path,
        export_params=True, 
        opset_version=14,  # 互換性のため14を使用
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f'ONNX exported: {onnx_path} ({onnx_size:.2f} MB)')
    
    # Commit volume changes
    volume.commit()
    
    return {
        'best_val_loss': best_val_loss,
        'onnx_path': onnx_path,
        'onnx_size_mb': onnx_size
    }


@app.function(
    image=image,
    volumes={"/data": volume},
)
def upload_mock_data(num_samples: int = 100):
    """
    テスト用のモックデータをVolumeにアップロード
    実際の学習には使えないが、パイプラインのテストに使用
    """
    
    import torch
    import os
    from tqdm import tqdm
    
    output_dir = "/data/pairs"
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/rgb", exist_ok=True)
    
    print(f"Generating {num_samples} mock pairs...")
    
    for i in tqdm(range(num_samples)):
        features = torch.randn(32, 512, 512) * 0.5
        rgb = torch.sigmoid(features[:3] * 2)
        
        torch.save(features, f"{output_dir}/features/{i:06d}.pt")
        torch.save(rgb, f"{output_dir}/rgb/{i:06d}.pt")
    
    volume.commit()
    print(f"Mock data saved to {output_dir}")
    return num_samples


@app.function(
    image=image,
    volumes={"/data": volume},
)
def download_results(local_dir: str = "./modal_output"):
    """
    学習結果をダウンロード
    """
    import shutil
    import os
    
    output_path = "/data/output"
    
    files = []
    for f in ['student_best.pt', 'student_final.pt', 'rfdn_refiner.onnx']:
        src = f"{output_path}/{f}"
        if os.path.exists(src):
            files.append(f)
    
    return files


@app.function(
    image=image,
    volumes={"/data": volume},
)
def get_file_bytes(filename: str):
    """
    ファイルのバイトデータを取得
    """
    import os
    
    filepath = f"/data/output/{filename}"
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return f.read()


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
)
def convert_to_onnx(
    model_path: str = "/data/output/student_best.pt",
    output_path: str = "/data/output/rfdn_refiner.onnx",
):
    """
    保存済みモデルをONNXに変換
    """
    import torch
    import os
    
    # RFDNモデルを動的にimport
    exec(rfdn_student_code, globals())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # モデルロード
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    model = RFDN_Refiner().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # ONNX変換 (外部データなし、1ファイルに統合)
    # dynamo=False で古いエクスポーターを使用（外部データ問題回避）
    dummy_input = torch.randn(1, 32, 512, 512)
    
    # CPUで変換
    model_cpu = model.cpu()
    model_cpu.eval()
    
    torch.onnx.export(
        model_cpu, 
        dummy_input, 
        output_path,
        export_params=True, 
        opset_version=14,  # 互換性のため14を使用
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # 古いエクスポーター使用
    )
    
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX exported: {output_path} ({onnx_size:.2f} MB)")
    
    # 検証
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated!")
    
    volume.commit()
    
    return {
        'onnx_path': output_path,
        'onnx_size_mb': onnx_size
    }


@app.local_entrypoint()
def main(
    action: str = "train",
    mock_samples: int = 100,
    epochs: int = 30,
    batch_size: int = 8,
):
    """
    Modal CLIからの実行エントリポイント
    
    使い方:
        modal run modal_train.py --action mock     # モックデータ生成
        modal run modal_train.py --action train    # 学習実行
        modal run modal_train.py --action convert  # ONNX変換
        modal run modal_train.py --action list     # 結果確認
        modal run modal_train.py --action download # ファイルダウンロード
    """
    import os
    
    if action == "mock":
        print("Generating mock data...")
        n = upload_mock_data.remote(mock_samples)
        print(f"Generated {n} mock samples")
        
    elif action == "train":
        print("Starting training on Modal...")
        result = train_distillation.remote(
            num_epochs=epochs,
            batch_size=batch_size
        )
        print(f"Training complete: {result}")
    
    elif action == "convert":
        print("Converting to ONNX...")
        result = convert_to_onnx.remote()
        print(f"Conversion complete: {result}")
        
    elif action == "list":
        print("Checking results...")
        files = download_results.remote()
        print(f"Available files: {files}")
    
    elif action == "download":
        print("Downloading files...")
        
        # 出力ディレクトリ作成
        output_dir = "./modal_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # ファイルリスト取得
        files = download_results.remote()
        print(f"Files to download: {files}")
        
        # 各ファイルをダウンロード
        for filename in files:
            print(f"  Downloading {filename}...")
            data = get_file_bytes.remote(filename)
            if data:
                local_path = os.path.join(output_dir, filename)
                with open(local_path, 'wb') as f:
                    f.write(data)
                size_kb = len(data) / 1024
                print(f"  ✅ Saved: {local_path} ({size_kb:.1f} KB)")
            else:
                print(f"  ❌ Failed: {filename}")
        
        print(f"\n📁 All files saved to: {output_dir}/")
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: mock, train, convert, list, download")
