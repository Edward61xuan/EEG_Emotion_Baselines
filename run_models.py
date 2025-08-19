"""
run_models.py
通用 EEG 模型训练入口
支持：
1. 命令行参数选择数据集、模型、划分模式
2. subject-internal CV / cross-subject CV
3. wandb + logging 记录
"""

import os
import sys
import argparse
import logging
import wandb
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
import torch

# =====================
# 1. 命令行参数
# =====================
def get_parser():
    parser = argparse.ArgumentParser(description="EEG Model Runner")

    parser.add_argument("--run_name", type=str, default="Test_Run", help="运行名称，用于wandb和日志文件命名")

    # 路径参数
    parser.add_argument("--data_root", type=str, default="./dataset", help="数据集父目录")
    parser.add_argument("--result_root", type=str, default="./results", help="结果保存路径")
    parser.add_argument("--log_root", type=str, default="./log", help="日志保存路径")

    # 数据集参数
    parser.add_argument("--dataset", type=str, default="dummy", help="数据集名称（如 SEED, DEAP）")
    parser.add_argument("--split_mode", type=str, choices=["subject_internal", "cross_subject"], required=True, help="划分模式")
    parser.add_argument("--subject_id", type=int, default=None, help="当split_mode=subject_internal时指定subject号")

    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="模型名称（如 Conformer, DAEST）")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser

# =====================
# 2. 日志初始化
# =====================
def init_logger(log_root, dataset, model_name):
    os.makedirs(log_root, exist_ok=True)
    log_file = os.path.join(log_root, f"{dataset}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# =====================
# 3. 数据集类
# =====================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================
# 4. 数据加载 & 划分
# =====================
def load_dataset(data_root, dataset_name):
    """
    Returns
    -------
    X : ndarray
        shape [N, channels, n_samples (or n_features)]
    y : ndarray
        shape [N]
    subject_ids : ndarray
        shape [N], subject id of each sample
    trial_ids : ndarray
        shape [N], trial id of each sample
    """
    if dataset_name == "dummy": 
        ds_path = os.path.join(data_root, 'dummy_eeg_data')
    X = np.load(os.path.join(ds_path, "X.npy"))
    y = np.load(os.path.join(ds_path, "y.npy"))
    subject_ids = np.load(os.path.join(ds_path, "subject_ids.npy"))
    trial_ids = np.load(os.path.join(ds_path, "trial_ids.npy"))
    return X, y, subject_ids, trial_ids

def split_dataset(X, y, subject_ids, mode, folds, subject_id=None):
    """
    划分数据集

    Args:
        X: ndarray, 输入特征, shape [N, channels, n_samples (or n_features)]
        y: ndarray, 标签, shape [N]
        subject_ids: ndarray, 每个样本对应的subject id, shape [N]
        mode: str, 'subject_internal' 或 'cross_subject'
        folds: int
            - 如果 mode == 'subject_internal'，表示标准 k-fold（k=folds）
            - 如果 mode == 'cross_subject'，表示每次留 folds 个 subject 做验证
        subject_id: int, 可选，仅 subject_internal 模式时指定
    
    Returns:
        splits: list of (train_idx, val_idx)
    """
    splits = []

    if mode == "subject_internal":
        assert subject_id is not None, "必须指定 subject_id"
        idx = np.where(subject_ids == subject_id)[0]   # 取该 subject 的所有样本
        # 按 k-fold 划分
        fold_indices = np.array_split(idx, folds)
        for i in range(folds):
            val_idx = fold_indices[i]
            train_idx = np.hstack([fold_indices[j] for j in range(folds) if j != i])
            splits.append((train_idx, val_idx))

    elif mode == "cross_subject":
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        if folds > n_subjects:
            raise ValueError(f"受试者数 {n_subjects} < folds={folds}，无法划分")

        # 每次留 folds 个 subject 做验证
        fold_subjects = [unique_subjects[i:i+folds] for i in range(0, n_subjects, folds)]
        for test_subjects in fold_subjects:
            train_subjects = [s for s in unique_subjects if s not in test_subjects]
            train_idx = np.isin(subject_ids, train_subjects)
            val_idx = np.isin(subject_ids, test_subjects)
            splits.append((np.where(train_idx)[0], np.where(val_idx)[0]))

    return splits



# =====================
# 5. 模型注册
# =====================
def build_model(model_name, input_shape, num_classes):
    if model_name.lower() == "conformer":
        from Models.Conformer import Conformer 
        return Conformer()
    elif model_name.lower() == "eegnet":
        from Models.EEGNet import EEGNet
        Chans, Samples = input_shape[0], input_shape[1]
        return EEGNet(nb_classes=num_classes, Chans=Chans, Samples=Samples)
    elif model_name.lower() == "dgcnn":
        from Models.DGCNN import DGCNNBlock   
        Chans, Features = input_shape[0], input_shape[1]
        return DGCNNBlock(in_features=Features, hidden_features=32,num_classes=num_classes, K=4,num_nodes=Chans)
        ### FIXME:hidden_size,k值待调
    elif model_name.lower() == "cbramod":
        raise NotImplementedError
        # from Models.CBraMod import CBraMod
        # return CBraMod(in_dim=input_shape[0], out_dim=input_shape[1], d_model=200, dim_feedforward=800, seq_len=input_shape[1], n_layer=12, nhead=8)

        # FIXME:CBraMode Needs to be pretrained, while this script is not for self-supervised training

    else:
        raise ValueError(f"未知模型: {model_name}")


# =====================
# 6. 训练 & 验证
# =====================
def train_one_fold(model, train_loader, val_loader, device, epochs, lr, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # X_batch = X_batch.unsqueeze(1)  # 确保输入形状正确
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # X_batch = X_batch.unsqueeze(1)  # 确保输入形状正确
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Val Acc: {acc:.4f}")
        # wandb.log({"loss": avg_loss, "val_acc": acc})


# =====================
# 7. 主流程
# =====================
def main():
    parser = get_parser()
    args = parser.parse_args()

    # 初始化日志
    logger = init_logger(args.log_root, args.dataset, args.model)
    logger.info(f"Args: {args}")

    # wandb 初始化
    # wandb.init(project="EEG_Models", config=vars(args))

    # 加载数据
    X, y, subject_ids, trial_ids = load_dataset(args.data_root, args.dataset)
    num_classes = len(np.unique(y))
    input_shape = X.shape[1:]  # [channels, samples]

    # 构造数据划分
    splits = split_dataset(X, y, subject_ids, args.split_mode, args.folds, args.subject_id)
    # 循环每个fold
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"=== Using Splits Mode :\"{args.split_mode}\"===Running fold {fold_idx+1}/{len(splits)} ===")
        if args.split_mode == "subject_internal":
            logger.info(f"Subject ID: {args.subject_id}",f"Train Samples: {len(train_idx)}, Val Samples: {len(test_idx)}")
        elif args.split_mode == "cross_subject":
            logger.info(f"Train Subjects: {np.unique(subject_ids[train_idx])}, Val Subjects: {np.unique(subject_ids[test_idx])}")
        train_dataset = EEGDataset(X[train_idx], y[train_idx])
        val_dataset = EEGDataset(X[test_idx], y[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 构造模型
        model = build_model(args.model, input_shape, num_classes)

        # 训练
        train_one_fold(model, train_loader, val_loader, args.device, args.epochs, args.lr, logger)


if __name__ == "__main__":
    main()
