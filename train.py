import os
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import gen_3D_DOSY
from model import SimpleModel
from tqdm import tqdm

def load_data(file_path, device):
    data = np.load(file_path)
    S = torch.tensor(data['S'], dtype=torch.float32).to(device)
    labels = torch.tensor(data['labels'], dtype=torch.float32).to(device)
    return S, labels

def train_one_epoch(model, criterion, optimizer, data_loader, device, pbar):
    model.train()
    epoch_loss = 0.0
    for S_batch, labels_batch in data_loader:
        optimizer.zero_grad()
        outputs = model(S_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix(batch_loss=loss.item())
        
    return epoch_loss / len(data_loader)

def validate(model, criterion, val_data_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for S_batch, labels_batch in val_data_loader:
            outputs = model(S_batch)
            val_loss += criterion(outputs, labels_batch).item()
    return val_loss / len(val_data_loader)

def train(args, need_generate=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = gen_3D_DOSY(args)
    num_datasets = args.num_datasets
    dataset_dir = args.data_path
    result_dir = args.result_path
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    dataset_files = []

    if need_generate:
        for i in range(num_datasets):
            S, labels, _ = dataset.generate_batch()
            np.savez(os.path.join(dataset_dir, f'dataset_{i}.npz'), S=S, labels=labels)
            dataset_files.append(os.path.join(dataset_dir, f'dataset_{i}.npz'))
        S, labels, _ = dataset.generate_batch()
        np.savez(os.path.join(dataset_dir, 'val_dataset.npz'), S=S, labels=labels)
    else:
        dataset_files = [os.path.join(dataset_dir, f'dataset_{i}.npz') for i in range(num_datasets)]
        if not all(os.path.exists(file) for file in dataset_files):
            raise FileNotFoundError("Some dataset files do not exist. Please generate the datasets first.")

    model = SimpleModel(args.signal_dim, args.label_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')

    with open(os.path.join(result_dir, 'train_loss.txt'), 'w') as loss_file:
        num_epochs = args.num_epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # 创建一个进度条来跟踪所有数据集的训练过程
            with tqdm(total=num_datasets*(args.num_samples//args.batch_size), desc='Training', unit='dataset') as pbar:
                for dataset_file in dataset_files:
                    S, labels = load_data(dataset_file, device)
                    data_loader = DataLoader(TensorDataset(S, labels), batch_size=args.batch_size, shuffle=True)

                    avg_train_loss = train_one_epoch(model, criterion, optimizer, data_loader, device, pbar)
                    epoch_loss += avg_train_loss

                    del S, labels, data_loader
                    torch.cuda.empty_cache()  # 清理 GPU 缓存

            avg_epoch_loss = epoch_loss / num_datasets
            loss_file.write(f"{epoch + 1},{avg_epoch_loss:.4f}\n")

            val_S, val_labels = load_data(os.path.join(dataset_dir, 'val_dataset.npz'), device)
            val_loader = DataLoader(TensorDataset(val_S, val_labels), batch_size=args.batch_size)

            val_loss = validate(model, criterion, val_loader, device)

            del val_S, val_labels, val_loader
            torch.cuda.empty_cache()  # 清理 GPU 缓存
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.optimizer.param_groups[0]["lr"]}')

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))
                print(f'Saved best model with validation loss: {best_val_loss:.4f}')
