import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ==========================================
# 1. DATASET & CHARGEMENT DE L'ARBORESCENCE
# ==========================================

class CSIDataset(Dataset):
    def __init__(self, root_dir, window_size=200, overlap=100):
        """
        Arborescence attendue :
        root_dir/
          ├── assis/
          │     ├── p1/
          │     │     ├── prise_001/ (contient RX1.csv ... RX4.csv)
          │     │     └── prise_002/
          └── chute/
                └── ...
        """
        self.window_size = window_size
        self.overlap = overlap
        self.samples = [] # Stocke les chemins et métadonnées
        self.labels_map = {}
        
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.labels_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for cls_name, label in self.labels_map.items():
            cls_path = os.path.join(root_dir, cls_name)
            for person in os.listdir(cls_path):
                person_path = os.path.join(cls_path, person)
                if not os.path.isdir(person_path):
                    continue
                for prise in os.listdir(person_path):
                    prise_path = os.path.join(person_path, prise)
                    if not os.path.isdir(prise_path):
                        continue
                    
                    # Vérifier la présence des 4 récepteurs
                    rx_files = [os.path.join(prise_path, f"RX{i}.csv") for i in range(1, 5)]
                    if all(os.path.exists(f) for f in rx_files):
                        self.samples.append({
                            'files': rx_files,
                            'label': label
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger et fusionner les 4 récepteurs (RX1 à RX4)
        # Chaque CSV contient des sous-porteuses sur les colonnes et le temps sur les lignes
        dfs = [pd.read_csv(f).values for f in sample['files']]
        
        # S'assurer de la même longueur temporelle
        min_len = min([df.shape[0] for df in dfs])
        dfs = [df[:min_len, :] for df in dfs]
        
        # Concaténation spatiale/canaux : Shape (Temps, Canaux_totaux)
        data = np.concatenate(dfs, axis=1)
        
        # Normalisation Z-score par colonne
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
        # Découpage ou padding pour atteindre window_size
        if data.shape[0] >= self.window_size:
            data = data[:self.window_size, :]
        else:
            padding = np.zeros((self.window_size - data.shape[0], data.shape[1]))
            data = np.concatenate([data, padding], axis=0)
            
        # PyTorch attend le format (Canaux, Temps)
        tensor_data = torch.tensor(data.T, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return tensor_data, label


# ==========================================
# 2. ARCHITECTURE INCEPTIONTIME (PyTorch)
# ==========================================

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32, kernel_sizes=[9, 19, 39]):
        super().__init__()
        
        # Bottleneck pour réduire la dimensionnalité si nécessaire
        if in_channels > 1 and bottleneck_channels > 0:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, out_channels, kernel_size=ks, padding=ks//2, bias=False)
            for ks in kernel_sizes
        ])
        
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x)
        out = [conv(z) for conv in self.convs]
        out.append(self.maxpool(x))
        out = torch.cat(out, dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_classes, nb_filters=32, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        current_channels = in_channels
        for d in range(depth):
            self.blocks.append(InceptionBlock1D(current_channels, nb_filters))
            current_channels = nb_filters * 4  # 3 convs + 1 maxpool
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# ==========================================
# 3. SCRIPT D'ENTRAÎNEMENT PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # Paramètres
    DATA_DIR = "./DATASET"  # Remplacez par le chemin de votre dossier
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WINDOW_SIZE = 200
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Chargement dataset & dataloader
    dataset = CSIDataset(root_dir=DATA_DIR, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(dataset.labels_map)
    print(f"Classes détectées : {dataset.labels_map}")
    print(f"Nombre total de prises valides : {len(dataset)}")

    if len(dataset) > 0:
        # Récupération du nombre de canaux à partir d'un échantillon
        sample_x, _ = dataset[0]
        in_channels = sample_x.shape[0]

        # Initialisation du modèle
        model = InceptionTime(in_channels=in_channels, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Boucle d'entraînement minimale
        model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Époque [{epoch+1}/{EPOCHS}] — Perte : {running_loss/len(dataloader):.4f}")
    else:
        print("Aucune donnée valide trouvée. Vérifiez le chemin du dossier DATASET.")