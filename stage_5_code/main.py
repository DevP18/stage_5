from .dataset_loader import Dataset_Loader
from .gcn import GCN
from .metrics import evaluate
import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATASET_CONFIG = {
    'cora':     {'nhid': 64, 'dropout': 0.5, 'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 200},
    'citeseer': {'nhid': 64, 'dropout': 0.5, 'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 150},
    'pubmed':   {'nhid': 64, 'dropout': 0.5, 'lr': 0.01, 'weight_decay': 5e-4, 'epochs': 200},
}


def run_dataset(dataset_name):

    train_losses = []

    loader = Dataset_Loader()
    loader.dataset_name = dataset_name
    loader.dataset_source_folder_path = f'../stage_5_data/{dataset_name}'

    data      = loader.load()
    graph     = data['graph']
    idx_train = data['train_test_val']['idx_train']
    idx_test  = data['train_test_val']['idx_test']

    x   = graph['X']
    y   = graph['y']
    adj = graph['utility']['A']
    cfg = DATASET_CONFIG[dataset_name]

    model = GCN(
        nfeat=x.shape[1],
        nhid=cfg['nhid'],
        nclass=len(y.unique()),
        dropout=cfg['dropout'],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )

    for epoch in range(cfg['epochs']):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(x, adj)[idx_train], y[idx_train])
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  [{dataset_name}] epoch {epoch:3d}  loss={loss.item():.4f}")
        # ---- PLOT LOSS CURVE ----
    plt.figure()
    plt.plot(train_losses)
    plt.title(f"{dataset_name} Training Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    model.eval()
    with torch.no_grad():
        pred = model(x, adj).max(1)[1]

    scores = evaluate(y[idx_test].numpy(), pred[idx_test].numpy())

    print()
    print("=" * 40)
    print(dataset_name.upper())
    print("=" * 40)
    print(f"accuracy: {scores[0]:.4f}")
    print(f"f1:       {scores[3]:.4f}")
    print()


run_dataset("cora")
run_dataset("citeseer")
run_dataset("pubmed")