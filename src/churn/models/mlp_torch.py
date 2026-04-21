
"""
MLP PyTorch para churn (notebook 04): logits, sigmoid na avaliação, BCEWithLogitsLoss no treino.
"""

from __future__ import annotationsimport numpy as npimport torchimport torch.nn as nnimport torch.optimfrom torch.utils.data import DataLoaderclass MLPChurn(nn.Module):
    """
    Rede feedforward simples: cada camada mistura linearmente os neurônios e aplica ReLU
    (ativação não linear). A última camada devolve um único número bruto (logit), não probabilidade.
    """

    def __init__(self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fluxo dados → primeira camada → não linearidade → … → logit final."""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Uma passagem completa pelo DataLoader de treino: para cada mini-lote calcula o erro,
    propaga o gradiente e atualiza os pesos (Adam). Devolve loss médio ponderado pelo tamanho.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_n = x_batch.size(0)
        total_loss += loss.item() * batch_n
        n_samples += batch_n

    return total_loss / max(n_samples, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Modo avaliação: não atualiza pesos; calcula loss médio e junta verdade e probabilidade
    prevista (sigmoid do logit) para métricas fora do PyTorch.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    y_true_list: list[np.ndarray] = []
    y_proba_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_n = x_batch.size(0)
            total_loss += loss.item() * batch_n
            n_samples += batch_n

            # Probabilidade para métricas sklearn (ROC, F1 com limiar, etc.).
            proba = torch.sigmoid(logits).cpu().numpy()
            y_true_list.append(y_batch.cpu().numpy())
            y_proba_list.append(proba)

    avg_loss = total_loss / max(n_samples, 1)
    y_true = np.vstack(y_true_list).flatten()
    y_proba = np.vstack(y_proba_list).flatten()
    return avg_loss, y_true, y_proba
