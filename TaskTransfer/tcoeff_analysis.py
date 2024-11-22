import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate parameters
def generate_parameters(dx, dy, r, num_tasks):
    # Randomly generate covariance of x
    U = 5 * torch.eye(dx, device=device) + torch.randn(dx, dx, device=device)
    Sigma_x = 0.5 * (U + U.T)  # Make Sigma_x symmetric
    Sigma_x = (dx / torch.trace(Sigma_x)) * Sigma_x  # Normalize to get Tr(Sigma_x) = dx

    # Randomly generate representation Phi*
    A = torch.randn(r, dx, device=device)
    _, _, Phi = torch.linalg.svd(A, full_matrices=False)  # Generate Phi* as orthonormalized matrix

    # Randomly generate heads F0 (for validation), F1,...,FT (for training)
    F0 = torch.randn(dy, r, device=device)
    Fs = [F0]
    for _ in range(num_tasks):
        gamma = 0.01
        B = torch.randn(dy, dy, device=device)
        rot = torch.linalg.matrix_exp(0.5 * gamma * (B - B.T))
        F = rot @ F0
        Fs.append(F)

    return Fs, Phi, Sigma_x

# Generate data
def generate_data(n_points, Fs, Phi, cov_x, mode="train"):
    U, S, V = torch.linalg.svd(cov_x)
    cov_x_sqrt = U @ torch.diag(torch.sqrt(S)) @ V
    dy = Fs[0].shape[0]
    dx = Phi.shape[-1]

    if mode == "train":
        X, Y = [], []
        for i in range(1, len(Fs)):
            M = Fs[i] @ Phi
            xs = cov_x_sqrt @ torch.randn(dx, n_points, device=device)
            ws = 0.1 * torch.randn(dy, n_points, device=device)
            ys = M @ xs + ws
            X.append(xs)
            Y.append(ys)
        return X, Y
    elif mode == "test":
        M = Fs[0] @ Phi
        xs = cov_x_sqrt @ torch.randn(dx, n_points, device=device)
        ws = torch.randn(dy, n_points, device=device)
        ys = M @ xs + ws
        return xs, ys

# Neural network representation
class PhiNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PhiNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

# Least squares head
def least_squares_head(X, Y, Phi):
    with torch.no_grad():
        Phi_X = Phi(X)
    l = Y.T @ Phi_X
    r = torch.linalg.inv(Phi_X.T @ Phi_X + 1e-5 * torch.eye(Phi_X.shape[1], device=device))
    return l @ r

# Excess risk calculation
def excess_risk(X, Y, F, Phi, F_star, Phi_star):
    with torch.no_grad():
        pred = F @ Phi(X).T
        true = F_star @ (Phi_star @ X)
        l = torch.mean((Y - pred.T) ** 2)
        r = torch.mean((Y - true.T) ** 2)
    return l - r

# Experiment
def experiment(dx, dy, r, T, n, lr, num_steps):
    F_star, Phi_star, Sigma_x = generate_parameters(dx, dy, r, T + 1)
    X_train, Y_train = generate_data(n, F_star, Phi_star, Sigma_x, mode="train")
    X_test, Y_test = generate_data(n, F_star, Phi_star, Sigma_x, mode="test")

    Phi = PhiNet(input_dim=dx, output_dim=r).to(device)
    optimizer = optim.SGD(Phi.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for step in range(num_steps):
        total_loss = 0.0
        for t in range(1, T + 1):
            x = X_train[t].T
            y = Y_train[t].T

            optimizer.zero_grad()
            embeddings = Phi(x)
            F_t = least_squares_head(x, y, Phi)
            preds = embeddings @ F_t.T
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {total_loss / T}")

    # Held-out task
    x_test = X_test.T
    y_test = Y_test.T
    F_heldout = least_squares_head(x_test, y_test, Phi)

    train_excess_risks = []
    for t in range(1, T + 1):
        x = X_train[t].T
        y = Y_train[t].T
        F_t = least_squares_head(x, y, Phi)
        train_excess_risks.append(excess_risk(x, y, F_t, Phi, F_star[t], Phi_star))

    avg_train_excess_risk = torch.tensor(train_excess_risks).mean().item()
    test_excess_risk = excess_risk(x_test, y_test, F_heldout, Phi, F_star[0], Phi_star).item()
    transfer_coefficient = avg_train_excess_risk / test_excess_risk

    return transfer_coefficient

# Main script
if __name__ == "__main__":
    dx, dy, r = 50, 15, 5
    num_tasks = 10
    dataset_sizes = [50, 100, 500, 1000, 5000]
    lr = 0.01
    num_steps = 1000

    all_transfer_coefficients = []

    for n in tqdm(dataset_sizes, desc="Running Experiments"):
        transfer_coefficients = []
        for s in torch.linspace(0.1, 1.0, 10):
            Fs, Phi_star, Sigma_x = generate_parameters(dx, dy, r, num_tasks)
            for i in range(1, num_tasks):
                Fs[i] *= diversity_factor
            transfer_coefficient = experiment(dx, dy, r, num_tasks, n, lr, num_steps)
            transfer_coefficients.append(transfer_coefficient)
        all_transfer_coefficients.append(transfer_coefficients)

    plt.figure(figsize=(10, 6))
    plt.hist(all_transfer_coefficients, bins=20, label=[f"n={n}" for n in dataset_sizes], alpha=0.7)
    plt.xlabel("Transfer Coefficient (ν)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Transfer Coefficients for Different Dataset Sizes")
    plt.legend()
    plt.grid()
    plt.show()
