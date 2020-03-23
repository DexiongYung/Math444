import torch
import math
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', help='Number of neurons in hidden layer', nargs='?', default=256, type=int)

args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SZ = args.hidden_size
LR = 0.0005
PRINT = 1

class MLP(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = torch.nn.Linear(2, hidden_size)
        self.activation = torch.nn.Sigmoid()
        self.output = torch.nn.Linear(hidden_size, 1)
        self.to(DEVICE)

    def forward(self, input: torch.Tensor):
        hidden_out = self.hidden.forward(input)
        activation_out = self.activation(hidden_out)
        
        return self.output.forward(activation_out)

def function_to_approximate(x: float, x_1: float):
    """
    Function is x + (1 / 2) * x^2 + (1 / 100) * x^3
    """
    return [x + (1/2) * math.pow(x, 2) + (1/100) * math.pow(x_1, 3)]

def plot_losses(loss: list, x_label: str = "Iteration", y_label: str = "MSE", folder: str = "Result", filename: str = HIDDEN_SZ):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'r--', label="Loss")
    plt.title("Losses")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

def train(model: MLP, x: float, x_1: float, criterion):
    optimizer.zero_grad()

    input = torch.Tensor([x, x_1]).to(DEVICE)
    approximation = model.forward(input)
    real_output = function_to_approximate(x, x_1)
    loss = criterion(approximation, torch.Tensor([real_output]).to(DEVICE))

    loss.backward()
    optimizer.step()

def train_iteration(iterations: int, model: MLP, criterion):
    all_losses = []
    total_loss = 0
    for i in range(iterations):
        x = random.uniform(0, 100)
        x_1 = random.uniform(0, 100)

        train(model, x, x_1, criterion)

        if i % PRINT == 0:
            all_losses.append(total_loss / PRINT)
            total_loss = 0
            plot_losses(all_losses)
            torch.save({'weights': model.state_dict()}, f"Weight/{HIDDEN_SZ}")

model = MLP(HIDDEN_SZ)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_iteration(1, model, criterion)
