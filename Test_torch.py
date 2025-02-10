import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out

# Set the number of threads
torch.set_num_threads(1)

# Generate synthetic data
num_samples = 1000
input_size = 2
hidden_size = 16
output_size = 2

X = torch.randn(num_samples, input_size)
Y = torch.zeros(num_samples, output_size)
Y[:,0] = 2.0 + 1.2*X[:,0] - X[:,1]**2 + torch.exp(-3.0*X[:,0] - 2.0*X[:,1])
Y[:,0] = 1.4 + 3.0*X[:,0] - X[:,1]**2 + torch.exp(-2.0*X[:,0] - 3.0*X[:,1])

# Initialize the model, loss function, and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
batch_size = 50
n_train = int(num_samples * 0.7)
num_batches = int(n_train / batch_size)
n_val = int(num_samples * 0.2)
print_every = 50
loss_train = []
loss_val = []


# Normalize the data
X = (X - torch.mean(X[:n_train,:], axis=0))/torch.std(X[:n_train,:])
Y = (Y - torch.mean(Y[:n_train,:], axis=0))/torch.std(Y[:n_train,:])

for epoch in range(num_epochs):
    loss_epoch = 0.0
    for batch in range(num_batches): 
        outputs = model(X[batch*batch_size:(batch+1)*batch_size,:])
        loss = criterion(outputs, Y[batch*batch_size:(batch+1)*batch_size,:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss/num_batches

    # Loss training
    loss_train.append(loss_epoch)

    # Loss validation
    outputs = model(X[n_train:(n_train+n_val), :])
    loss_val.append(criterion(outputs, Y[n_train:(n_train+n_val), :]))

    if epoch % print_every == 0:
        print(f"Loss[%d] = (%e, %e)" %(epoch, loss_train[-1], loss_val[-1]))
