import torch.nn as nn

def mlp(input_size=10, num_hidden=3, hidden_dim=1000,
        num_classes=2, activation_fn=nn.ReLU()):
    model = nn.Sequential()
    model.add_module("fc1", nn.Linear(input_size, 100))
    model.add_module("activation1", activation_fn)
    model.add_module("fc2", nn.Linear(100, hidden_dim))
    model.add_module("activation2", activation_fn)

    for i in range(3, num_hidden + 1):
        model.add_module(f"fc{i}", nn.Linear(hidden_dim, hidden_dim))
        model.add_module(f"activation{i}", activation_fn)

    model.add_module(f"fc{num_hidden + 1}", nn.Linear(hidden_dim, 20))
    model.add_module(f"activation{num_hidden + 1}", activation_fn)
    model.add_module(f"fc{num_hidden + 2}", nn.Linear(20, num_classes))

    return model

def synthetic_mlp_default():
    return mlp()

def synthetic_mlp_4_hidden_ch1():
    return mlp(num_hidden=4)

def synthetic_mlp_5_hidden_ch1():
    return mlp(num_hidden=5)