import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def synthetic_cnn_1_conv_ch1(input_size=5, activation_fn=nn.ReLU()):
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Flatten(),
        nn.Linear(input_size * input_size * 10, 1000),
        activation_fn,
        nn.Linear(1000, 100),
        activation_fn,
        nn.Linear(100, 20),
        activation_fn,
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_1_conv_ch3(input_size=5, activation_fn=nn.ReLU()):
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Flatten(),
        nn.Linear(input_size * input_size * 10, 1000),
        activation_fn,
        nn.Linear(1000, 100),
        activation_fn,
        nn.Linear(100, 20),
        activation_fn,
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_default():
    return synthetic_cnn_1_conv_ch1()

def synthetic_cnn_2_conv_ch1(input_size=5, activation_fn=nn.ReLU()):
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Flatten(),
        nn.Linear(input_size * input_size * 10, 1000),
        activation_fn,
        nn.Linear(1000, 100),
        activation_fn,
        nn.Linear(100, 20),
        activation_fn,
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_2_conv_ch3(input_size=5, activation_fn=nn.ReLU()):
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Flatten(),
        nn.Linear(input_size * input_size * 10, 1000),
        activation_fn,
        nn.Linear(1000, 100),
        activation_fn,
        nn.Linear(100, 20),
        activation_fn,
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_3_conv_ch1(input_size=5, activation_fn=nn.ReLU()):
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(input_size * input_size * 20, 1000),
        activation_fn,
        nn.Linear(1000, 100),
        activation_fn,
        nn.Linear(100, 20),
        activation_fn,
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_3_conv_ch3(input_size=5, activation_fn=nn.ReLU()):
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1),
        activation_fn,
        nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(input_size * input_size * 20, 1000),
        activation_fn,
        nn.Linear(1000, 100),
        activation_fn,
        nn.Linear(100, 20),
        activation_fn,
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_avgpool_ch1(input_size=5):
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(input_size * input_size * 10, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_avgpool_ch3(input_size=5):
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(input_size * input_size * 10, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    return model

def synthetic_cnn_sigmoid_ch1(input_size=5):
    return synthetic_cnn_1_conv_ch1(input_size=input_size, activation_fn=nn.Sigmoid())

def synthetic_cnn_tanh_ch1(input_size=5):
    return synthetic_cnn_1_conv_ch1(input_size=input_size, activation_fn=nn.Tanh())

def synthetic_cnn_sigmoid_ch3(input_size=5):
    return synthetic_cnn_1_conv_ch3(input_size=input_size, activation_fn=nn.Sigmoid())

def synthetic_cnn_tanh_ch3(input_size=5):
    return synthetic_cnn_1_conv_ch3(input_size=input_size, activation_fn=nn.Tanh())
