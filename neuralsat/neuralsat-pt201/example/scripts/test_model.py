import torch.nn.functional as F
import onnxruntime as ort
import torch.nn as nn
import numpy as np
import torch
import time
import onnx
import os

    
class NetSigmoid(nn.Module):
    
    def __init__(self):
        super(NetSigmoid, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.layer(x)
    

    
class ReLUNet(nn.Module):
    
    def __init__(self):
        super(ReLUNet, self).__init__()
        
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = nn.Parameter(torch.tensor([[1, 2], [3, 4], [5, 6]]).float())
        self.linear1.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
        self.linear2 = nn.Linear(3, 2)
        self.linear2.weight.data = nn.Parameter(torch.tensor([1, 2, 3, -4, -5, -6]).view(2, 3).float())
        self.linear2.bias.data = nn.Parameter(torch.tensor([2, 3]).float())
        # print( self.linear1.weight.data.shape)
        # print( self.linear1.bias.data.shape)
        
        self.linear3 = nn.Linear(2, 3)
        self.linear3.weight.data = nn.Parameter(torch.tensor([[1, 2], [-3, -4], [-5, -6]]).float())
        self.linear3.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        
        # x = torch.log(x)
        # x = x.max(dim=1).values
        return x
    
    
def test_relu2():
    
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA import BoundedModule, BoundedTensor

    net = ReLUNet()
    x_U = torch.tensor([[1.0, 2.0]])
    x_L = torch.tensor([[-1.0, -2.0]])
    
    if 0:
        torch.onnx.export(
            net, 
            x_L, 
            "example/onnx/relu2.onnx", 
            verbose=False,
        )
    
    print(x_L.shape)
    device = 'cpu'
    
        
    abstractor = BoundedModule(
        model=net, 
        global_input=torch.zeros_like(x_L, device=device),
        bound_opts={
            'relu': 'adaptive', 
            'conv_mode': 'matrix', 
        },
        
        device=device,
        verbose=False,
    )
    new_x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)
    
    
    # print(lb, ub)
    C = torch.tensor([1, -1]).view(1, 1, 2).float()
    # print(x_L.shape, C.shape)
    # print(abstractor(x_U))
    if 0:
        method = 'forward'
        with torch.no_grad():
            lb, ub = abstractor.compute_bounds(x=(new_x,), method=method, C=None, bound_upper=True)
            print('[forward] lower', lb)
            print('[forward] upper', ub)
        print()
    else:
        method = 'backward'
        with torch.no_grad():
            lb, ub = abstractor.compute_bounds(x=(new_x,), method=method, C=None, bound_upper=True)
            print('[backward] lower', lb)
            print('[backward] upper', ub)
    
    
    
class NetReLU(nn.Module):
    
    def __init__(self):
        super(NetReLU, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 10),
        )

    def forward(self, x):
        return self.layer(x)

    
class NetConvranspose(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Linear(5, 512)
        self.bn1 = nn.BatchNorm2d(128)
        self.ct1 = nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.ct2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2))
        self.ct3 = nn.ConvTranspose2d(64, 4, kernel_size=(4, 4), stride=(2, 2))
        self.l2 = nn.Linear(3600, 1)

    def forward(self, x):
        x = self.l1(x)
        # print(1, x.shape)
        x = x.view(-1, 128, 2, 2)
        # print(2, x.shape)
        x = self.bn1(x)
        # print(3, x.shape)
        x = self.ct1(x)
        # print(4, x.shape)
        x = self.bn1(x)
        # print(5, x.shape)
        x = x.relu()
        x = self.ct2(x)
        # print(6, x.shape)
        x = x.relu()
        x = self.ct3(x)
        x = x.relu()
        x = x.flatten(1)
        x = self.l2(x)
        return x
    
    

class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=5, kernel_size=3)
        # Fully connected layer
        hidden_size = 180
        self.fc1 = nn.Linear(hidden_size, 32)  # assuming a classification problem with 10 classes
        self.fc2 = nn.Linear(32, 2)  # assuming a classification problem with 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def extract_instance(net_path, vnnlib_path):
    from util.spec.read_vnnlib import read_vnnlib
    from util.network.read_onnx import parse_onnx
    from verifier.objective import Objective, DnfObjectives
    
    vnnlibs = read_vnnlib(vnnlib_path)
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives

 
def test():
    net = NetReLU()
    x = torch.randn(1, 1, 28, 28)
    print(net(x).shape)
   
    net.eval()
    output_name = "example/onnx/net_relu_random.onnx"
    torch.onnx.export(
        net,
        x,
        output_name,
        verbose=False,
        opset_version=12,
    )
    
    print('Export onnx to:', output_name)
    
    net_path = output_name
    vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
    device = 'cpu'
    
    print('Running test with', net_path, vnnlib_path)
    START_TIME = time.time()
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)

    from verifier.verifier import Verifier 
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    status = verifier.verify(objectives)
    print(f'{status},{verifier.iteration}')
    return status, verifier.iteration
    
def test1():
    net = NetReLU()
    x = torch.randn(1, 2)
    print(net(x).shape)
   
    net.eval()
    output_name = "example/onnx/fnn_relu_4x2.onnx"
    torch.onnx.export(
        net,
        x,
        output_name,
        verbose=False,
        opset_version=12,
    )
    
    print('Export onnx to:', output_name)
    
    net_path = output_name
    vnnlib_path = 'example/vnnlib/motivation_example.vnnlib'
    device = 'cpu'
    
    print('Running test with', net_path, vnnlib_path)
    START_TIME = time.time()
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)

    from verifier.verifier import Verifier 
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    status = verifier.verify(objectives)
    print(f'{status},{verifier.iteration}')
    return status, verifier.iteration


def test2():
    from setting import Settings
    Settings.setup(None)
    print(Settings)
    
    net = NetConv()
    x = torch.randn(1, 3, 10, 10)
    print(net(x).shape)
   
    net.eval()
    output_name = "example/onnx/net_conv_small.onnx"
    torch.onnx.export(
        net,
        x,
        output_name,
        verbose=False,
        opset_version=12,
    )
    
    print('Export onnx to:', output_name)
    return
    
    net_path = output_name
    vnnlib_path = 'example/vnnlib/net_conv_small.vnnlib'
    device = 'cpu'
    
    print('Running test with', net_path, vnnlib_path)
    START_TIME = time.time()
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)

    from verifier.verifier import Verifier 
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    status = verifier.verify(objectives)
    print(f'{status},{verifier.iteration}')
    return status, verifier.iteration
    
    
def trail1():
    from util.misc.logger import logger
    import logging
    logger.setLevel(logging.INFO)
    
    trial = 0
    while True:
        status, iteration = test2()
        if status == 'unsat' and iteration > 0:
            break
        print(f'Trail {trial} failed\n\n')
        trial += 1
        time.sleep(1.0)
    

def inference_onnx(path: str, *inputs: np.ndarray) -> list[np.ndarray]:
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


def simplify_network():
    # load('/home/droars/Desktop/neuralsat/benchmark/cifar2020/nnet/cifar10_2_255_simplified.onnx')
    root_dir = '/home/droars/Desktop/tool/neuralsat/benchmark/mnistfc'
    with open(f'{root_dir}/instances.csv', 'w') as fp:
        for line in open(f'{root_dir}/instances_old.csv').read().strip().split('\n'):
            onnx_path, vnnlib_path, _ = line.split(',')
            # extract_instance(os.path.exists(f'{root_dir}/{onnx_path}'))
            model, input_shape, _ = extract_instance(f'{root_dir}/{onnx_path}', f'{root_dir}/{vnnlib_path}')
            if len([_ for _ in list(model.modules())[1:] if isinstance(_, torch.nn.ReLU)]) == 1:
                # print(model)
                continue
            
            # input_shape = [1, 1, 28 , 28]
            print(onnx_path, input_shape)
            x = torch.randn(input_shape)
            with torch.no_grad():
                output_pytorch = model(x)
            model.eval()
            
            output_name = f'{root_dir}/{onnx_path[:-5]}_simplified.onnx' #'example/cacmodel.onnx'
            os.system(f'rm -rf {output_name}')
            
            torch.onnx.export(
                model,
                x,
                output_name,
                verbose=False,
                opset_version=12,
            )
            
            assert os.path.exists(output_name)
            output_onnx = inference_onnx(output_name, x.view(input_shape).float().numpy())[0]
            assert np.allclose(output_pytorch, output_onnx, 1e-5, 1e-5)
            
            # break
            print(f'{onnx_path[:-5]}_simplified.onnx,{vnnlib_path},1000', file=fp)
            
 
if __name__ == '__main__':
    # trail1()
    test()