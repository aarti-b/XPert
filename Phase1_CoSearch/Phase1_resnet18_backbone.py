import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
import compute_score as score_compute
import os
import time
import numpy as np
from torchvision.models import resnet18
import torchvision.transforms as transforms

from scipy.special import softmax
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=12, type=float, help='learning rate')
parser.add_argument('--hw_params', default='16,4,274,64', help='n_pe_tile, n_xbar_pe, total_tiles, xbar_size')
parser.add_argument('--t_latency', default=5000., type=float, help='target latency')
parser.add_argument('--t_area', default=50e6, type=float, help='target area')
parser.add_argument('--factor', default=0.75, type=float, help='factor for PEs')
parser.add_argument('--epochs', default=2000, type=int, help='Epochs')
parser.add_argument('--wt_prec', default=8, type=int, help='Weight Precision')
parser.add_argument('--cellbit', default=4, type=int, help='Number of Bits/Cell')
parser.add_argument('--area_tolerance', default=5, type=int, help='Area Tolerance (%)')

args = parser.parse_args()
class masking(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a_param):

        m = nn.Softmax()
        p_vals = m(a_param)

        gates = torch.FloatTensor([1]).cuda() * (1. *(p_vals == torch.max(p_vals))).float()

        return gates

    @staticmethod
    def backward(ctx, grad_output):

        grad_return = grad_output
        return grad_return, None

n_pe_tile, n_xbar_pe, total_tiles, xbar_size = map(float, args.hw_params.split(',')) #torch.tensor([8]).cuda()

class compute_latency(nn.Module):
    def forward(self, layer_latency_nodes, layer_arch_params):
        softmax = nn.Softmax()
        prob = softmax(layer_arch_params)
        lat = torch.sum(layer_latency_nodes * prob)
        return lat


# Define ResNet18 structure
resnet_stages = [
    {"in_channels": 3, "out_channels": 64, "blocks": 2, "stride": 1},   
    {"in_channels": 64, "out_channels": 128, "blocks": 2, "stride": 2},  
    {"in_channels": 128, "out_channels": 256, "blocks": 2, "stride": 2},  
    {"in_channels": 256, "out_channels": 512, "blocks": 2, "stride": 2},  
]


# Architectural search spaces
k_space = [3, 5, 7]
adc_share_space = [2, 4, 8, 16, 32, 64]
p_space = [1, 2, 4, 8, 16, 32, 64]  
adc_type_space = [1, 2]  

# Initialize network architecture and latency nodes
network_latency_nodes = []
network_arch = []

# Iterate over ResNet18 stages and blocks
for stage in resnet_stages:
    in_channels = stage["in_channels"]
    out_channels = stage["out_channels"]
    for block in range(stage["blocks"]):
        layer_arch = []
        layer_latency_nodes = []
        stride = stage["stride"] if block == 0 else 1  

        for adc in adc_type_space:
            for kernel_size in k_space:
                for p in p_space:
                    for adc_share in adc_share_space:
                        # Append configurations for each layer/block
                        layer_arch.append((adc, out_channels, kernel_size, stride, p, adc_share))

                        # Compute latency (adjust formula as needed)
                        latency = (n_pe_tile * kernel_size**2 * adc_share) / float(p)
                        layer_latency_nodes.append(latency)

        # Append to global architecture and latency nodes
        network_arch.append(layer_arch)
        network_latency_nodes.append(layer_latency_nodes)



def LUT_area(tiles, mux, adc_type):
    if adc_type == 1:
        a,b,c,d,e = 2557.166495434807, 1325750339.0150185, 0.0037802706563706337, -1325744426.1570153, 355177.96577219985
    if adc_type == 2:
        a, b, c, d, e = 1662.1420546233026, 1647434588.0067654, 0.0011610296696243961, -1647433897.2349775, 328615.0508042259

    area = (a * mux + e) * tiles + b * torch.exp(c * tiles / mux) + d

    return area



def LUT_latency(tiles, mux, adc_type, speedup, feature_size):
    # a, b, c, d, e = -0.0013059562090877996, -176121.8732468568, 0.020443304045075883, 88062.15068343304, 88062.150683433
    # print(f'tiles {tiles}, mux {mux}')


    if adc_type == 1:
        a, b, c, d, e = 0.08836970306542265, 8.466777271635184e-05, 1.0, 0.10469741426869324, 0.12532326984887984

        # latency = latency
    if adc_type == 2:
        a, b, c, d, e = 0.24051026469311507, 0.000432503086158724, 1.0, 0.4050948974138533, 0.16033299026875378


    latency = a * tiles * e + b * mux * mux + d
    # latency = (a * mux * mux * tiles + b) + (c * tiles * tiles * mux + d) + e
    latency = latency * feature_size ** 2 / speedup
    # a * x * e + b * y ** 2 + d
    return latency


network_arch = np.array(network_arch)
network_arch = torch.from_numpy(network_arch)
network_arch = network_arch.cuda()


total_layers = len(network_arch)
print("*****************************************",total_layers)

arch_params = [
    torch.nn.Parameter(torch.randn(len(network_arch[i])).cuda(), requires_grad=True)
    for i in range(len(network_arch))
]

optimizer = optim.SGD(arch_params, lr=args.lr, momentum=0.99, weight_decay=0.00001)


loss_fn = torch.nn.MSELoss()
s = nn.Softmax()

target_latency = torch.FloatTensor([args.t_latency])
target_latency = target_latency.cuda()
target_latency = Variable(target_latency, requires_grad = True)

target_util = torch.FloatTensor([1])
target_util = target_util.cuda()
target_util = Variable(target_util, requires_grad = True)
comp_lat = compute_latency()

latency_list = []
arch_param_epoch = []
lat_err_list, r_loss_list, err_list, tile_list = [], [], [], []
best_pe = 0
best_diff = 50e6
best_area = 500e6
best_latency = 5000000.
best_score = 0
pool_layers = [1, 3, 6, 9]



class custom_model(nn.Module):
    def __init__(self, features, num_classes, input_size=(3, 32, 32)):
        super(custom_model, self).__init__()
        self.features = features
        # Dynamically determine the flattened feature vector size
        dummy_input = torch.zeros(1, *input_size)  
        flattened_size = self._get_flattened_size(dummy_input)
        self.classifier = nn.Linear(flattened_size, num_classes)

    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


best_network = nn.Sequential(*([torch.nn.BatchNorm2d(3)]))
best_arch = copy.deepcopy(arch_params)
best_adc_arch = None

prev_list = [3]
for epochs in range(args.epochs):
    print(epochs)
    total_latency = 0
    total_pe = 0
    if epochs % 100 == 0:
        # print(f'arch param {arch_params[0][0]}')
        a_param = copy.deepcopy(arch_params)
        arch_param_epoch.append(a_param)
        
    parallel = []
    mux = []
    adc_arch = []
    if epochs % 1 == 0:
        in_ch = 3
        total_pe = 0
        tile_area = 0
        tile_latency = 0
        t_speedup = 0
        act_tile_area = 0
        act_tile_latency = 0
        expected_util = 0
        abs_tile = 0
        sup, t_use, util_list = [], [], []
        # print(f'arch parame {arch_params}')
        layers = []
        o_list = []
        for i in range(total_layers):
            prob = s(arch_params[i])
            index = torch.argmax(prob)
            a = network_arch[i][index]

            gate = masking.apply(prob)
            # print(f'================ layer  {i} ================')
            adc_type = a[0]
            o_ch = a[1]
            k = a[2]
            f = a[3]
            m = a[5]

            xbar_s = torch.FloatTensor([xbar_size]).cuda()
            total_area = torch.FloatTensor([args.t_area]).cuda()

            layers += [torch.nn.Conv2d(in_ch, int(o_ch.item()), kernel_size=int(k.item()), padding=1),
                       torch.nn.BatchNorm2d(int(o_ch.item())),
                       torch.nn.ReLU(inplace=True)]
            o_list.append(int(o_ch.item()))
            if i in pool_layers:
                layers += [torch.nn.MaxPool2d(2,2)]

            if i < total_layers-1:

                n_cells = int(args.wt_prec / args.cellbit)

                n_xbars = torch.ceil(in_ch*k*k / xbar_s) * torch.ceil(o_ch*n_cells / xbar_s)
                # n_adcs = xbar_s/m * n_xbars
                n_tiles = torch.ceil(n_xbars/(n_pe_tile*n_xbar_pe))
                if (o_ch * n_cells / xbar_s) >= 1:
                    speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars)
                else:
                    speedup = (n_tiles * n_xbar_pe * n_pe_tile / n_xbars) * (xbar_s / (o_ch * n_cells))


                act_tile_area += LUT_area(n_tiles, m, adc_type)
                act_tile_latency += LUT_latency(n_tiles, m, adc_type, speedup, f)

                tile_area += gate.sum() * LUT_area(n_tiles, m, adc_type) #(668468 * (0.9981761863406945 * n_tiles - 0.041827731986675554))
                tile_latency += gate.sum() * LUT_latency(n_tiles, m, adc_type, speedup, f)


                t_use.append(n_tiles.item())


            if i == total_layers-1:
                
                feature_size = int(32 / (2**len(pool_layers)))
                n_cells = int(args.wt_prec / args.cellbit)
                fc_xbars = torch.ceil(o_ch / xbar_s) * torch.ceil(10*n_cells / xbar_s)
                n_tiles = torch.ceil(fc_xbars / 64.)

                act_tile_area += LUT_area(n_tiles, m, adc_type)
                tile_area += gate.sum() * LUT_area(n_tiles, m, adc_type)

                layers = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)]

                # Add more layers for ResNet
                features = nn.Sequential(*layers)

                # Pass the number of classes (e.g., 10 for CIFAR-10)
                num_classes = 10
                network = custom_model(features, num_classes)


            parallel.append(speedup.item())
            adc_arch.append(adc_type.item())
            mux.append(m.item())
            in_ch = int(o_ch.item())
        print(f' %%%%%% act tile speedup {act_tile_latency.item(), tile_latency.item()} best_latency {best_latency} act_area {act_tile_area.item(), tile_area.item()} adc_arch {np.mean(np.array(adc_arch))} mux {np.mean(np.array(mux))} tiles {np.sum(np.array(t_use))}')
        factor = args.factor
        target_area = torch.FloatTensor([total_area]).cuda()

        error = 0.01 * tile_latency + 0.1 * 1e-6 * (loss_fn(tile_area, total_area))
        
        err_list.append(error.item())
        # pe_in_hardware = float(total_tiles*n_pe_tile)
        if np.abs(act_tile_area.item()-args.t_area) < (args.area_tolerance * args.t_area / 100) and o_list != prev_list and act_tile_latency.item() < best_latency:

 


            prev_list = o_list
            score = score_compute.compute_network_score(network)
            print('**************************',score)
            # if score > best_score and layers != best_network: # and score > 1600 and score <= 1650:
            if act_tile_latency.item() < best_latency:
                print("Updating best_latency")
                best_diff = act_tile_latency.item()
                best_area = act_tile_area.detach().cpu().numpy()
                best_latency = act_tile_latency.detach().cpu().numpy()
                best_mux = mux
                best_network = layers
                best_par = parallel
                best_tile_use = t_use
                best_score = score
                best_adc_arch = adc_arch
                best_adc_mean = np.mean(np.array(best_adc_arch))
                best_mux_mean = np.mean(np.array(best_mux))
                best_tile_sum = np.sum(np.array(best_tile_use))
                best_arch = copy.deepcopy(arch_params)


    flag=4
    optimizer.zero_grad()
    error.backward(retain_graph=True)
    optimizer.step()
arch = []
for i in range(total_layers):
    prob = s(best_arch[i])
    index = torch.argmax(prob)
    a = network_arch[i][index]  
    print(a)
    arch.append(a)

arch2 = []
arch3 = []
csv = []
in_ch = 3
l_c = 0
for i in arch:
    if i[2] == 3:
        pad = 1
    if i[2] == 5:
        pad = 2
    if i[2] == 7:
        pad = 3
    if int(i[3].item()) == 4 or int(i[3].item()) == 2:
        f_size = 6
    else:
        f_size = int(i[3].item())

    csv.append((f_size+2, f_size+2, int(in_ch),  int(i[2].item()),
                int(i[2].item()), int(i[1].item()), 0, 1, int(i[5]), int(i[0]) ))

    if int(in_ch) == 64 and int(i[1].item()) == 512:
        csv.append((int(i[3].item())+2, int(i[3].item())+2,512,3,3,64,0,1,int(i[5]), int(i[0])))
    else:
        csv.append((int(i[3].item())+2, int(i[3].item())+2,64,3,3,512,0,1,int(i[5]), int(i[0])))

    arch3.append(('C', int(i[1].item()), int(i[2].item()), 'same', int(8)))
    if l_c == 1 or l_c == 3 or l_c == 6 or l_c == 9:
        arch3.append(('M',2,2))


    arch2.append(('C', int(in_ch), int(i[1].item()), int(i[2].item()), 'same', int(8)))
    if int(in_ch) == 64 and int(i[1].item()) == 512:
        arch2.append(('C', 512, 64, 3, 'same', int(8)))
    else:
        arch2.append(('C',64, 512, 3, 'same', int(8)))
    if l_c == 1 or l_c == 3 or l_c == 6 or l_c == 9:
        arch2.append(('M',2,2))
    l_c += 1
    in_ch = int(i[1].item())

print(f'latency {best_latency},\nbest_adc_arch {best_adc_arch}, \nbest_area {best_area}, \ncolumn_sharing {best_mux} \nbest_score {best_score}')
# torch.save(arch_param_epoch, './cifar10/arch_param_epoch.pt')
# torch.save(lat_err_list, './lat_err_list.pt')
# torch.save(err_list, './err_list.pt')
# torch.save(r_loss_list, './r_loss_list.pt')
# torch.save(tile_list, './tile_list.pt')
for i in range(len(csv)):
    print(str(csv[i])[1:-1])

# print(csv)
print(arch3)
print(arch2)
print('\n')
