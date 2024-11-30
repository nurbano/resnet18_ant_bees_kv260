# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Net::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/Conv2d[conv1]/20
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #Net::Net/21
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Net::Net/MaxPool2d[pool]/input.2
        self.module_4 = py_nndct.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/Conv2d[conv2]/37
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #Net::Net/38
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Net::Net/MaxPool2d[pool]/44
        self.module_7 = py_nndct.nn.Module('flatten') #Net::Net/input.3
        self.module_8 = py_nndct.nn.Linear(in_features=400, out_features=120, bias=True) #Net::Net/Linear[fc1]/51
        self.module_9 = py_nndct.nn.ReLU(inplace=False) #Net::Net/input.4
        self.module_10 = py_nndct.nn.Linear(in_features=120, out_features=84, bias=True) #Net::Net/Linear[fc2]/56
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #Net::Net/input
        self.module_12 = py_nndct.nn.Linear(in_features=84, out_features=10, bias=True) #Net::Net/Linear[fc3]/61

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(end_dim=3, start_dim=1, input=self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        return self.output_module_12
