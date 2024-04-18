import torch


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input):
        output = self.weight.mv(input)

        # This calls the `forward` method of the `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
        output = self.linear(output)
        return output


scripted_module = torch.jit.script(MyModule(2, 3))
print(scripted_module)


import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.jit.trace produces a ScriptModule's conv1 and conv2
        self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
        self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

    def forward(self, input):
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        return input


scripted_module = torch.jit.script(MyModule())
print(scripted_module)


import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.jit.export
    def some_entry_point(self, input):
        return input + 10

    @torch.jit.ignore
    def python_only_fn(self, input):
        # This function won't be compiled, so any
        # Python APIs can be used
        import pdb

        pdb.set_trace()

    def forward(self, input):
        if self.training:
            self.python_only_fn(input)
        return input * 99


scripted_module = torch.jit.script(MyModule())
print(scripted_module.some_entry_point(torch.randn(2, 2)))
print(scripted_module(torch.randn(2, 2)))
print(scripted_module)
