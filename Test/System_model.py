import numpy as np
import torch

class System_model(object):
    def __init__(self, N: int, D: int):
        self.N = N
        self.D = D
        self.dist = 1 / 2
        self.create_array()

    def create_array(self):
        self.array = np.linspace(0, self.N, self.N, endpoint=False)

    def SV_Creation(self, theta, f=1, Array_form="ULA", SV_tensor = False):
        if Array_form == "ULA":
            if SV_tensor:
                return torch.exp(-2 * 1j * torch.pi * f * torch.tensor(self.dist) * torch.tensor(self.array) * torch.sin(theta))
            else:
                return np.exp(-2 * 1j * np.pi * f * self.dist * self.array * np.sin(theta))