import torch


class Gradient(torch.nn.Module):
    '''
    Implements the gradient and its adjoint with zero boundary conditions.
    '''
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x = [batch, channel, dim_x, dim_y]
        dxright = torch.cat([x[:, :, 1:, :], x[:, :, -1:, :]], dim=2)
        dyright = torch.cat([x[:, :, :, 1:], x[:, :, :, -1:]], dim=3)
        return torch.cat([dxright - x, dyright - x], dim=1)

    @staticmethod
    def adjoint(y):
        y1 = y[:, 0:1, ...]
        y2 = y[:, 1:, ...]
        dxleft = torch.cat([torch.zeros(y.size(0), 1, 1, y.size(3)).to(y.device), y1[:, :, :-1, :]], dim=2)
        dyleft = torch.cat([torch.zeros(y.size(0), 1, y.size(2), 1).to(y.device), y2[:, :, :, :-1]], dim=3)
        return -((y1 - dxleft) + (y2 - dyleft))
