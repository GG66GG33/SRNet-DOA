import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torchOrth(Q):
    r = torch.linalg.matrix_rank(Q)
    u,s,v = torch.svd(Q)
    return u[:,:r]
class orthogonality_loss(nn.Module):
    def __init__(self):
        super(orthogonality_loss, self).__init__()

    # y_pred: Reconstructed noise subspace
    # y_true: Steering vector
    def forward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        loss = 0.0
        for i in range(batch_size):
            A = y_true[i].squeeze()
            Un = y_pred[i].squeeze()
            A = A / torch.max(torch.abs(A))
            Un = Un / torch.max(torch.abs(Un))
            Un_orth = torchOrth(Un)
            A_orth = torchOrth(A)
            P = Un_orth.conj().T @ A_orth
            orthogonality_measure = torch.norm(P, 'fro')
            loss += orthogonality_measure
        return loss / batch_size


