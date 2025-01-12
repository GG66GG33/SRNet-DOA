import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm
from Paper_compare_method_ZhengjiaoNum import *
from Signal_creation import *
from Paper_evaluation import *
import warnings
import os

warnings.simplefilter("ignore")
############################################
##      Loss_orthogonality_compute        ##
############################################
def torchOrth(Q):
    r = torch.linalg.matrix_rank(Q)
    u,s,v = torch.svd(Q)
    return u[:,:r]

def Compute_orthogonal_value(Un, A):
    A = torch.tensor(A, dtype=torch.complex64)
    Un = torch.tensor(Un, dtype=torch.complex64)
    A = A / torch.max(torch.abs(A))
    Un = Un / torch.max(torch.abs(Un))
    Un_orth = torchOrth(Un)
    A_orth = torchOrth(A)
    P = Un_orth.conj().T @ A_orth
    orthogonality_measure = torch.norm(P, 'fro')
    orthogonality_measure = np.array(orthogonality_measure.item())
    return orthogonality_measure
############################################
##                  parameter             ##
############################################
DOA = [-10.49, 10.62]
computer_num = 1000
Snap = 100
N = 16
D = 2
Angle_interval = 0.01
gap = 8
sub_array_size = 4
print_DOA = False

############################################
##                  model                 ##
############################################
mode = "exp-coherent"
amp_fad = 1
phi_fad = 0
Model_Param_Best_path = "/media/data/wtg/Pycharm_code/ULA_DOA/Model_Param/DataSet_RA_v5_zhengjiao_exp-coherent_16110_D=2_N=16_Snap=200_SNR=-10to10/model_best.pth"
DeepCNN_Model_Param = "/media/data/wtg/Pycharm_code/ULA_DOA/Model_Param_CompareDL/ULA_N16/DeepCNN_DataSet_R3_DOA_exp-coherent_16110_D=2_N=16_Snap=200_SNR=-10to10/model_best.pth"
CV_Model_Param = "/media/data/wtg/Pycharm_code/ULA_DOA/Compare_CV_CNN/CV_CNN_DOA-main/Code/Python/CV_CNN_DOA/Model_Param/Fine_tuning/model_best.pth"

md = Methods_DOA(N, D, Angle_interval)

SNR_range = range(-9, 11, 2)
results = {}

total_steps = len(SNR_range) * computer_num
progress_bar = tqdm(total=total_steps, desc='Processing SNR values', unit='steps')

for SNR in SNR_range:
    orthogonal_Model = []
    orthogonal_Music = []
    Error_CBF = []
    Error_MVDR = []
    Error_SS_Music = []
    Error_Toeplitz_Music = []
    Error_PM = []
    Error_CRLB = []

    for i in range(computer_num):
        System_model = Samples(N=N, D=D, DOA=DOA, Snap=Snap, gap=gap)
        X, signal, A, noise = System_model.samples_creation(mode=mode, SNR=SNR, amp_fad=amp_fad, phi_fad=phi_fad)
        X = torch.tensor(X, dtype=torch.complex64)
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / Snap
        R = torch.tensor(R, dtype=torch.complex64)

        X = X / torch.max(torch.abs(X))
        R = R / torch.max(torch.abs(R))

        # Proposed
        DOA_pred_model, Spectrum_model, Un_model = md.Model(R, Model_Param_Best_path, D)
        orthogonal1_Model = Compute_orthogonal_value(Un_model, A)
        orthogonal_Model.append(orthogonal1_Model)

        # MUSIC
        DOA_pred_music, Spectrum_music, Un_music = md.Music(X, D)
        orthogonal1_Music = Compute_orthogonal_value(Un_music, A)
        orthogonal_Music.append(orthogonal1_Music)

        # Toeplitz_Music
        DOA_pred_Toeplitz_Music, Spectrum_Toeplitz_Music, Un_Toeplitz_Music = md.Toeplitz_Music(X, D)
        orthogonal1_Toeplitz_Music = Compute_orthogonal_value(Un_Toeplitz_Music, A)
        orthogonal_Toeplitz_Music.append(orthogonal1_Toeplitz_Music)

        progress_bar.update(1)

    results[SNR] = {
        'Model': np.mean(orthogonal_Model),
        'Music': np.mean(orthogonal_Music),
        'Toeplitz_Music': np.mean(orthogonal_Toeplitz_Music)
    }

    print(f"SNR = {SNR}")
    print("Model:", results[SNR]['Model'])
    print("Music:", results[SNR]['Music'])
    print("Toeplitz_Music:", results[SNR]['Toeplitz_Music'])
    print()


progress_bar.close()
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('SNR-9to9_Snap100_-10.49_10.62_CV_orthogonal.csv')