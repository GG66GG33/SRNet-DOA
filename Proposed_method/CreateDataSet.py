import torch
from tqdm import tqdm
from itertools import combinations
from torch.distributions import Normal
from Signal_creation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def CreateDataSet_Train(mode, N, D, Snap, NumberOfSampels, SNR_values, Real, DataSet, Array_form, gap=None, Normalized = True, Save = True, DataSet_path= None, True_DOA=None):
    DataSet_RA = []
    print("Start Creating DataSet...")
    for SNR in SNR_values:
        print("Start Creating DataSet with SNR =", SNR)

        for i in tqdm(range(NumberOfSampels)):
            Sys_Model = Samples(N=N, D=D, DOA=True_DOA, Snap=Snap, gap=gap, Array_form=Array_form)
            X, signal, A, noise = Sys_Model.samples_creation(mode=mode, SNR=SNR, N_mean=0, N_Var=1,
                                                             S_mean=0, S_Var=1)
            X = torch.tensor(X, dtype=torch.complex64)
            A = torch.tensor(A, dtype=torch.complex64)
            DOA = torch.tensor(Sys_Model.DOA, dtype=torch.float64)
            # print(f"DOA = {DOA}")

            X_H = torch.conj(X).t()
            R = torch.matmul(X, X_H) / Snap
            R = torch.tensor(R, dtype=torch.complex64)

            if Normalized:
                A = A / torch.max(torch.abs(A))
                R = R / torch.max(torch.abs(R))

            if Real == True:
                R2 = torch.stack((torch.real(R), torch.imag(R)), dim=0)
                A = torch.stack((torch.real(A), torch.imag(A)), dim=0)

            DataSet_RA.append((R2, A))


    if Save:
        if DataSet == "DataSet_RA":
            torch.save(obj=DataSet_RA,
                   f=DataSet_path + '/DataSet_RA_{}_{}_D={}_N={}_Snap={}_SNR={}to{}'.format(mode, NumberOfSampels, D,
                                                                                        N, Snap, SNR_values[0], SNR_values[-1]) + '.h5')

        torch.save(obj=Sys_Model,
                   f=DataSet_path + '/Sys_Model_{}_{}_D={}_N={}_Snap={}_SNR={}to{}'.format(mode, NumberOfSampels, D,
                                                                                        N, Snap, SNR_values[0], SNR_values[-1]) + '.h5')

    return DataSet_RA, Sys_Model

def Read_Data(Data_path):
    Data = torch.load(Data_path)
    return Data