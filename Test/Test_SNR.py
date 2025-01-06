import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from Paper_compare_method import *
from Signal_creation import *
from Paper_evaluation import *
import warnings
import os

warnings.simplefilter("ignore")

############################################
##               Parameter setting        ##
############################################
DOA = [-10.49, 10.62]
computer_num = 1000
Snap = 100
N = 16
D = 2
Angle_interval = 0.01
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
    Error_Model = []
    Error_DeepCNN_Model = []
    Error_CV_Model = []
    Error_Music = []
    Error_CBF = []
    Error_MVDR = []
    Error_SS_Music = []
    Error_Toeplitz_Music = []
    Error_PM = []

    for i in range(computer_num):
        System_model = Samples(N=N, D=D, DOA=DOA, Snap=Snap)
        X, signal, A, noise = System_model.samples_creation(mode=mode, SNR=SNR, amp_fad=amp_fad, phi_fad=phi_fad)
        X = torch.tensor(X, dtype=torch.complex64)
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / Snap
        R = torch.tensor(R, dtype=torch.complex64)

        X = X / torch.max(torch.abs(X))
        R = R / torch.max(torch.abs(R))

        # Proposed
        DOA_pred_model, Spectrum_model, Un_model = md.Proposed_Model(R, Model_Param_Best_path, D)
        if print_DOA:
            print("\nModel:", DOA_pred_model)
        error1_Model = RMSE(DOA_pred_model, DOA)
        Error_Model.append(error1_Model)

        # DeepCNN_MODEL
        DOA_pred_DeepCNN_model, Spectrum_DeepCNN_model = md.DeepCNN_Model(signal, DeepCNN_Model_Param, D, A, SNR, Snap)
        if print_DOA:
            print("\nDeepCNN:", DOA_pred_DeepCNN_model)
        if DOA_pred_DeepCNN_model is not None and len(DOA_pred_DeepCNN_model) > 0:
            error1_DeepCNN_Model = RMSE(DOA_pred_DeepCNN_model, DOA)
            Error_DeepCNN_Model.append(error1_DeepCNN_Model)

        # CV_MODEL
        DOA_pred_CV_model, Spectrum_CV_model = md.CV_Model(R, CV_Model_Param, D)
        if print_DOA:
            print("\nDeepCNN:", DOA_pred_CV_model)
        error1_CV_Model = RMSE(DOA_pred_CV_model, DOA)
        Error_CV_Model.append(error1_CV_Model)


        # MUSIC
        DOA_pred_music, Spectrum_music, Un_music = md.Music(X, D)
        if print_DOA:
            print("\nMusic:", DOA_pred_music)
        error1_Music = RMSE(DOA_pred_music, DOA)
        Error_Music.append(error1_Music)

        # CBF
        DOA_pred_CBF, Spectrum_CBF = md.CBF(X, D)
        if print_DOA:
            print("\nCBF:", DOA_pred_CBF)
        error1_CBF = RMSE(DOA_pred_CBF, DOA)
        Error_CBF.append(error1_CBF)

        # MVDR
        DOA_pred_MVDR, Spectrum_MVDR = md.MVDR(X, D)
        if print_DOA:
            print("\nMVDR:", DOA_pred_MVDR)
        error1_MVDR = RMSE(DOA_pred_MVDR, DOA)
        Error_MVDR.append(error1_MVDR)

        # SS_Music
        DOA_pred_SS_music, Spectrum_SS_music, Un_SS_music = md.SS_Music(X, D, sub_array_size)
        if print_DOA:
            print("\nSS_Music:", DOA_pred_SS_music)
        error1_SS_Music = RMSE(DOA_pred_SS_music, DOA)
        Error_SS_Music.append(error1_SS_Music)

        # Toeplitz_Music
        DOA_pred_Toeplitz_Music, Spectrum_Toeplitz_Music, Un_Toeplitz_Music = md.Toeplitz_Music(X, D)
        if print_DOA:
            print("\nToeplitz_Music:", DOA_pred_Toeplitz_Music)
        error1_Toeplitz_Music = RMSE(DOA_pred_Toeplitz_Music, DOA)
        Error_Toeplitz_Music.append(error1_Toeplitz_Music)

        # PM
        DOA_pred_PM, Spectrum_PM = md.PM(X, D)
        if print_DOA:
            print("\nPM:", DOA_pred_PM)
        error1_PM = RMSE(DOA_pred_PM, DOA)
        Error_PM.append(error1_PM)

        progress_bar.update(1)

    results[SNR] = {
        'Model': np.sqrt(np.mean(Error_Model)),
        'DeepCNN': np.sqrt(np.mean(Error_DeepCNN_Model)),
        'CV': np.sqrt(np.mean(Error_CV_Model)),
        'Music': np.sqrt(np.mean(Error_Music)),
        'CBF': np.sqrt(np.mean(Error_CBF)),
        'MVDR': np.sqrt(np.mean(Error_MVDR)),
        'PM': np.sqrt(np.mean(Error_PM)),
        'SS_Music': np.sqrt(np.mean(Error_SS_Music)),
        'Toeplitz_Music': np.sqrt(np.mean(Error_Toeplitz_Music))
    }

    print(f"SNR = {SNR}")
    print("Proposed_Model RMSE:", results[SNR]['Model'])
    print("DeepCNN RMSE:", results[SNR]['DeepCNN'])
    print("CV RMSE:", results[SNR]['CV'])
    print("Music RMSE:", results[SNR]['Music'])
    print("CBF RMSE:", results[SNR]['CBF'])
    print("MVDR RMSE:", results[SNR]['MVDR'])
    print("PM RMSE:", results[SNR]['PM'])
    print("SS_Music RMSE:", results[SNR]['SS_Music'])
    print("Toeplitz_Music RMSE:", results[SNR]['Toeplitz_Music'])
    print()

progress_bar.close()

df = pd.DataFrame.from_dict(results, orient='index')

df.to_csv('SNR-9to9_Snap100_-10.49_10.62.csv')