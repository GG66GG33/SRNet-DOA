import pandas as pd
from tqdm import tqdm
from Paper_compare_method import *
from Signal_creation import *
from Paper_evaluation import *
import warnings

warnings.simplefilter("ignore")

def generate_DOA(diff):
    while True:
        theta1 = np.random.normal(0, 1)
        theta2 = theta1 + diff
        if -90 <= theta1 < 90 and -90 <= theta2 < 90:
            return sorted([theta1, theta2], reverse=True)


def check_estimation(theta1, theta2, theta1_hat, theta2_hat, diff):
    return (abs(theta1_hat - theta1) < diff / 2) and (abs(theta2_hat - theta2) < diff / 2)

############################################
##        Parameter setting               ##
############################################
computer_num = 1000
Snap = 200
SNR = 10
N = 16
D = 2
Angle_interval = 0.01
sub_array_size = int(N / 4)
print_DOA = False

############################################
##                  model               ##
############################################
mode = "exp-coherent"
amp_fad = 1
phi_fad = 0
Model_Param_Best_path = "/media/data/wtg/Pycharm_code/ULA_DOA/Model_Param/DataSet_RA_v5_zhengjiao_exp-coherent_16110_D=2_N=16_Snap=200_SNR=-10to10/model_best.pth"
DeepCNN_Model_Param = "/media/data/wtg/Pycharm_code/ULA_DOA/Model_Param_CompareDL/ULA_N16/DeepCNN_DataSet_R3_DOA_exp-coherent_16110_D=2_N=16_Snap=200_SNR=-10to10/model_best.pth"
CV_Model_Param = "/media/data/wtg/Pycharm_code/ULA_DOA/Compare_CV_CNN/CV_CNN_DOA-main/Code/Python/CV_CNN_DOA/Model_Param/Fine_tuning/model_best.pth"

md = Methods_DOA(N, D, Angle_interval)

diff_range = np.arange(0.5, 8.5, 0.5)


results = {}


total_steps = len(diff_range) * computer_num
progress_bar = tqdm(total=total_steps, desc='Processing resolution values', unit='steps')

for diff in diff_range:
    Success_Model = 0
    Success_DeepCNN_Model = 0
    Success_CV_Model = 0
    Success_Music = 0
    Success_CBF = 0
    Success_MVDR = 0
    Success_SS_Music = 0
    Success_Toeplitz_Music = 0
    Success_PM = 0
    for i in range(computer_num):
        DOA = generate_DOA(diff)

        System_model = Samples(N=N, D=D, DOA=DOA, Snap=Snap)
        X, signal, A, noise = System_model.samples_creation(mode=mode, SNR=SNR, amp_fad=amp_fad, phi_fad=phi_fad)
        X = torch.tensor(X, dtype=torch.complex64)
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / Snap
        R = torch.tensor(R, dtype=torch.complex64)  # R

        X = X / torch.max(torch.abs(X))
        R = R / torch.max(torch.abs(R))

        # Proposed_Model
        DOA_pred_model, Spectrum_model, Un_model = md.Proposed_Model(R, Model_Param_Best_path, D)
        if print_DOA:
            print("\nModel:", DOA_pred_model)
        predicted_DOA_model = sorted([DOA_pred_model[0], DOA_pred_model[1]], reverse=True)
        if check_estimation(DOA[0], DOA[1], predicted_DOA_model[0], predicted_DOA_model[1], diff):
            Success_Model += 1


        # DeepCNN_MODEL
        DOA_pred_DeepCNN_model, Spectrum_DeepCNN_model = md.DeepCNN_Model(signal, DeepCNN_Model_Param, D, A, SNR, Snap)
        if print_DOA:
            print("\nDeepCNN:", DOA_pred_DeepCNN_model)
        if DOA_pred_DeepCNN_model is not None and len(DOA_pred_DeepCNN_model) > D-1:
            DOA_pred_DeepCNN_model = sorted([DOA_pred_DeepCNN_model[0], DOA_pred_DeepCNN_model[1]], reverse=True)
            if check_estimation(DOA[0], DOA[1], DOA_pred_DeepCNN_model[0], DOA_pred_DeepCNN_model[1], diff):
                Success_DeepCNN_Model += 1

        # CV_MODEL
        DOA_pred_CV_model, Spectrum_CV_model = md.CV_Model(R, CV_Model_Param, D)
        if print_DOA:
            print("\nDeepCNN:", DOA_pred_CV_model)
        error1_CV_Model = RMSE(DOA_pred_CV_model, DOA)
        if DOA_pred_CV_model is not None and len(DOA_pred_CV_model) > D - 1:
            DOA_pred_CV_model = sorted([DOA_pred_CV_model[0], DOA_pred_CV_model[1]], reverse=True)
            if check_estimation(DOA[0], DOA[1], DOA_pred_CV_model[0], DOA_pred_CV_model[1], diff):
                Success_CV_Model += 1

        # MUSIC
        DOA_pred_music, Spectrum_music, Un_music = md.Music(X, D)
        if print_DOA:
            print("\nMusic:", DOA_pred_music)
        DOA_pred_music = sorted([DOA_pred_music[0], DOA_pred_music[1]], reverse=True)
        if check_estimation(DOA[0], DOA[1], DOA_pred_music[0], DOA_pred_music[1], diff):
            Success_Music += 1


        # CBF
        DOA_pred_CBF, Spectrum_CBF = md.CBF(X, D)
        if print_DOA:
            print("\nCBF:", DOA_pred_CBF)
        DOA_pred_CBF = sorted([DOA_pred_CBF[0], DOA_pred_CBF[1]], reverse=True)
        if check_estimation(DOA[0], DOA[1], DOA_pred_CBF[0], DOA_pred_CBF[1], diff):
            Success_CBF += 1


        # MVDR
        DOA_pred_MVDR, Spectrum_MVDR = md.MVDR(X, D)
        if print_DOA:
            print("\nMVDR:", DOA_pred_MVDR)
        DOA_pred_MVDR = sorted([DOA_pred_MVDR[0], DOA_pred_MVDR[1]], reverse=True)
        if check_estimation(DOA[0], DOA[1], DOA_pred_MVDR[0], DOA_pred_MVDR[1], diff):
            Success_MVDR += 1

        # SS_Music
        DOA_pred_SS_music, Spectrum_SS_music, Un_SS_music = md.SS_Music(X, D, sub_array_size)
        if print_DOA:
            print("\nSS_Music:", DOA_pred_SS_music)
        if DOA_pred_SS_music is not None and len(DOA_pred_SS_music) > D-1:
            DOA_pred_SS_music = sorted([DOA_pred_SS_music[0], DOA_pred_SS_music[1]], reverse=True)
            if check_estimation(DOA[0], DOA[1], DOA_pred_SS_music[0], DOA_pred_SS_music[1], diff):
                Success_SS_Music += 1


        # Toeplitz_Music
        DOA_pred_Toeplitz_Music, Spectrum_Toeplitz_Music, Un_Toeplitz_Music = md.Toeplitz_Music(X, D)
        if print_DOA:
            print("\nToeplitz_Music:", DOA_pred_Toeplitz_Music)
        DOA_pred_Toeplitz_Music = sorted([DOA_pred_Toeplitz_Music[0], DOA_pred_Toeplitz_Music[1]], reverse=True)
        if check_estimation(DOA[0], DOA[1], DOA_pred_Toeplitz_Music[0], DOA_pred_Toeplitz_Music[1], diff):
            Success_Toeplitz_Music += 1

        # PM
        DOA_pred_PM, Spectrum_PM = md.PM(X, D)
        if print_DOA:
            print("\nPM:", DOA_pred_PM)
        DOA_pred_PM = sorted([DOA_pred_PM[0], DOA_pred_PM[1]], reverse=True)
        if check_estimation(DOA[0], DOA[1], DOA_pred_PM[0], DOA_pred_PM[1], diff):
            Success_PM += 1

        progress_bar.update(1)

    results[diff] = {
        'Model': (Success_Model / computer_num) * 100,
        'DeepCNN': (Success_DeepCNN_Model / computer_num) * 100,
        'CV': (Success_CV_Model / computer_num) * 100,
        'Music': (Success_Music / computer_num) * 100,
        'CBF': (Success_CBF / computer_num) * 100,
        'MVDR': (Success_MVDR / computer_num) * 100,
        'PM': (Success_PM / computer_num) * 100,
        'SS_Music': (Success_SS_Music / computer_num) * 100,
        'Toeplitz_Music': (Success_Toeplitz_Music / computer_num) * 100
    }

    print(f"diff = {diff}")
    print("Proposed_Model:", results[diff]['Model'])
    print("DeepCNN:", results[diff]['DeepCNN'])
    print("CV:", results[diff]['CV'])
    print("Music:", results[diff]['Music'])
    print("CBF:", results[diff]['CBF'])
    print("MVDR:", results[diff]['MVDR'])
    print("PM:", results[diff]['PM'])
    print("SS_Music:", results[diff]['SS_Music'])
    print("Toeplitz_Music:", results[diff]['Toeplitz_Music'])
    print()


progress_bar.close()

df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('Diff0.5to8_SNR10_Snap200_angle_rand.csv')
