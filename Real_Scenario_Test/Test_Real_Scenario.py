import h5py
import scipy.io
from Paper_compare_method import *
from Signal_creation import *
from Paper_evaluation import *
import warnings

warnings.simplefilter("ignore")

############################################
##             Parameter setting          ##
############################################
DOA = [-2.8, 2.1]
N = 8
D = 2
Angle_interval = 0.01
sub_array_size = 8
print_DOA = False

############################################
##                  model                 ##
############################################
Model_Param_Best_path = "/media/data/wtg/Pycharm_code/ULA_DOA/Model_Param/ULA_N8/DataSet_RA_v5_zhengjiao_exp-coherent_16110_D=2_N=8_Snap=200_SNR=-10to10/model_best.pth"
DeepCNN_Model_Param = "/media/data/wtg/Pycharm_code/ULA_DOA/Model_Param_CompareDL/ULA_N8/DeepCNN_DataSet_R3_DOA_exp-coherent_16110_D=2_N=8_Snap=200_SNR=-10to10/model_best.pth"
CV_Model_Param = "/media/data/wtg/Pycharm_code/ULA_DOA/Compare_CV_CNN/CV_CNN_DOA-main/Code/Python/CV_CNN_DOA/Model_Param/Fine_tuning_N8/model_best.pth"

md = Methods_DOA(N, D, Angle_interval)

results = {}

file_path = '/media/data/wtg/Pycharm_code/ULA_DOA/Matlab_code/data_room/data_analys/09_06_2024_17_09_43_aoadata.h5'
with h5py.File(file_path, 'r') as h5_file:
    X_real = h5_file['/X_real'][:]
    X_imag = h5_file['/X_imag'][:]
    X = X_real + 1j * X_imag
    R_real = h5_file['/R_real'][:]
    R_imag = h5_file['/R_imag'][:]
    R = R_real + 1j * R_imag

X = torch.tensor(X, dtype=torch.complex64)
X = torch.conj(X).t()
R = torch.tensor(R, dtype=torch.complex64)

X = X / torch.max(torch.abs(X))
R = R / torch.max(torch.abs(R))

# Proposed
DOA_pred_model, Spectrum_model, Un_model = md.Proposed_Model(R, Model_Param_Best_path, D)
if print_DOA:
    print("\nModel:", DOA_pred_model)

# DeepCNN_MODEL
DOA_pred_DeepCNN_model, Spectrum_DeepCNN_model = md.DeepCNN_Model_Shice(X, DeepCNN_Model_Param)
if print_DOA:
    print("\nDeepCNN:", DOA_pred_DeepCNN_model)

# CV_MODEL
DOA_pred_CV_model, Spectrum_CV_model = md.CV_Model(R, CV_Model_Param, D)
if print_DOA:
    print("\nModel:", DOA_pred_CV_model)

# MUSIC
DOA_pred_music, Spectrum_music, Un_music = md.Music(X, D)
if print_DOA:
    print("\nMusic:", DOA_pred_music)


# CBF
DOA_pred_CBF, Spectrum_CBF = md.CBF(X, D)
if print_DOA:
    print("\nCBF:", DOA_pred_CBF)

# MVDR
DOA_pred_MVDR, Spectrum_MVDR = md.MVDR(X, D)
if print_DOA:
    print("\nMVDR:", DOA_pred_MVDR)

# SS_Music
DOA_pred_SS_music, Spectrum_SS_music, Un_SS_music = md.SS_Music(X, D, sub_array_size)
if print_DOA:
    print("\nSS_Music:", DOA_pred_SS_music)

# Toeplitz_Music
DOA_pred_Toeplitz_Music, Spectrum_Toeplitz_Music, Un_Toeplitz_Music = md.Toeplitz_Music(X, D)
if print_DOA:
    print("\nToeplitz_Music:", DOA_pred_Toeplitz_Music)

# PM
DOA_pred_PM, Spectrum_PM = md.PM(X, D)
if print_DOA:
    print("\nPM:", DOA_pred_PM)


angle_min = -90
angle_max = 90
angle_interval = 0.01

length = int((angle_max - angle_min) / angle_interval) + 1
true_spectrum = np.zeros(length)
angles = np.arange(angle_min, angle_max + angle_interval, angle_interval)
for angle in DOA:
    index = np.argmin(np.abs(angles - angle))
    true_spectrum[index] = 1


def save_all_spectra_to_mat(file_name, spectra_dict):
    scipy.io.savemat(file_name, spectra_dict)

spectra_dict = {
    'Proposed_MODEL': Spectrum_model,
    'DeepCNN_MODEL': Spectrum_DeepCNN_model,
    'CV_MODEL': Spectrum_CV_model,
    'MUSIC': Spectrum_music,
    'CBF': Spectrum_CBF,
    'MVDR': Spectrum_MVDR,
    'SS_Music': Spectrum_SS_music,
    'Toeplitz_Music': Spectrum_Toeplitz_Music,
    'PM': Spectrum_PM,
    'true_spectrum': true_spectrum
}

save_all_spectra_to_mat('09_06_2024_17_09_43_aoadata_IQ.mat', spectra_dict)
