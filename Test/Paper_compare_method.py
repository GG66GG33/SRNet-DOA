import scipy.signal
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Methods_DOA(object):
    def __init__(self, N, D, Angle_interval):
        self.angels = np.linspace(-1 * np.pi / 2, np.pi / 2, int(180/Angle_interval), endpoint=False)
        self.N = N
        self.D = D
        self.dist = 1 / 2
        self.create_array()

    def create_array(self):
        self.array = np.linspace(0, self.N, self.N, endpoint=False)


    def SV_Creation(self, theta, f=1, Array_form="ULA"):
        if Array_form == "ULA":
            return np.exp(-2 * 1j * np.pi * f * self.dist * self.array * np.sin(theta))

    def spectrum_calculation(self, Un, f=1, Array_form="ULA"):
        Spectrum_equation = []
        for angle in self.angels:
            a = self.SV_Creation(theta=angle, f=f, Array_form=Array_form)
            a = a[:Un.shape[0]]
            Spectrum_equation.append(np.conj(a).T @ Un @ np.conj(Un).T @ a)
        Spectrum_equation = np.array(Spectrum_equation, dtype=complex)
        Spectrum = 1 / Spectrum_equation
        return np.abs(Spectrum)


    def R_inv_spectrum_calculation(self, R_inv, f=1, Array_form="ULA"):
        Spectrum_equation = []
        for angle in self.angels:
            a = self.SV_Creation(theta=angle, f=f, Array_form=Array_form)
            a = a[:R_inv.shape[0]]
            Spectrum_equation.append(np.conj(a).T @ R_inv @ a)
        Spectrum_equation = np.array(Spectrum_equation, dtype=complex)
        Spectrum = 1 / Spectrum_equation
        return np.abs(Spectrum)


    def R_spectrum_calculation(self, R, f=1, Array_form="ULA"):
        Spectrum_equation = []
        for angle in self.angels:
            a = self.SV_Creation(theta=angle, f=f, Array_form=Array_form)
            a = a[:R.shape[0]]
            Spectrum_equation.append(np.conj(a).T @ R @ a)
        Spectrum_equation = np.array(Spectrum_equation, dtype=complex)
        Spectrum = Spectrum_equation
        return np.abs(Spectrum)


    def Q_spectrum_calculation(self, Q, f=1, Array_form="ULA"):
        Spectrum_equation = []
        for angle in self.angels:
            a = self.SV_Creation(theta=angle, f=f, Array_form=Array_form)

            Spectrum_equation.append(np.conj(a).T @ np.conj(Q).T @ Q @ a)
        Spectrum_equation = np.array(Spectrum_equation, dtype=complex)
        Spectrum = 1 / Spectrum_equation
        return np.abs(Spectrum)


    #####################################
    ##       Proposed_Model      ##
    #####################################
    def Proposed_Model(self, R, model_path, D):
        model = torch.load(model_path, map_location=device).to(device)
        R2 = torch.stack((torch.real(R), torch.imag(R)), dim=0)
        R2_batch = R2.float()
        R2_batch = R2_batch.unsqueeze(0)
        Un_train = model(R2_batch.to(device))
        Un_train_save = Un_train.squeeze()
        Un_pred = torch.complex(Un_train_save[0], Un_train_save[1])
        Un_pred = Un_pred.cpu().detach().numpy()
        Un_pred = Un_pred.astype(np.complex64)    # 网络预测的Un

        Spectrum = self.spectrum_calculation(Un_pred)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred = self.angels[DOA_pred] * 180 / np.pi
        predicted_DOA = DOA_pred[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return predicted_DOA, Spectrum, Un_pred

    #####################################
    ##       CV_Model      ##
    #####################################
    def CV_Model(self, R, model_path, D):
        model = torch.load(model_path, map_location=device).to(device)
        # R2 = torch.stack((torch.real(R), torch.imag(R)), dim=0)
        # R2_batch = R2.float()
        R2_batch = R.unsqueeze(0)
        Spectrum_pred = model(R2_batch.to(device))
        Spectrum_pred = Spectrum_pred.cpu().detach().numpy()
        Spectrum_pred = np.squeeze(Spectrum_pred)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum_pred)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum_pred[x], reverse=True)
        angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 180, endpoint=False)
        DOA_pred = angels[DOA_pred] * 180 / np.pi
        predicted_DOA = DOA_pred[:D]
        Spectrum_pred = Spectrum_pred / np.max(np.abs(Spectrum_pred))
        return predicted_DOA, Spectrum_pred

    #####################################
    ##       DeepCNN_Model      #########
    #####################################
    def DeepCNN_Model(self, signal, model_path, D, A, SNR, Snap):
        noise_power = 10 ** (-SNR / 10)
        A = torch.tensor(A, dtype=torch.complex64)
        A_H = torch.conj(A).t()
        Eta = np.sqrt(noise_power) * (np.random.randn(self.N, Snap) + 1j * np.random.randn(self.N, Snap)) / np.sqrt(2)
        amp = np.sqrt((10 ** (SNR / 10)))
        signal = signal / amp
        Y = A @ signal + Eta
        R = np.cov(Y)
        R = torch.tensor(R)

        model = torch.load(model_path, map_location=device).to(device)
        model.to(device)
        R3 = torch.stack((torch.real(R), torch.imag(R), torch.angle(R)), dim=0)
        R3_batch = R3.float()
        R3_batch = R3_batch.unsqueeze(0)
        Spectrum_pred = model(R3_batch.to(device))
        Spectrum_pred = Spectrum_pred.cpu().detach().numpy()
        Spectrum_pred = np.squeeze(Spectrum_pred)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum_pred)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum_pred[x], reverse=True)
        angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 180, endpoint=False)
        DOA_pred = angels[DOA_pred] * 180 / np.pi
        predicted_DOA = DOA_pred[:D]
        Spectrum_pred = Spectrum_pred / np.max(np.abs(Spectrum_pred))

        return predicted_DOA, Spectrum_pred




    #####################################
    ##               Music             ##
    #####################################
    def Music(self, X, D):
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / X.size(1)
        R = R.numpy()
        eig_values, eig_vectors = np.linalg.eig(R)
        Un = eig_vectors[:, np.argsort(eig_values)[::-1]][:, D:]
        Spectrum= self.spectrum_calculation(Un)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred_music = self.angels[DOA_pred] * 180 / np.pi
        DOA_pred_music = DOA_pred_music[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return DOA_pred_music, Spectrum, Un

    #####################################
    ##               CBF             ##
    #####################################
    def CBF(self, X, D):
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / X.size(1)
        R = R.numpy()
        Spectrum = self.R_spectrum_calculation(R)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred_CBF = self.angels[DOA_pred] * 180 / np.pi
        DOA_pred_CBF = DOA_pred_CBF[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return DOA_pred_CBF, Spectrum


    #####################################
    ##               MVDR             ##
    #####################################
    def MVDR(self, X, D):
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / X.size(1)
        R = R.numpy()
        R_inv = np.linalg.inv(R)
        Spectrum = self.R_inv_spectrum_calculation(R_inv)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred_MVDR = self.angels[DOA_pred] * 180 / np.pi
        DOA_pred_MVDR = DOA_pred_MVDR[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return DOA_pred_MVDR, Spectrum


    #####################################
    ##            SS_Music             ##
    #####################################
    def SS_Music(self, X, D, sub_array_size):
        R = np.cov(X)
        number_of_sub_arrays = self.N - sub_array_size + 1
        R_x = np.zeros((sub_array_size, sub_array_size)) + 1j * np.zeros((sub_array_size, sub_array_size))
        for j in range(number_of_sub_arrays):
            R_sub = R[j:j + sub_array_size, j:j + sub_array_size]
            R_x += R_sub
        R_x /= number_of_sub_arrays

        eig_values, eig_vectors = np.linalg.eig(R_x)
        Un = eig_vectors[:, np.argsort(eig_values)[::-1]][:, D:]
        Spectrum = self.spectrum_calculation(Un)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred_SS_music = self.angels[DOA_pred] * 180 / np.pi
        DOA_pred_SS_music = DOA_pred_SS_music[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return DOA_pred_SS_music, Spectrum, Un


    #####################################
    ##         Toeplitz_music          ##
    #####################################
    def Toeplitz_Music(self, X, D):
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / X.size(1)
        R = R.numpy()
        dd = np.zeros((2*self.N - 1), dtype=complex)
        N = self.N
        for i in range(-(N-1), N):
            diag_elements = np.diag(R, i)
            dd[i + N - 1] = np.mean(diag_elements)

        R_toeplitz = np.zeros((N, N), dtype=complex)

        for k in range(N):
            R_toeplitz[k, k] = dd[N - 1]

        for i in range(1, N):
            for k in range(N - i):
                R_toeplitz[k + i, k] = dd[N - i - 1]
                R_toeplitz[k, k + i] = dd[N + i - 1]

        eig_values, eig_vectors = np.linalg.eig(R_toeplitz)
        Un = eig_vectors[:, np.argsort(eig_values)[::-1]][:, D:]
        Spectrum = self.spectrum_calculation(Un)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred_Toeplitz_music = self.angels[DOA_pred] * 180 / np.pi
        DOA_pred_Toeplitz_music = DOA_pred_Toeplitz_music[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return DOA_pred_Toeplitz_music, Spectrum, Un




    #####################################
    ##                 PM              ##
    #####################################
    def PM(self, X, D):
        X_H = torch.conj(X).t()
        R = torch.matmul(X, X_H) / X.size(1)
        R = R.numpy()
        G = R[:, :D]
        H = R[:, D:]
        G_H = G.conj().T
        P = np.linalg.inv(G_H @ G) @ G_H @ H    # 传播算子矩阵
        P_H = P.conj().T
        Q = np.hstack([P_H, -np.diag(np.ones(self.N-D))])    # Q矩阵
        Spectrum = self.Q_spectrum_calculation(Q)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key=lambda x: Spectrum[x], reverse=True)
        DOA_pred_pm = self.angels[DOA_pred] * 180 / np.pi
        DOA_pred_pm = DOA_pred_pm[:D]
        Spectrum = Spectrum / np.max(np.abs(Spectrum))
        return DOA_pred_pm, Spectrum