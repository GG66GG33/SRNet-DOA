import numpy as np

from System_model import *

def create_DOA_with_gap(D, gap, Array_form='ULA'):    
    if gap == None:
        if Array_form == 'ULA':
            DOA = np.random.randint(-90, 90, size=D)
    return DOA


class Samples(System_model):
    def __init__(self, N: int,
                 D: int,
                 DOA: list,
                 Snap: int,
                 gap: int,
                 Array_form = 'ULA'):
        super().__init__(N, D)
        self.Snap = Snap
        if DOA == None:
            self.DOA = (np.pi / 180) * np.array(create_DOA_with_gap(D=self.D, gap = gap, Array_form = Array_form))
        else:
            self.DOA = (np.pi / 180) * np.array(DOA)

    def samples_creation(self, mode, SNR, Array_form='ULA', amp_fad=1, phi_fad="rand", N_mean=0, N_Var=1,
                         S_mean=0, S_Var=1):
       signal = self.signal_creation(mode, SNR, amp_fad, phi_fad, S_mean, S_Var)
       noise = self.noise_creation(N_mean, N_Var)
       if Array_form == 'ULA':
           A = np.array([self.SV_Creation(theta, Array_form=Array_form) for theta in self.DOA]).T

       X = (A @ signal) + noise
       return X, signal, A, noise


    def signal_creation(self, mode:str, SNR:int, amp_fad=1, phi_fad="rand", S_mean=0, S_Var=1):
        power = (10 ** (SNR/10))
        amplitude = np.sqrt(power)
        if mode == "exp-coherent":
            sig = amplitude * (np.sqrt(2) / 2) * np.sqrt(S_Var) * (
                    np.random.randn(1, self.Snap) + 1j * np.random.randn(1, self.Snap)) + S_mean
            generator = Exp_CoherentSignalGenerator(sig, self.D - 1, amp_fad, phi_fad)
            coherent_signals = generator.generate_signals()
            combined_signals = np.vstack([sig, coherent_signals])

            return combined_signals

    def noise_creation(self, N_mean, N_Var):
        noise_value = np.sqrt(N_Var) * (np.sqrt(2) / 2) * (
                np.random.randn(self.N, self.Snap) + 1j * np.random.randn(self.N, self.Snap)) + N_mean
        return noise_value


class Exp_CoherentSignalGenerator:
    def __init__(self, reference_signal, num_sources, amp_fad, phi_fad):
        self.s1 = reference_signal
        self.num_sources = num_sources
        self.amp_fad = amp_fad
        self.phi_fad = phi_fad

    def generate_signals(self):
        _, snap_count = self.s1.shape
        coherent_signals = np.zeros((self.num_sources, snap_count), dtype=complex)

        for p in range(self.num_sources):
            if self.amp_fad == "rand":
                amp_fad = np.random.uniform(0, 1)
            else:
                amp_fad = self.amp_fad

            if self.phi_fad == "rand":
                phi_fad = np.random.uniform(0, 2*np.pi)
            else:
                phi_fad = self.phi_fad

            cp = amp_fad * np.exp(1j * phi_fad)

            coherent_signals[p, :] = cp * self.s1


        return coherent_signals
