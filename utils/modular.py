import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pywt

class PCAAnalyzer:
    def __init__(self, n_components=None):
        self.pca = PCA(n_components=n_components)
    
    def fit(self, emg_signals):
        # Expect a list of EMGSignal objects
        data = np.array([signal.data for signal in emg_signals]).T
        self.pca.fit(data)
    
    def transform(self, emg_signals):
        data = np.array([signal.data for signal in emg_signals]).T
        transformed_data = self.pca.transform(data)
        return [EMGSignal(transformed_data[:, i], emg_signals[0].sampling_rate) 
                for i in range(transformed_data.shape[1])]
    
    def plot_explained_variance(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance Ratio')
        plt.grid(True)
        plt.show()

class ICAAnalyzer:
    def __init__(self, n_components=None):
        self.ica = FastICA(n_components=n_components, random_state=42)
    
    def fit(self, emg_signals):
        data = np.array([signal.data for signal in emg_signals]).T
        self.ica.fit(data)
    
    def transform(self, emg_signals):
        data = np.array([signal.data for signal in emg_signals]).T
        transformed_data = self.ica.transform(data)
        return [EMGSignal(transformed_data[:, i], emg_signals[0].sampling_rate) 
                for i in range(transformed_data.shape[1])]

class WaveletTransformer:
    def __init__(self, wavelet='db4', level=None):
        self.wavelet = wavelet
        self.level = level
    
    def decompose(self, emg_signal):
        coeffs = pywt.wavedec(emg_signal.data, self.wavelet, level=self.level)
        return coeffs
    
    def reconstruct(self, coeffs):
        return pywt.waverec(coeffs, self.wavelet)
    
    def denoise(self, emg_signal, threshold_func=None):
        coeffs = self.decompose(emg_signal)
        if threshold_func is None:
            threshold_func = lambda x: pywt.threshold(x, np.std(x)/2, mode='soft')
        denoised_coeffs = [threshold_func(coeff) if i > 0 else coeff 
                           for i, coeff in enumerate(coeffs)]
        denoised_data = self.reconstruct(denoised_coeffs)
        return EMGSignal(denoised_data, emg_signal.sampling_rate)
    
    def plot_coefficients(self, coeffs):
        fig, axes = plt.subplots(len(coeffs), 1, figsize=(12, 3*len(coeffs)), sharex=True)
        for i, coeff in enumerate(coeffs):
            axes[i].plot(coeff)
            axes[i].set_title(f'Level {len(coeffs)-1-i}')
            axes[i].set_ylabel('Amplitude')
        axes[-1].set_xlabel('Sample')
        plt.tight_layout()
        plt.show()

class LaplacianFilter:
    @staticmethod
    def apply_1d(emg_signal):
        # 1D Laplacian filter
        kernel = np.array([1, -2, 1])
        filtered_data = signal.convolve(emg_signal.data, kernel, mode='same')
        return EMGSignal(filtered_data, emg_signal.sampling_rate)
    
    @staticmethod
    def apply_2d(emg_signals):
        # 2D Laplacian filter for multi-channel EMG
        # Assumes emg_signals is a list of EMGSignal objects
        data = np.array([signal.data for signal in emg_signals])
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
        filtered_data = signal.convolve2d(data, kernel, mode='same', boundary='symm')
        return [EMGSignal(filtered_data[i], emg_signals[0].sampling_rate) 
                for i in range(filtered_data.shape[0])]

class EMGSignal:
    def __init__(self, data, sampling_rate):
        self.data = np.array(data)
        self.sampling_rate = sampling_rate
        self.time = np.arange(len(data)) / sampling_rate

class Visualizer:
    @staticmethod
    def plot_time_domain(emg_signal):
        plt.figure(figsize=(12, 4))
        plt.plot(emg_signal.time, emg_signal.data)
        plt.title('EMG Signal in Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_frequency_domain(emg_signal):
        n = len(emg_signal.data)
        freq = np.fft.fftfreq(n, 1/emg_signal.sampling_rate)
        fft_vals = np.abs(np.fft.fft(emg_signal.data))
        
        plt.figure(figsize=(12, 4))
        plt.plot(freq[:n//2], fft_vals[:n//2])
        plt.title('EMG Signal in Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

class Preprocessor:
    @staticmethod
    def bandpass_filter(emg_signal, lowcut, highcut, order=4):
        nyq = 0.5 * emg_signal.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, emg_signal.data)
        return EMGSignal(filtered_data, emg_signal.sampling_rate)

    @staticmethod
    def notch_filter(emg_signal, notch_freq, quality_factor=30):
        nyq = 0.5 * emg_signal.sampling_rate
        freq = notch_freq / nyq
        b, a = signal.iirnotch(freq, quality_factor)
        filtered_data = signal.lfilter(b, a, emg_signal.data)
        return EMGSignal(filtered_data, emg_signal.sampling_rate)

class Transformer:
    @staticmethod
    def fft(emg_signal):
        return np.fft.fft(emg_signal.data)

    @staticmethod
    def ifft(fft_data):
        return np.fft.ifft(fft_data).real

    @staticmethod
    def stft(emg_signal, window_size, hop_length):
        return signal.stft(emg_signal.data, fs=emg_signal.sampling_rate, 
                           window='hann', nperseg=window_size, noverlap=window_size-hop_length)

class Convolver:
    @staticmethod
    def convolve(emg_signal, kernel):
        return np.convolve(emg_signal.data, kernel, mode='same')

    @staticmethod
    def correlate(emg_signal1, emg_signal2):
        return np.correlate(emg_signal1.data, emg_signal2.data, mode='full')

class Sorter:
    @staticmethod
    def threshold_crossing(emg_signal, threshold):
        return np.where(emg_signal.data > threshold)[0]

    @staticmethod
    def peak_detection(emg_signal, height, distance):
        peaks, _ = signal.find_peaks(emg_signal.data, height=height, distance=distance)
        return peaks

class Clusterer:
    @staticmethod
    def kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data.reshape(-1, 1))
        return labels

