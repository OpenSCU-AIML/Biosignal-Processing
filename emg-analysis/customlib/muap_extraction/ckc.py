import numpy as np
from scipy import signal
from ..signal_processing import Sensors

def apply_spatial_filter(emg_data):
    # Apply Laplacian spatial filter
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return signal.convolve2d(emg_data, kernel, mode='same', boundary='symm')

def estimate_muap_template(emg_data, sampling_rate):
    # Simplified MUAP template estimation
    return np.mean(emg_data, axis=1)

def ckc_extraction(sensors: Sensors, window_size=100):
    emg_data = np.array([sensor.numpy_channels for sensor in sensors.sensors]).T
    
    # Apply spatial filtering
    filtered_data = apply_spatial_filter(emg_data)
    
    # Estimate MUAP template
    template = estimate_muap_template(filtered_data, sensors.sensors[0].sr)
    
    # Perform CKC
    correlation = signal.correlate2d(filtered_data, template.reshape(-1, 1), mode='same')
    
    # Detect MUAP occurrences
    threshold = np.mean(correlation) + 2 * np.std(correlation)
    muap_occurrences = np.where(correlation > threshold)
    
    return muap_occurrences, template

# Usage
# muap_occurrences, template = ckc_extraction(trial.sensors)