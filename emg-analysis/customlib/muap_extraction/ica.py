import numpy as np
from sklearn.decomposition import FastICA
from scipy import signal
from ..signal_processing import Sensors

def ica_extraction(sensors: Sensors, n_components=5):
    emg_data = np.array([sensor.numpy_channels for sensor in sensors.sensors]).T
    
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(emg_data)
    mixing_matrix = ica.mixing_
    
    # Post-processing: Identify MUAP-like components
    muap_components = []
    for i, source in enumerate(sources.T):
        # Calculate power spectrum
        f, pxx = signal.welch(source, fs=sensors.sensors[0].sr)
        
        # Check if the power spectrum resembles a typical MUAP
        if np.argmax(pxx) > 20 and np.argmax(pxx) < 150:  # Typical MUAP frequency range
            muap_components.append(i)
    
    # Extract MUAP templates and activations
    muap_templates = mixing_matrix[:, muap_components]
    muap_activations = sources[:, muap_components]
    
    return muap_templates, muap_activations

# Usage
# muap_templates, muap_activations = ica_extraction(trial.sensors)