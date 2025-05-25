import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from ..signal_processing import Sensors

def tensor_decomposition_extraction(sensors: Sensors, n_components=5, window_size=100):
    emg_data = np.array([sensor.numpy_channels for sensor in sensors.sensors])
    n_channels, n_samples = emg_data.shape
    
    # Reshape data into a 3D tensor: (channel, time, epoch)
    n_epochs = n_samples // window_size
    tensor = emg_data[:, :n_epochs * window_size].reshape(n_channels, window_size, n_epochs)
    
    # Perform Tucker decomposition
    core, factors = tucker(tensor, rank=[n_channels, n_components, n_components])
    
    # Extract MUAP templates and activation patterns
    spatial_patterns = factors[0]
    temporal_patterns = factors[1]
    activation_patterns = factors[2]
    
    return spatial_patterns, temporal_patterns, activation_patterns

# Usage
# spatial_patterns, temporal_patterns, activation_patterns = tensor_decomposition_extraction(trial.sensors)