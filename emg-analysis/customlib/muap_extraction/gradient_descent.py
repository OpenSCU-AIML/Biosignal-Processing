import numpy as np
from scipy.optimize import minimize
from ..signal_processing import Sensors

def cost_function(params, emg_data, n_muaps, n_samples):
    templates = params[:n_muaps * n_samples].reshape(n_muaps, n_samples)
    activations = params[n_muaps * n_samples:].reshape(n_muaps, -1)
    
    reconstruction = np.sum(templates[:, :, np.newaxis] * activations[:, np.newaxis, :], axis=0)
    error = np.sum((emg_data - reconstruction) ** 2)
    
    # Spatio-temporal regularization
    spatial_reg = np.sum(np.diff(templates, axis=1) ** 2)
    temporal_reg = np.sum(np.diff(activations, axis=1) ** 2)
    
    return error + 0.1 * spatial_reg + 0.1 * temporal_reg

def gradient_descent_extraction(sensors: Sensors, n_muaps=5, template_length=50):
    emg_data = np.array([sensor.numpy_channels for sensor in sensors.sensors]).T
    n_channels, n_samples = emg_data.shape
    
    # Initialize templates and activations
    initial_params = np.random.rand(n_muaps * template_length + n_muaps * n_samples)
    
    # Optimize using L-BFGS-B algorithm
    result = minimize(cost_function, initial_params, args=(emg_data, n_muaps, template_length),
                      method='L-BFGS-B', options={'maxiter': 100})
    
    # Extract optimized templates and activations
    optimized_params = result.x
    templates = optimized_params[:n_muaps * template_length].reshape(n_muaps, template_length)
    activations = optimized_params[n_muaps * template_length:].reshape(n_muaps, n_samples)
    
    return templates, activations

# Usage
# templates, activations = gradient_descent_extraction(trial.sensors)