# %%
from customlib import DataSet, Trial
from customlib.muap_extraction import ckc, gradient_descent, tensor_decomposition, ica
from customlib.visualizations import plot_sensor_data
import matplotlib.pyplot as plt
import os

# Create a dataset
dataset = DataSet("../datasets", "naruto")

# Load a trial
trial = Trial(dataset.trials[0])
trial.load_metadata()
trial.load_sensors()

# %%
trial.sensors

#%%
# # Apply CKC
# muap_occurrences, ckc_template = ckc.ckc_extraction(trial.sensors)

# # Apply Gradient Descent
# gd_templates, gd_activations = gradient_descent.gradient_descent_extraction(trial.sensors)

# # Apply Tensor Decomposition
# spatial_patterns, temporal_patterns, activation_patterns = tensor_decomposition.tensor_decomposition_extraction(trial.sensors)

# # Apply ICA
# ica_templates, ica_activations = ica.ica_extraction(trial.sensors)

# # Visualize results
# plt.figure(figsize=(15, 10))

# plt.subplot(2, 2, 1)
# plt.plot(ckc_template)
# plt.title('CKC Template')

# plt.subplot(2, 2, 2)
# plt.plot(gd_templates.T)
# plt.title('Gradient Descent Templates')

# plt.subplot(2, 2, 3)
# plt.plot(temporal_patterns)
# plt.title('Tensor Decomposition Temporal Patterns')

# plt.subplot(2, 2, 4)
# plt.plot(ica_templates)
# plt.title('ICA Templates')

# plt.tight_layout()
# plt.show()