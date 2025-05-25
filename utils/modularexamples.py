import modular
# Example usage:
if __name__ == "__main__":
    # Generate sample EMG data
    t = np.linspace(0, 1, 1000)
    emg_data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(1000)
    emg_signal = EMGSignal(emg_data, sampling_rate=1000)

    # Visualization
    Visualizer.plot_time_domain(emg_signal)
    Visualizer.plot_frequency_domain(emg_signal)

    # Preprocessing
    filtered_signal = Preprocessor.bandpass_filter(emg_signal, lowcut=10, highcut=450)
    Visualizer.plot_time_domain(filtered_signal)

    # Transformation
    fft_data = Transformer.fft(filtered_signal)
    plt.plot(np.abs(fft_data))
    plt.show()

    # Convolution
    kernel = signal.gaussian(50, std=7)
    convolved_data = Convolver.convolve(filtered_signal, kernel)
    plt.plot(convolved_data)
    plt.show()

    # Sorting
    peaks = Sorter.peak_detection(filtered_signal, height=0.5, distance=50)
    plt.plot(filtered_signal.data)
    plt.plot(peaks, filtered_signal.data[peaks], "x")
    plt.show()

    # Clustering
    clusters = Clusterer.kmeans(filtered_signal.data, n_clusters=3)
    plt.scatter(np.arange(len(filtered_signal.data)), filtered_signal.data, c=clusters)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate sample multi-channel EMG data
    t = np.linspace(0, 1, 1000)
    emg_data1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(1000)
    emg_data2 = np.sin(2 * np.pi * 20 * t) + 0.5 * np.random.randn(1000)
    emg_data3 = np.sin(2 * np.pi * 30 * t) + 0.5 * np.random.randn(1000)
    
    emg_signals = [EMGSignal(data, sampling_rate=1000) for data in [emg_data1, emg_data2, emg_data3]]

    # PCA
    pca_analyzer = PCAAnalyzer(n_components=2)
    pca_analyzer.fit(emg_signals)
    pca_components = pca_analyzer.transform(emg_signals)
    pca_analyzer.plot_explained_variance()
    
    # ICA
    ica_analyzer = ICAAnalyzer(n_components=3)
    ica_analyzer.fit(emg_signals)
    ica_components = ica_analyzer.transform(emg_signals)
    
    # Wavelet Transform
    wt = WaveletTransformer(wavelet='db4', level=5)
    coeffs = wt.decompose(emg_signals[0])
    wt.plot_coefficients(coeffs)
    denoised_signal = wt.denoise(emg_signals[0])
    
    # Laplacian Filter
    laplacian_1d = LaplacianFilter.apply_1d(emg_signals[0])
    laplacian_2d = LaplacianFilter.apply_2d(emg_signals)
    
    # Visualize results
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    axes[0].plot(emg_signals[0].time, emg_signals[0].data)
    axes[0].set_title('Original Signal')
    axes[1].plot(pca_components[0].time, pca_components[0].data)
    axes[1].set_title('First PCA Component')
    axes[2].plot(ica_components[0].time, ica_components[0].data)
    axes[2].set_title('First ICA Component')
    axes[3].plot(denoised_signal.time, denoised_signal.data)
    axes[3].set_title('Wavelet Denoised Signal')
    axes[4].plot(laplacian_1d.time, laplacian_1d.data)
    axes[4].set_title('1D Laplacian Filtered Signal')
    plt.tight_layout()
    plt.show()