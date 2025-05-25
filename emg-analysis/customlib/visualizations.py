import matplotlib.pyplot as plt

def plot_sensor_data(sensor):
    fig, axes = plt.subplots(len(sensor.channels), 1, figsize=(12, 4*len(sensor.channels)), sharex=True)
    for i, channel in enumerate(sensor.channels):
        axes[i].plot(channel.time, channel.voltage)
        axes[i].set_title(f'Channel {channel.id}')
        axes[i].set_ylabel('Voltage')
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
