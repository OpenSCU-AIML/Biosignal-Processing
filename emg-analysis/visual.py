# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot
import numpy as np


# chanarr = [[ch1.V.numpy][ch2.V.numpy]]
class ChanPlot:
    def __init__(self, chanlist):
        self.chanlist = chanlist

    def tickplot(self, data, ticks):
        plt.style.use("ggplot")
        emg_plots = []
        nchan = len(data)
        t = range(len(data[0]))
        figure, axs = plt.subplots(nchan, 1, figsize=(10, nchan))
        figure.suptitle("Raw Data", fontsize=16)
        for i in range(nchan):
            plt.subplot(nchan, 1, i + 1)
            spike = np.zeros(len(data[i]))
            for j in ticks[i]:
                spike[j] = 1
            plt.plot(t, data[i], label=f"ch {i+1}")
            plt.plot(
                t,
                spike,
                "|",
                markersize=20,
                color="green",
                label="spikes",
            )
        plt.xlabel("time")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.show()

    def chanplot(self, channels):
        plt.style.use("ggplot")
        emg_plots = []
        nchan = len(channels)
        figure, axs = plt.subplots(nchan, 1, figsize=(10, nchan * 2))
        figure.suptitle("Raw Data", fontsize=16)

        # Initialize the plots
        for i in range(nchan):
            axs[i].set_ylabel(f"Channel {i + 1}")
            (line,) = axs[i].plot([], [], lw=2)
            emg_plots.append(line)

        def update(index=0, nsamples=500):
            cur = (index, nsamples)
            # next(slices)
            data = [channels[i][cur[0] : cur[1]] for i in range(nchan)]
            if len(data[0]) > nsamples:
                data = data[-nsamples:][:]
            x_data = list(range(len(data[0])))
            for i, line in enumerate(emg_plots):
                y_data = data[i]
                line.set_data(x_data, y_data)

                axs[i].relim()
                axs[i].autoscale_view()

        update()
        plt.show()

    def visualize(self, rate=0.1, sampling=2222, num_samples=500, y_axes=None):
        """Visualize the incoming raw EMG in a plot (all channels together).

        Parameters
        ----------
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        y_axes: list (optional)
            A list of two elements consisting the bounds for the y-axis (e.g., [-1,1]).
        """
        pyplot.style.use("ggplot")
        num_channels = len(self.chanlist)  ##
        emg_plots = []
        figure, ax = pyplot.subplots()
        figure.suptitle("Raw Data", fontsize=16)
        for i in range(0, num_channels):
            emg_plots.append(ax.plot([], [], label="CH" + str(i + 1)))
        figure.legend()

        ttotal = self.chanlist[0].shape[0] // sampling
        slices = iter(
            [
                (idx, idx + round(rate * sampling))
                for idx in range(0, ttotal, round(rate * sampling))
            ]
        )
        print(len(slices))

        def update(frame):
            cur = next(slices)
            data = self.chanlist[cur[0] : cur[1]][:]
            if len(data) > num_samples:
                data = data[-num_samples:]
            if len(data) > 0:
                x_data = list(range(0, len(data)))
                for i in range(0, num_channels):
                    y_data = data[:, i]
                    emg_plots[i][0].set_data(x_data, y_data)
                figure.gca().relim()
                figure.gca().autoscale_view()
                if not y_axes is None:
                    figure.gca().set_ylim(y_axes)
            return (emg_plots,)

        animation = FuncAnimation(figure, update, interval=100)
        pyplot.show()

    # visualize_channels(self.chanraw())
    def visualize_channel(channel, rate=0.1, sampling=2222, num_samples=500):
        """Visualize a single channel with real-time updating.

        Parameters
        ----------
        channel: np.ndarray
            A 1D array representing the data for a single channel.
        rate: float (optional), default=0.1
            The rate at which to update the plot (in seconds).
        sampling: int (optional), default=2222
            The number of samples per second.
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        """
        fig, ax = plt.subplots()
        ax.set_title("Channel Data")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Sample Index")

        (line,) = ax.plot([], [], lw=2)
        ax.set_xlim(0, num_samples)
        ax.set_ylim(np.min(channel), np.max(channel))

        def update(frame):
            start_idx = frame * int(rate * sampling)
            end_idx = start_idx + int(rate * sampling)

            data = channel[start_idx:end_idx]
            if len(data) > num_samples:
                data = data[-num_samples:]

            x_data = np.arange(len(data))
            line.set_data(x_data, data)

            return (line,)

        num_frames = len(channel) // int(rate * sampling)
        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=100,
            blit=True,
            cache_frame_data=False,
        )

        plt.show()

    def visualize_channels(
        self, channels, rate=0.1, sampling=2222, num_samples=500, y_axes=None
    ):
        """Visualize individual channels (each channel in its own plot).

        Parameters
        ----------
        channels: np.ndarray
            A 2D array where each row is a channel and each column is a sample.
        rate: float (optional), default=0.1
            The rate at which to update the plot.
        sampling: int (optional), default=2222
            The number of samples per second.
        num_samples: int (optional), default=500
            The number of samples to show in the plot.
        y_axes: list of tuples (optional)
            A list of tuples specifying the y-axis limits for each channel plot.
        """
        pyplot.style.use("ggplot")
        num_channels = channels.shape[0]
        emg_plots = []
        figure, axs = pyplot.subplots(num_channels, 1, figsize=(10, num_channels * 2))
        figure.suptitle("Raw Data", fontsize=16)

        # Initialize the plots
        for i in range(num_channels):
            axs[i].set_ylabel(f"Channel {i + 1}")
            (line,) = axs[i].plot([], [], lw=2)
            emg_plots.append(line)

        ttotal = channels.shape[1]
        slices = iter(
            [
                (idx, idx + round(rate * sampling))
                for idx in range(0, ttotal, round(rate * sampling))
            ]
        )

        def update(frame):
            try:
                cur = next(slices)
            except StopIteration:
                return emg_plots  # Stop iteration when slices are exhausted

            # Correct slicing for a 2D array
            data = channels[:, cur[0] : cur[1]]
            if data.shape[1] > num_samples:
                data = data[:, -num_samples:]

            x_data = list(range(data.shape[1]))
            for i, line in enumerate(emg_plots):
                y_data = data[i]
                line.set_data(x_data, y_data)

                axs[i].relim()
                axs[i].autoscale_view()
                if y_axes is not None and i < len(y_axes):
                    axs[i].set_ylim(y_axes[i])

            return emg_plots

        animation = FuncAnimation(
            figure, update, interval=100, blit=True, cache_frame_data=False
        )
        pyplot.show()


# %%
