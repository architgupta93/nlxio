import numpy as np
from scipy import signal
from .io import loadNtt

# Graphics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class NTT(object):

    """Class structure for storing tetrode data
       TODO: Import MotionAnimation package from
       https://github.com/architgupta93/MotionAnimation and use it as a
       submodule here - It contains most of the basic functionality that we
       might need here.    
    """

    _SPIKE_THRESHOLD_H = 80.0
    _SPIKE_THRESHOLD_L = 20.0

    def __init__(self, filename=None ):
        """ Class constructor for NTT class
            Takes in a file which has the data (NTT format) representing the
            spike events and creates the class instance

            Class members:
        """

        self.n_spikes            = -1
        self.n_samples_per_spike = -1
        self.n_channels          = -1
        self.t_pts               = np.array([], dtype=int)
        self.spikes              = np.array([], dtype=float)

        if filename is not None:
            # TODO: Add a try/catch block around this
            ts, sp, fs = loadNtt(filename)

            # Initially n_spikes serves as the total number of data records
            # obtained. Later, invalid spikes are removed for better
            # clustering.
            (self.n_spikes, self.n_samples_per_spike, self.n_channels) = np.shape(sp)
            invalid_spike_list = []

            # Allocate an empty array for the spikes
            self.t_pts = ts

            # We do <n_samples, n_channels> because this seems more amicable to scikit's tools
            self.spikes = np.empty((self.n_spikes, self.n_channels))

            for wvf_index, spike_wvf_set in enumerate(sp):
                # Each of these is a n_samples_per_spike x n_channel array
                spike_values, invalid_spike = self._thresholdSpike(sp[wvf_index])

                if invalid_spike:
                    # DEBUG message, fix later.
                    invalid_spike_list.append(invalid_spike)
                else:
                    self.spikes[wvf_index] = spike_values

            # Clean up the invalid spikes
            self.spikes = np.delete(self.spikes, invalid_spike_list, axis=0)

            # DEBUG: Finding the range of values for a typical spike
            print('Total spikes: %d, valid: %d'% (self.n_spikes, self.n_spikes-len(invalid_spike_list)))
            print('Max: %f,'% np.max(self.spikes))
            print('Min: %f,'% np.min(self.spikes))
            print('Median: %f.'% np.median(self.spikes))

            self.n_spikes = len(self.spikes)

            return

        print("NTT filename not specified. Instantiating empty class object.")
        return

    def _thresholdSpike(self, spike_data):
        """Protected function for converting a spike waveform into a scalar spike
           Details TBD

        :spike_data: Raw spike waveform data that is directly read from the NTT
            file
        :invalid_spike: In case the data looks somewhat absurd, we should ignore
            it. The criteria for that is decided by the SPIKE_THRESHOLD in the
            class. If the peak value exceeds this value, it should be ignored
        :returns: TODO

        """

        # Debug condition
        dbg_cond = (1 == 0)

        """
        Method 01: Finding the correlation between the spike waveforms and
        looking at the peak value/index. This should be a good proxy for
        propagation delay.
        """

        """
        # Normalize the data, clear the mean and make the area 1
        spike_data[:,0] = spike_data[:,0] - np.mean(spike_data[:,0])
        spike_data[:,1] = spike_data[:,1] - np.mean(spike_data[:,1])
        spike_data[:,2] = spike_data[:,2] - np.mean(spike_data[:,2])
        spike_data[:,3] = spike_data[:,3] - np.mean(spike_data[:,3])

        spike_data[:,0] = spike_data[:,0]/np.linalg.norm(spike_data[:,0])
        spike_data[:,1] = spike_data[:,1]/np.linalg.norm(spike_data[:,1])
        spike_data[:,2] = spike_data[:,2]/np.linalg.norm(spike_data[:,2])
        spike_data[:,3] = spike_data[:,3]/np.linalg.norm(spike_data[:,3])

        # Get the cross-correlation
        xcorr_s0s1 = signal.correlate(spike_data[:,0], spike_data[:,1])
        xcorr_s0s2 = signal.correlate(spike_data[:,0], spike_data[:,2])
        xcorr_s0s3 = signal.correlate(spike_data[:,0], spike_data[:,3])

        corr_index = [np.linalg.norm(xcorr_s0s1), np.linalg.norm(xcorr_s0s2), np.linalg.norm(xcorr_s0s3)]

        if dbg_cond:
            ax_spikes = plt.subplot(2, 1, 1)
            ax_spikes.plot(spike_data)

            ax_corr = plt.subplot(2, 1, 2)
            ax_corr.plot(xcorr_s0s1)
            ax_corr.plot(xcorr_s0s2)
            ax_corr.plot(xcorr_s0s3)

            plt.show()

        return corr_index
        """

        spike_peak = np.max(spike_data, 0)
        invalid_spike = (max(spike_peak) >= self._SPIKE_THRESHOLD_H) or (max(spike_peak) <= self._SPIKE_THRESHOLD_L)
        if dbg_cond and invalid_spike:
            plt.plot(spike_data)
            plt.show()

        return(spike_peak, invalid_spike)


    def visualize(self, plt_axes=[0, 1, 2]):
        """ Visualize the data as a 3D scatter plot

        :plt_axes: The indices [x, y, z] to be selected for plotting
        :returns: TODO

        """
         
        if (len(plt_axes) != 3):
            print("Need 3 axes dimensions for a scatter plot!")
            return

        # TODO: If only 2 values are supplied, we could just do a 2D scatter plot
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter3D(self.spikes[:,plt_axes[0]], self.spikes[:,plt_axes[1]], self.spikes[:,plt_axes[2]]);
        plt.show()
