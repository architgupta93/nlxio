import numpy as np
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

            (self.n_spikes, self.n_samples_per_spike, self.n_channels) = np.shape(sp)

            # Allocate an empty array for the spikes
            self.t_pts = ts

            # We do <n_samples, n_channels> because this seems more amicable to scikit's tools
            self.spikes = np.empty((self.n_spikes, self.n_channels))

            for wvf_index, spike_wvf_set in enumerate(sp):
                # Each of these is a n_samples_per_spike x n_channel array
                for channel in range(self.n_channels):
                    self.spikes[wvf_index, channel] = self._thresholdSpike(sp[wvf_index,:,channel])
            return

        print("NTT filename not specified. Instantiating empty class object.")
        return

    def _thresholdSpike(self, spike_data):
        """Protected function for converting a spike waveform into a scalar spike
           Details TBD

        :spike_data: Raw spike waveform data that is directly read from the NTT file
        :returns: TODO

        """
        
        # DEBUG
        if 1 is None:
            plt.plot(spike_data)
            plt.show()

        # TODO: See if there is a faster way of doing this! Right now, this takes a lot of time!
        peak_sample_index, peak_sample_value = max(enumerate(spike_data), key=lambda v: v[1])
    
        # TODO: We can make use of the  sample index as well! Both of them have
        # a very similar meaning though
        # return peak_sample_value / np.log(2+peak_sample_index)
        # return np.log(1e-10 + abs(peak_sample_value))
        return peak_sample_index

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
