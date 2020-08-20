import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime
from collections import deque
from Rx.NiAdc import NiAdc

tableau20 = np.array([(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                      (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                      (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                      (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                      (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)], dtype=np.float32) / 255.0


class VirtualOsc:
    # xmax is ms, sample_freq in Hz
    def __init__(self, sample_freq=1E3, xmax=2000, ymax=1.0):
        self._fs = sample_freq
        self._size = int(xmax / 1000 * sample_freq)
        self.src_queue = deque([0.0] * self._size, maxlen=self._size)
        self._deltaT = 1000 / sample_freq

        self._xmax = xmax
        self._ymax = ymax

        self._xmajor_loc = xmax / 10
        self._xminor_loc = self._xmajor_loc / 5
        self._ymajor_loc = ymax / 10
        self._yminor_loc = self._ymajor_loc / 5

        self._fig, self._ax = plt.subplots()
        self._line, = self._ax.plot([], [], lw=0.5, color=tableau20[0])
        self._ax.grid()
        self._ani = animation.FuncAnimation(self._fig, self._run, None, blit=True, interval=10,
                                            repeat=False, init_func=self._init)

        self._fig.canvas.mpl_connect('button_press_event', self._onClick)
        self._pause = False

    def _onClick(self, event):
        if self._pause:
            self._ani.event_source.stop()
        else:
            self._ani.event_source.start()
        self._pause ^= True

    def _init(self):
        self._ax.set_ylim(-self._ymax, self._ymax)
        self._ax.set_xlim(0, self._xmax)
        self._ax.set_xlabel('Time (ms)')
        self._ax.set_ylabel('Voltage (V)')
        self._ax.xaxis.set_major_locator(ticker.MultipleLocator(self._xmajor_loc))
        self._ax.xaxis.set_minor_locator(ticker.MultipleLocator(self._xminor_loc))
        self._ax.yaxis.set_major_locator(ticker.MultipleLocator(self._ymajor_loc))
        self._ax.yaxis.set_minor_locator(ticker.MultipleLocator(self._yminor_loc))

        self._line.set_data(np.arange(0, self._size) * self._deltaT, list(self.src_queue))
        return self._line,

    def _run(self, data):
        self._line.set_data(np.arange(0, self._size) * self._deltaT, list(self.src_queue))
        return self._line,


if __name__ == '__main__':
    test_osc = VirtualOsc()
    test_adc = NiAdc(vmax=2.0)
    test_adc.dst_queue = test_osc.src_queue
    test_adc.StartTask()
    plt.show()
    test_adc.StopTask()
    test_adc.ClearTask()
