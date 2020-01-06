from Rx.NiAdc import NiAdc
from Rx.Synchronizer import Synchronizer
from Rx.Demodulator import Demodulator
import queue

CONFIG = {
    'fs': 1e3,
}


class ThzReceiver:
    def __init__(self, fs):
        self.demodulator = Demodulator(sample_freq=fs)

        self.synchronizer = Synchronizer(
            dst_queue=self.demodulator.src_queue,
            sample_freq=fs)

        self.adc = NiAdc(
            dst_queue=self.synchronizer.src_queue,
            sample_freq=fs)

    def start(self):
        self.adc.start()
        self.synchronizer.start()
        self.demodulator.start()




if __name__ == '__main__':
    test_rx = ThzReceiver(**CONFIG)
    test_rx.start()
    while True:
        pass
