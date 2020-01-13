from Rx.NiAdc import NiAdc
from Rx.Synchronizer import Synchronizer
from Rx.Demodulator import Demodulator
import queue
import threading
import time

CONFIG = {
    'fs': 1e3,
}


class ThzReceiver:
    def __init__(self, fs):
        self.adc = NiAdc(
            sample_freq=fs)

        self.synchronizer = Synchronizer(
            sample_freq=fs)

        self.demodulator = Demodulator(
            header_queue=self.synchronizer.header_queue,
            sample_freq=fs)

        self.adc.dst_queue = self.synchronizer.src_queue
        self.synchronizer.dst_queue = self.demodulator.src_queue

        self.thread = threading.Thread(target=self.terminal)

    def terminal(self):
        while True:
            mode = input()
            if mode in ['data', 'cal']:
                self.synchronizer.mode_queue.put(mode)
                time.sleep(1)

    def start(self):
        self.adc.start()
        self.synchronizer.start()
        self.demodulator.start()
        self.thread.start()
        print('terminal starts')


if __name__ == '__main__':
    test_rx = ThzReceiver(**CONFIG)
    test_rx.start()
    while True:
        pass
