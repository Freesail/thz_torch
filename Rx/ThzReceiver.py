from Rx.NiAdc import NiAdc
from Rx.Synchronizer import Synchronizer
from Rx.Demodulator import Demodulator
import queue
import threading
import time
from Rx.VirtualOsc import VirtualOsc

CONFIG = {
    'fs': 1e3,
    'channel_id': 'single',
    'channel_range': 1000,
    'bit_rate': 50,
    'frame_header': (1, 1, 1, 0),
    'frame_bits': 50,
}


class ThzReceiver:
    def __init__(self, fs, bit_rate, frame_header, frame_bits, channel_id, channel_range):
        self.adc = NiAdc(
            sample_freq=fs, vmax=5.0)

        self.synchronizer = Synchronizer(
            sample_freq=fs,
            bit_rate=bit_rate,
            frame_header=frame_header,
            frame_bits=frame_bits
        )

        self.demodulator = Demodulator(
            header_queue=self.synchronizer.header_queue,
            sample_freq=fs,
            bit_rate=bit_rate,
            frame_header=frame_header,
            frame_bits=frame_bits,
            channel_id=channel_id,
            channel_range=channel_range
        )

        self.adc.dst_queue = self.synchronizer.src_queue
        self.synchronizer.dst_queue = self.demodulator.src_queue

        self.thread = threading.Thread(target=self.console)

    def console(self):
        while True:
            mode = input()
            if mode in ['data', 'cal', 'tx_cal', 'record']:
                self.synchronizer.mode_queue.put(mode)
                time.sleep(1)

    def start(self):
        self.adc.start()
        self.synchronizer.start()
        self.demodulator.start()
        self.thread.start()
        print('Console: starts')


if __name__ == '__main__':
    test_rx = ThzReceiver(**CONFIG)
    test_rx.start()
    while True:
        pass
