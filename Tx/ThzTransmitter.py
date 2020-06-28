from serial import *
import numpy as np
import pickle
from time import sleep

# import numpy as np
# import pickle
# import pyqrcode
# from math import ceil
# import getpass
# from skimage.io import imshow
# from matplotlib import pyplot as plt
# from easygui import passwordbox

DEBUG = False


class ThzTransmitter:
    def __init__(self, com='/dev/tty.usbmodem14101'):
        # print('Thz Transmitter Running...\n')
        self._com = Serial(com, 115200, timeout=500)
        assert self._com_read() == 'Hello, Python'
        self._com_write('Hello, Arduino')
        # self._ch_num = ch_num
        # assert self._ch_num in (1, 8, 16)
        self.br = '50'

    def _com_read(self):
        raw_byte = self._com.readline()
        return raw_byte.decode().rstrip('\r\n')

    def _com_write(self, raw_str):
        self._com.write((raw_str + '\n').encode())

    def send_data(self, frame):
        self._com_write('Data')
        if self._com_read() == 'Ready':
            # print('Transmitting...')
            self._com_write(str(len(frame)))
            self._com_write(frame)
            if self._com_read() == 'Done':
                pass
                # print('Transmission finished\n')
            else:
                pass
        else:
            pass

    def send_calibration(self):
        self._com_write('Calibration')
        if self._com_read() == 'Ready':
            print('calibrating...')
            if self._com_read() == 'Done':
                pass
            else:
                pass
        else:
            pass

    def send_bitrate(self, br):
        self._com_write(br)
        if self._com_read() == 'Ready':
            print('set bit rate...')
            if self._com_read() == 'Done':
                pass
            else:
                pass
        else:
            pass


if __name__ == '__main__':
    thz_transmitter = ThzTransmitter()

    ch = 0
    dist = 1000

    frame_bits = 50
    n_frames = 200
    frame_header = (1, 1, 1, 0)
    # ber_file = 'ber_test.pkl'

    while True:
        msg = input('thz_com_%sHz >' % thz_transmitter.br)

        if msg == 'break':
            break
        elif msg == 'cal':
            thz_transmitter.send_calibration()
        elif msg == 'ber':
            ber_labels = []
            n = len(frame_header) + frame_bits
            for i in range(n_frames):
                frame = np.random.randint(2, size=n)
                frame[:len(frame_header)] = frame_header
                ber_labels.append(frame)
                frame_str = ''.join(map(str, frame.tolist()))
                thz_transmitter.send_data(frame_str)

            ber_labels = np.array(ber_labels)
            with open('../Rx/result/record/labels.pkl', 'wb') as f:
                pickle.dump(ber_labels, f)
        elif 'br' in msg:
            thz_transmitter.br = msg[2:]
            thz_transmitter.send_bitrate(msg)
        else:
            frame = ''.join(map(str, frame_header)) + msg
            thz_transmitter.send_data(frame)
