from PyDAQmx import *
import numpy as np
from ctypes import byref

DEBUG = False


class NiAdc(Task):
    def __init__(self, device='Dev2', ch='ai0', vmax=5.0, sample_freq=1E3, event_freq=1E2, dst_queue=None):
        Task.__init__(self)
        self._device = device
        self._ch = ch

        self._sample_freq = sample_freq

        self._event_freq = event_freq
        self._everyN = int(sample_freq / event_freq)
        self._adc_buf = np.zeros(self._everyN * 2)
        self._dst_queue = dst_queue

        self.CreateAIVoltageChan("%s/%s" % (device, ch), '', DAQmx_Val_RSE, -vmax, vmax, DAQmx_Val_Volts, None)
        self.CfgSampClkTiming('', self._sample_freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 0)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self._everyN, 0)

    def EveryNCallback(self):
        read = int32()
        self.ReadAnalogF64(DAQmx_Val_Auto, 0, DAQmx_Val_GroupByScanNumber, self._adc_buf, len(self._adc_buf),
                           byref(read),
                           None)
        rec_cnt = read.value
        self._dst_queue.extend(self._adc_buf[:rec_cnt].tolist())
        # if rec_cnt != self._everyN:
        # print(rec_cnt)
        return 0  # The function should return an integer


if __name__ == '__main__':
    pass
