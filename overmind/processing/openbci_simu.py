# -*- coding: utf-8 -*-
""" Simulates data being sent from OpenBCI V3 board. Useful for applications testing.
The output data is always 0 as if the daisy module was not connected. """
import time
import timeit
import numpy as np

# SAMPLE_RATE = 250.0 #Hz

class OpenBCIBoard(object):
    """ Handle a connection to an OpenBCI board.
    Args:   port: The port to connect to
            baud: The baud of the serial connection
            daisy: Enable or disable daisy module and 16 chans readings """

    def __init__(self, port=None, baud=115200, data=[], filter_data=True, scaled_output=True, daisy=False, sample_rate=250.0, log=True, timeout=None):
        print("Connecting to Acquisition Simulator")
        # wait for device to be ready
        self.streaming = False
        self.scaling_output = scaled_output
        self.t = 0
        self.playback_data = data
        self.daisy = daisy
        self.sample_rate = sample_rate/2 if self.daisy else sample_rate # self.getSampleRate()
        # print(self.sample_rate)
        self.packet_id = 0

    def start_stream(self, callback, lapse=-1):
        """ Start handling streaming data from the board.
        Call a provided callback for every single sample that is processed (every two samples with daisy module).
        Args: A callback function -- or a list of functions -- that will receive a single argument of the OpenBCISample object captured."""

        if not self.streaming: self.streaming = True
        start_time = timeit.default_timer()
        # Enclose callback function in a list if it comes alone
        if not isinstance(callback, list): callback = [callback]
        
        self.sample_counter = 0
        while self.streaming:
            st = time.time()
            
            # read current sample
            if self.daisy: self.packet_id = (self.packet_id + 2) % (self.sample_rate + 6) # 256 (atraso delay)
            else: self.packet_id = (self.packet_id + 1) % (self.sample_rate + 6)
            
            channels_data = self.playback_data[self.sample_counter, :].tolist()

            sample = OpenBCISample(self.packet_id, channels_data, [])
            
            # if a daisy module is attached, wait to concatenate two samples (main board + daisy) before passing it to callback
            for call in callback: call(sample) # executa a função GetData de SampleManager
            
            if (lapse > 0) and ((timeit.default_timer() - start_time) > lapse): self.stop()
            self.sample_counter += 1

            # if sample_counter == counter_max: sample_counter = 0
            while 1.0 / self.sample_rate > time.time() - st:
                pass
                #print(1.0 / self.sample_rate, time.time() - st)
            # time.sleep(1.0 / self.sample_rate)

    """Clean Up (atexit)"""

    def stop(self):
        print("Stopping streaming...\nWait for buffer to flush...")
        self.streaming = False

    def disconnect(self):
        print("Closing Serial...")

    """SETTINGS AND HELPERS"""
    def warn(self, text):
        print("Warning: %s" % text)

    # def getSampleRate(self):
    #     if self.daisy: return SAMPLE_RATE / 2.0
    #     else: return SAMPLE_RATE


    # def start_streaming(self, callback, lapse=-1):
    #     """ Start handling streaming data from the board. Call a provided callback
    #     for every single sample that is processed (every two samples with daisy module).
    #     Args: A callback function -- or a list of functions -- that will receive a single argument of the OpenBCISample object captured."""
    #     if not self.streaming: self.streaming = True
    #     start_time = timeit.default_timer()
    #     # Enclose callback function in a list if it comes alone
    #     if not isinstance(callback, list): callback = [callback]
    #
    #     self.sample_counter = 0
    #     while self.streaming:
    #         st = time.time()
    #         # read current sample
    #         if self.daisy: self.packet_id = (self.packet_id + 2) % 256
    #         else: self.packet_id = (self.packet_id + 1) % 256
    #         channels_data = self.playback_data[self.sample_counter, :].tolist()
    #         # print self.playback_data.shape
    #         sample = OpenBCISample(self.packet_id, channels_data, [])
    #         # if a daisy module is attached, wait to concatenate two samples (main board + daisy) before passing it to callback
    #         for call in callback: call(sample)
    #         if (lapse > 0) and ((timeit.default_timer() - start_time) > lapse): self.stop()
    #         self.sample_counter += 1
    #         # if sample_counter == counter_max: sample_counter = 0
    #         while 1.0 / self.sample_rate > time.time() - st:
    #             pass
    #         # time.sleep(1.0)
    #
    # """Clean Up (atexit)"""

class OpenBCISample(object):
    """Object encapulsating a single sample from the OpenBCI board."""
    def __init__(self, packet_id, channels_data, aux_data):
        self.id = packet_id
        self.channels_data = channels_data
        self.aux_data = aux_data