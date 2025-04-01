import numpy as np

class DetectionParameters:
    def __init__(self,
                 min_SNR_p: float,
                 min_SNR_s: float,
                 min_stations_p: int,
                 min_stations_s: int,
                 f_p: float=None,
                 f_s: float=None,
                 f_p_corner: float=None,
                 f_s_corner: float=None,
                 free_surface: bool=False):

        self.min_SNR_p = min_SNR_p
        self.min_SNR_s = min_SNR_s
        self.min_stations_p = min_stations_p
        self.min_stations_s = min_stations_s
        self.f_p = f_p
        self.f_s = f_s
        self.f_p_corner = f_p_corner
        self.f_s_corner = f_s_corner
        self.free_surface = free_surface

    def validate(self):

        if (self.min_SNR_p is None or self.min_SNR_s is None):
            raise ValueError("Minimum S/N ratio can not be None")

        if (self.min_stations_p is None and self.min_stations_s is None):
            raise ValueError("Minimum number of stations can not be None")

        if not (self.min_SNR_p > 0 and self.min_SNR_s > 0):
            raise ValueError("Minimum S/N ratios must be positive")

        if not (self.min_stations_p > 0 and self.min_stations_s > 0):
            raise ValueError("Minimum number of stations must be positive")

        if self.f_p is not None:
            if self.f_p <= 0:
                raise ValueError("P-wave frequency must be positive")

        if self.f_s is not None:
            if self.f_s <= 0:
                raise ValueError("S-wave frequency must be positive")

        if self.f_p_corner is not None:
            if self.f_p_corner <= 0:
                raise ValueError("P-wave corner frequency must be positive")
        else:
            self.f_p_corner = 100

        if self.f_s_corner is not None:
            if self.f_s_corner <= 0:
                raise ValueError("S-wave corner frequency must be positive")
        else:
            self.f_s_corner = 100
