import time
from datetime import datetime


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def reset(self):
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0


def time_for_filename():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
