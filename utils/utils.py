from torchvision import transforms
import time

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
NORMALIZE_IMAGENETTE = transforms.Normalize(mean=[0.4618, 0.4571, 0.4288], std=[0.2531, 0.2472, 0.2564])
NORMALIZE_CIFAR10 = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

        return self

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time

    def stop_string(self):
        elapsed = self.stop()
        return f"Took {elapsed:0.2f}s"