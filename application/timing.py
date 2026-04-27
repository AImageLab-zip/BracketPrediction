from functools import wraps
import time
from collections import defaultdict


class TimingRegistry:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, name, duration):
        self.data[name].append(duration)

    def report(self, clear=True):
        print("=======================")
        for name, values in self.data.items():
            total = sum(values)
            count = len(values)
            avg = total / count
            print(f"⏳ {name}: called {count} times, avg={avg:.6f}, total={total:.6f}")
        print("=======================")
        if clear: self.data.clear()


# single shared instance
timings = TimingRegistry()


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        timings.add(func.__name__, duration)
        return result
    return wrapper