import time

import torch
from accelerate import Accelerator


class CudaMemoryMonitor:
    base = 1024 ** 3  # GB

    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator
        self.max_memory_allocated = 0
        self.total_memory_allocated = 0
        self.step_count = 0

    def start_epoch(self):
        self.max_memory_allocated = 0
        self.total_memory_allocated = 0
        self.step_count = 0

        # reset
        if self.accelerator.is_main_process:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

    def record_step(self):
        step_memory = 0

        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / self.base
            step_memory += memory_allocated

            self.max_memory_allocated = max(
                torch.cuda.max_memory_allocated(i) / self.base,
                self.max_memory_allocated,
            )
        step_memory /= torch.cuda.device_count()

        self.total_memory_allocated += step_memory
        self.step_count += 1

        return step_memory

    def get_stats(self):
        if self.step_count == 0:
            return 0, 0

        avg_memory = self.total_memory_allocated / self.step_count
        return self.max_memory_allocated, avg_memory

    def monitor_memory(self):
        def decorator(train_step_func):
            def wrapper(*args, **kwargs):
                result = train_step_func(*args, **kwargs)
                self.record_step()
                return result

            return wrapper

        return decorator


class TimeMonitor:
    def __init__(self):
        self.step_count = 0
        self.total_step_time = 0
        self.max_step_time = 0
        self.prev_time = None

    def start_epoch(self):
        self.step_count = 0
        self.total_step_time = 0
        self.max_step_time = 0
        self.prev_time = None

    def start_step(self):
        self.prev_time = time.time()
        self.step_count += 1

    def end_step(self):
        now = time.time()
        current_time = now - self.prev_time

        self.max_step_time = max(
            current_time,
            self.max_step_time,
        )
        self.total_step_time += current_time

        self.prev_time = now

    def get_state(self):
        if self.step_count == 0:
            return 0, 0

        avg_step_time = self.total_step_time / self.step_count
        return self.max_step_time, avg_step_time
