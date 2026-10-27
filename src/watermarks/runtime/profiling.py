# src/watermarks/runtime/profiling.py
# -*- coding: utf-8 -*-

import time


def safe_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


class ProfilingLogitsProcessor:
    """
    对任意 logits processor 做透明包装：
    - 不改变行为
    - 只统计 __call__ 次数与耗时
    """
    def __init__(self, inner_processor, name="processor", enabled=False):
        self.inner = inner_processor
        self.name = name
        self.enabled = enabled
        self.call_count = 0
        self.total_time = 0.0

    def __call__(self, input_ids, scores):
        if not self.enabled:
            return self.inner(input_ids, scores)

        t0 = time.perf_counter()
        out = self.inner(input_ids, scores)
        dt = time.perf_counter() - t0

        self.call_count += 1
        self.total_time += dt
        return out

    def reset_profile(self):
        self.call_count = 0
        self.total_time = 0.0

    def get_profile(self):
        avg = self.total_time / self.call_count if self.call_count > 0 else 0.0
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": avg,
        }

    def __getattr__(self, item):
        return getattr(self.inner, item)