# src/watermarks/algorithms/base.py
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class BaseMicroAlgorithm(ABC):
    uses_custom_candidate_generation = False

    def build_processor(self):
        raise NotImplementedError

    @abstractmethod
    def detect(self, text: str) -> dict:
        raise NotImplementedError

    def generate_candidates(self, current_ids, max_new_tokens: int, num_candidates: int):
        raise NotImplementedError

    def reset_profile(self):
        return None

    def get_profile(self) -> dict:
        return {}


class BaseMacroAlgorithm(ABC):
    needs_dual_calibration = True

    @abstractmethod
    def init_context(self, prompt: str):
        raise NotImplementedError

    @abstractmethod
    def select_best_candidate(self, candidates: list, context):
        raise NotImplementedError

    @abstractmethod
    def detect(self, text: str) -> dict:
        raise NotImplementedError