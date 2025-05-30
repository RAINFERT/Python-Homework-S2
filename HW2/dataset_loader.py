import os
import random

class DatasetLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.patterns = {"backgrounds": [], "cells": []}
        self._load_patterns()

    def _load_patterns(self):
        for root, _, files in os.walk(self.dataset_dir):
            category = os.path.basename(root)
            if category in self.patterns:
                self.patterns[category] = sorted([os.path.join(root, f) for f in files])

        if not self.patterns["backgrounds"] or not self.patterns["cells"]:
            raise ValueError("Error: backgrounds or cells not found")

    def get_random_background(self):
        return random.choice(self.patterns["backgrounds"])

    def get_random_cell(self):
        return random.choice(self.patterns["cells"])