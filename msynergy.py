import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import preprocess as pp
from sklearn.decomposition import NMF

class MuscleSynergy:
    def __init__(self, max_n_components):
        self.max_n_components = max_n_components


