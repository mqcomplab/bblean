# type: ignore
import numpy as np


def calculate_comp_sim(data):
    """Returns vector of complementary similarities"""
    n_objects = len(data) - 1
    c_total = np.sum(data, axis=0)
    comp_matrix = c_total - data
    a = comp_matrix * (comp_matrix - 1) / 2
    comp_sims = np.sum(a, axis=1) / np.sum(
        (a + comp_matrix * (n_objects - comp_matrix)), axis=1
    )
    return comp_sims


def calculate_medoid(data):
    """Returns index of medoid"""
    return data[np.argmin(calculate_comp_sim(data))]


def calculate_medoid_idx(data):
    """Returns index of medoid"""
    return np.argmin(calculate_comp_sim(data))
