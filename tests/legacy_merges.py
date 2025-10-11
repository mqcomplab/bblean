# type: ignore
import numpy as np


def jt_isim(c_total, n_objects):
    sum_kq = np.sum(c_total)
    sum_kqsq = np.dot(c_total, c_total)
    a = (sum_kqsq - sum_kq) / 2

    return a / (a + n_objects * sum_kq - sum_kqsq)


def merge_tolerance(
    threshold,
    new_ls,
    new_centroid,
    new_n,
    old_ls,
    nom_ls,
    old_n,
    nom_n,
    tolerance=0.05,
):
    jt_radius = jt_isim(new_ls, new_n)
    if jt_radius < threshold:
        return False
    else:
        if old_n == 1 and nom_n == 1:
            return True
        elif nom_n == 1:
            return (
                jt_isim(old_ls + nom_ls, old_n + 1) * (old_n + 1)
                - jt_isim(old_ls, old_n) * (old_n - 1)
            ) / 2 >= jt_isim(old_ls, old_n) - tolerance and (jt_radius >= threshold)
        else:
            return True


def merge_radius(
    threshold,
    new_ls,
    new_centroid,
    new_n,
    old_ls,
    nom_ls,
    old_n,
    nom_n,
):
    jt_sim = jt_isim(new_ls + new_centroid, new_n + 1) * (new_n + 1) - jt_isim(
        new_ls, new_n
    ) * (new_n - 1)
    return jt_sim >= threshold * 2


def merge_diameter(
    threshold,
    new_ls,
    new_centroid,
    new_n,
    old_ls,
    nom_ls,
    old_n,
    nom_n,
):
    jt_radius = jt_isim(new_ls, new_n)
    return jt_radius >= threshold
