"""
2022.11.1
    demo:计算一个点到一个分布的距离
"""

import numpy as np
from scipy.spatial.distance import mahalanobis


def mahalanobis_distance(p, distr):
    # p: 一个点
    # distr : 一个分布
    # 计算分布的协方差矩阵
    cov = np.cov(distr, rowvar=False)
    # 选取分布中各维度均值所在点
    avg_distri = np.average(distr, axis=0)
    dis = mahalanobis(p, avg_distri, cov)
    return dis
