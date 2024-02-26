import numpy as np
from utils import get_feature, get_ldm_gpu
from sklearn.metrics.pairwise import cosine_distances
from scipy import stats
from tqdm import tqdm


def query_by_LDMS(model, X_pool, X_S, nQuery):
    ldm = get_ldm_gpu(model, X_pool, X_S)
    idx_ordered = np.argsort(ldm)
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    gamma = np.exp(-np.maximum(ldm - ldm_q, 0) / ldm_q)
    gamma[nQuery:] *= (nQuery / np.sum(gamma[nQuery:]))
    gamma[gamma > 1] = 1
    feature = get_feature(model, X_pool)
    feature = feature[idx_ordered]
    D_mat = cosine_distances(feature, feature)
    D_mat[D_mat < 1e-5] = 0

    # seeding
    idxs, D2 = [0], D_mat[0]
    pbar = tqdm(total=nQuery, initial=1)
    while len(idxs) < nQuery:
        if len(idxs) > 1: D2 = np.min([D2, D_mat[idx]], axis=0)
        px = gamma * D2
        Ddist = (px ** 2) / np.sum(px ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        idx = customDist.rvs(size=1)[0]
        while idx in idxs: idx = customDist.rvs(size=1)[0]
        idxs.append(idx)
        pbar.update(1)
    pbar.close()

    return idx_ordered[idxs]
