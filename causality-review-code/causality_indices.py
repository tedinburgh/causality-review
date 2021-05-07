##############################################################
##
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.linalg import norm
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
##
## Notes:
##
## Only the first transfer entropy and the similarity indices are independent
## of scaling/normalisation of either input, unclear how this affects them
##
## In all of the functions, x and y must be vectors of the same length
## mx and my are embedding vector length
##
##############################################################
##
def embedding(x, y, mx, my, h = 1):
    ## Embedding in state space: the first mx columns of pts are the
    ## x-embeddingthe next my columns are the y-embedding and the final columns
    ## are the horizon values
    m = max(mx, my)
    n = len(x)
    pts = np.zeros((n - m - h + 1, mx + my + 2))
    pts[:, mx + my] = x[m + h - 1:]
    pts[:, mx + my + 1] = y[m + h - 1:]
    for ii in range(1, mx + 1):
        pts[:, mx - ii] = x[(m - ii):(n - ii - h + 1)]
    for ii in range(1, my + 1):
        pts[:, mx + my - ii] = y[(m - ii):(n - ii - h + 1)]
    ##
    ## As there are potentially NaNs in the data, we need to remove these
    ## embeddings. However, if we had done this earlier, we might have removed
    ## valid points in the embedding space (e.g. x = [0, 1, 2, 3, nan, 4, 5, 6]
    ## should have embedding vectors [0, 1, 2], [1, 2, 3], [4, 5, 6] but an
    ## earlier removal of [2, 3, nan] from x might only leave the embedding
    ## vector [4, 5, 6] - this is a problem in other code that does not
    ## consider NaNs)
    pts = pts[~np.any(np.isnan(pts), axis = 1), :]
    if h <= 0:
        pts = pts[:, :(mx + my)]
    return pts
##
##############################################################
##############################################################
##############################################################
## INFORMATION THEORETIC MEASURES
##
##############################################################
##############################################################
## TRANSFER ENTROPY
##
##############################################################
## Transfer entropy using histogram
##
def transfer_entropy(x, y, mx = 1, my = 1, r = 8, units = 'nats', \
        effective = True, shuffle = 10, seed = 2212):
    ##
    ## Transfer entropy (Schreiber 2000) is an information-theoretic approach
    ## which measures deviation of the time series from the generalised Markov
    ## property.
    if units == 'nats':
        log = np.log
    elif units == 'bits':
        log = np.log2
    else:
        raise ValueError('Units must be "bits" or "nats"')
    ##
    ## Create the embedding array
    pts = embedding(x, y, mx = mx, my = my, h = 1)
    n = pts.shape[0]
    ##
    def te_hist_xy(pts_xy, x_cols, y_cols, hx_cols = [-1], r = None):
        p_hxy = np.histogramdd(pts_xy, bins = r)[0] / pts_xy.shape[0]
        mask = p_hxy != 0
        def tile_reps(cols, pts = pts_xy):
            reps = np.ones((pts.shape[1], ), dtype = 'int')
            reps[cols] = r
            return reps
        def sum_tile(p, cols):
            t = tile_reps(cols)
            return np.tile(p.sum(axis = tuple(cols)).reshape(r + 1 - t), t)
        p_hx = (mask * sum_tile(p_hxy, y_cols))[mask]
        p_x = (mask * sum_tile(p_hxy, y_cols + hx_cols))[mask]
        p_xy = (mask * sum_tile(p_hxy, hx_cols))[mask]
        p_hxy = p_hxy[mask]
        te = np.sum(p_hxy * log((p_hxy * p_x) / (p_hx * p_xy)))
        return te
    ##
    pts_yx = pts[:, list(range(mx + my)) + [mx + my]]
    pts_xy = pts[:, list(range(mx + my)) + [mx + my + 1]]
    te_yx = te_hist_xy(pts_yx, list(range(mx)), list(range(mx, mx + my)), r = r)
    te_xy = te_hist_xy(pts_xy, list(range(mx, mx + my)), list(range(mx)), r = r)
    ##
    ##
    ## Effective transfer entropy (Marchinski and Kantz, 2002) tries to remedy
    ## small sample effect bias, by shuffling y to destroy time series
    ## dependencies in y and statistical dependencies between x and y. The
    ## transfer entropy computed on x and shuffled y converges to zero as the
    ## sample size increases and is an estimator for the bias induced by small
    ## sample effects. Repeating the shuffling and averaging the result
    ## provides a consistent estimator.
    if effective:
        ete_xy = 0
        ete_yx = 0
        np.random.seed(seed)
        for s in range(shuffle):
            ind = np.random.choice(range(len(x)), size = len(x))
            ## We need to fully recreate pts, since there seem to be potential
            ## problems with NaNs if we block shuffle the x components of the
            ## previous pts
            pts_xy_shuffled = embedding(x[ind], y, mx = mx, my = my, h = 1)
            pts_xy_shuffled = \
                pts_xy_shuffled[:, list(range(mx + my)) + [mx + my + 1]]
            pts_yx_shuffled = embedding(x, y[ind], mx = mx, my = my, h = 1)
            pts_yx_shuffled = \
                pts_yx_shuffled[:, list(range(mx + my)) + [mx + my]]
            ## Any single shuffled estimator returning no valid embedding vectors
            ## and we set ete to NaN completely
            if pts_yx_shuffled.shape[0] == 0:
                ete_yx = np.nan
            else:
                n = pts_yx_shuffled.shape[0]
                ete_yx -= te_hist_xy(pts_yx_shuffled, list(range(mx)), \
                    list(range(mx, mx + my)), r = r) / shuffle
            ##
            if pts_xy_shuffled.shape[0] == 0:
                ete_xy = np.nan
            else:
                n = pts_xy_shuffled.shape[0]
                ete_xy -= te_hist_xy(pts_xy_shuffled, \
                    list(range(mx, mx + my)), list(range(mx)), r = r) / shuffle
        ## After averaging the shuffled, subtract from the values from the
        ## already computed transfer entropies
        ete_xy += te_xy
        ete_yx += te_yx
        ## te_xy is the transfer entropy of x given y e.g. TE(X|Y)
        return te_xy, te_yx, ete_xy, ete_yx
        ##
    else:
        ## te_xy is the transfer entropy of x given y e.g. TE(X|Y)
        return te_xy, te_yx, np.nan, np.nan
    ## end function for transfer entropy (histogram)
##
##############################################################
## Mutual information using Kraskov-Sollbauer-Grassberger
##
def mi_ksg(x, y, mx = 1, my = 1, k = 4, algorithm = 1, metric = 'chebyshev'):
    ##
    pts = embedding(x, y, mx = mx, my = my, h = 0)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    xyTree = KDTree(pts, metric = metric)
    ##
    dists, _ = xyTree.query(pts, k = k + 1)
    dists = dists[:, k] - 1e-16
    ## Note the inequality is strict here! Hence the (- 1e-16) abovw
    n_x = KDTree(pts[:, x_cols], metric = metric).query_radius( \
        pts[:, x_cols], dists, count_only = True) - 1
    n_y = KDTree(pts[:, y_cols], metric = metric).query_radius( \
        pts[:, y_cols], dists, count_only = True) - 1
    ##
    ## Mutual information via Krakov estimator (2004) is given in Gomez-Herrero
    ## et al (2015) as:
    ## MI_{Y->X} = digamma(k) + digamma(n) - (digamma(n_x) + digamma(n_y)) / n
    vfunc = np.vectorize(lambda z: digamma(max(z, 1)))
    mi = digamma(k) + digamma(n) - (vfunc(n_x + 1) + vfunc(n_y + 1)).mean()
    ##
    return mi
    ## end function for mutual information (KSG)
##
##############################################################
## Transfer entropy using Kraskov-Sollbauer-Grassberger
##
def transfer_entropy_ksg(x, y, mx = 1, my = 1, k = 4, effective = False, \
        shuffle = 10, seed = 2212, algorithm = 1, metric = 'chebyshev'):
    ## NOTE: this doesn't seem to work currently for mx > 1, my > 1
    ##
    ## This function also computes the transfer entropy, but using Kraskov (KSG)
    ## mutual information estimates (Zhu et al 2015).
    ##
    pts = embedding(x, y, mx = mx, my = my, h = 1)
    n = pts.shape[0]
    ##
    if pts.shape[0] < k + 1:
        return np.nan, np.nan
    ##
    ## ksg_algorithm1 computes the 1st Kraskov based estimator for TE, which
    ## computes hypercubes in each subspace
    ## (cols defines the x subspace or y subspace, full subspace in either case
    ## is xy_cols)
    def ksg_xy_algorithm1(pts, x_cols, y_cols, hx_cols, k = k, metric = metric):
        ##
        xyTree1 = KDTree(pts[:, x_cols + y_cols + hx_cols], metric = metric)
        ##
        dists, _ = xyTree1.query(pts[:, x_cols + y_cols + hx_cols], k = k + 1)
        dists = dists[:, k] - 1e-16
        ## The (- 1e-16) is necessary here because the inequality in KSG is
        ## strict and the KDTree query is not
        ##
        ## For each point, compute the number of points within the hypercube
        ## centred at that point, in each subspace (minus the point itself)
        n_xh = KDTree(pts[:, x_cols + hx_cols], metric = metric).query_radius( \
            pts[:, x_cols + hx_cols], dists, count_only = True) - 1
        n_xy = KDTree(pts[:, x_cols + y_cols], metric = metric).query_radius( \
            pts[:, x_cols + y_cols], dists, count_only = True) - 1
        n_x = KDTree(pts[:, x_cols], metric = metric).query_radius( \
            pts[:, x_cols], dists, count_only = True) - 1
        ##
        ## The transfer entropy via Krakov estimator (2004) is given in
        ## Gomez-Herrero et al (2015) as:
        ## TE_{Y->X} = digamma(k) - \
        ##      (digamma(n_x1) + digamma(n_xy) - digamma(n_x)) / n
        vfunc = np.vectorize(lambda z: digamma(max(z, 1)))
        te = digamma(k) - \
            (vfunc(n_xh + 1) + vfunc(n_xy + 1) - vfunc(n_x + 1)).mean()
        return te, n_x, n_xh, n_xy
    ##
    ## ksg_algorithm2 computes the 2nd Kraskov based estimator for TE, which
    ## computes hyperrectangles in each subspace
    ## (cols defines the x subspace or y subspace, full subspace in either case
    ## is xy_cols)
    def ksg_xy_algorithm2(pts, x_cols, y_cols, hx_cols, k = k, metric = metric):
        ##
        xyTree1 = KDTree(pts[:, x_cols + y_cols + hx_cols], metric = metric)
        ##
        ## If each point is unique, then distances (hyperrectangle dimensions)
        ## are easily defined, if not then find all points that share the same
        ## distance in the full subspace and maximise the size of the
        ## hyperrectangle
        ## This is important if the data is discretised to some precision (d.p.)
        if np.unique(pts[:, x_cols + y_cols + hx_cols], axis = 1).shape[0] == n:
            _,idxs = xyTree1.query(pts[:, x_cols + y_cols + hx_cols], k = k + 1)
            idx = idxs[:, k]
            dists = np.abs(pts[:, np.newaxis, :] - pts[idxs])
        else:
            dists, _ = xyTree1.query( \
                pts[:, x_cols + y_cols + hx_cols], k = k + 1)
            idxs, dists_new = xyTree1.query_radius( \
                pts[:, x_cols + y_cols + hx_cols], dists[:, k], \
                return_distance = True)
            dists = np.amax([np.abs(pts[ii, np.newaxis, :] - \
                pts[idxs[ii][dists_new[ii] <= dists[ii, k]]]) \
                    for ii in range(n)], axis = 1)
        ##
        ## Need to find the number of points within hyperrectangular distances
        ## for each subspace, then find the set intersection of these to give
        ## e.g. n_xy, n_x1, n_x
        dists = np.amax(dists, axis = 1)
        idxs = np.zeros(mx + my + 2, dtype = object)
        for ii in range(mx + my + 2):
            if ii in xy_cols:
                idxs[ii] = KDTree( \
                    pts[:, ii].reshape(-1, 1), metric = metric).query_radius( \
                        pts[:, ii].reshape(-1, 1), dists[:, ii])
        def set_intersection(idxs, cols):
            count = np.zeros(len(idxs[0]), dtype = 'object')
            for jj in range(len(idxs[0])):
                count[jj] = set(idxs[cols[0]][jj])
                for ii in cols[1:]:
                    count[jj] = count[jj].intersection(idxs[ii][jj])
            return np.array([len(x) for x in count])
        n_xh = set_intersection(idxs, x_cols + hx_cols) - 1
        n_xy = set_intersection(idxs, x_cols + y_cols) - 1
        n_x = set_intersection(idxs, x_cols) - 1
        ##
        ## The transfer entropy via Krakov estimator (2004) is given in
        ## Gomez-Herror et al (2015) as:
        ## TE_{Y->X} = digamma(k) - 2 / k - 1 / n * /
        ##  (digamma(n_x1) - 1 / n_x1 + digamma(n_xy) - 1 / n_xy - digamma(n_x))
        vfunc = np.vectorize(lambda z: digamma(max(z, 1)))
        te = digamma(k) - 2 / k + (vfunc(n_x) - vfunc(n_xy) + \
            1 / n_xy - vfunc(n_xh) + 1 / n_x1).mean()
        return te, n_x, n_xh, n_xy
    ##
    if algorithm == 1:
        te_ksg_xy = ksg_xy_algorithm1
    elif algorithm == 2:
        te_ksg_xy = ksg_xy_algorithm2
    else:
        return
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ## te_xy is the transfer entropy of x given y e.g. TE(X|Y)
    te_xy, n_x, n_xh, n_xy = te_ksg_xy(pts, y_cols, x_cols, [mx + my + 1])
    te_yx, n_y, n_yh, n_yx = te_ksg_xy(pts, x_cols, y_cols, [mx + my])
    return te_xy, te_yx
    ## end function for transfer entropy (KSG)
##
##
##
##
##############################################################
##############################################################
## COARSE-GRAINED TRANSINFORMATION RATE
##
def coarse_grained_transinformation_rate(x, y, k = 4, tau_max = 15, \
        tau_threshold = None, metric = 'chebyshev'):
    ## If tau_max is not prespecified, it must be estimated from x and y
    ## tau_max is only loosely defined by Palus et al. [2001] as
    ## tau_max = argmax_tau (MI(x, x_tau) ~= 0)
    ##
    vfunc = np.vectorize(lambda z: digamma(max(z, 1)))
    def mi_ksg(x, y, k = k, metric = metric):
        ##
        pts = np.stack((x, y), axis = 1)
        ##
        pts = pts[~np.any(np.isnan(pts), axis = 1), :]
        n = pts.shape[0]
        if n < k + 1:
            return np.nan
        ##
        xyTree = KDTree(pts, metric = metric)
        ##
        dists, _ = xyTree.query(pts, k = k + 1)
        dists = dists[:, k] - 1e-16
        ## Note the inequality is strict here! Hence the - 1e-16 below
        n_x = KDTree(pts[:, 0].reshape(-1, 1), metric = metric).query_radius( \
            pts[:, 0].reshape(-1, 1), dists, count_only = True) - 1
        n_y = KDTree(pts[:, 1].reshape(-1, 1), metric = metric).query_radius( \
            pts[:, 1].reshape(-1, 1), dists, count_only = True) - 1
        ##
        ## Mutual information via Krakov estimator (2004) is given in
        ## Gomez-Herrero et al (2015) as MI_{Y->X}
        ## = digamma(k) + digamma(n) - (digamma(n_x) + digamma(n_y)) / n
        vfunc = np.vectorize(lambda z: digamma(max(z, 1)))
        mi = digamma(k) + digamma(n) - (vfunc(n_x + 1) + vfunc(n_y + 1)).mean()
        ##
        return mi
        ## end function for mutual information (KSG)
    def cmi_ksg(x, y, z, k = k, metric = metric):
        ##
        pts = np.stack((x, y, z), axis = 1)
        ##
        pts = pts[~np.any(np.isnan(pts), axis = 1), :]
        n = pts.shape[0]
        if n < k + 1:
            return np.nan
        ##
        xyzTree = KDTree(pts, metric = metric)
        ##
        dists, _ = xyzTree.query(pts, k = k + 1)
        dists = dists[:, k] - 1e-16
        ## Note the inequality is strict here! Hence the - 1e-16 below
        n_xz = KDTree(pts[:, [0, 2]], metric = metric).query_radius( \
            pts[:, [0, 2]], dists, count_only = True) - 1
        n_yz = KDTree(pts[:, [1, 2]], metric = metric).query_radius( \
            pts[:, [1, 2]], dists, count_only = True) - 1
        n_z = KDTree(pts[:, 2].reshape(-1, 1), metric = metric).query_radius( \
            pts[:, 2].reshape(-1, 1), dists, count_only = True) - 1
        ##
        vfunc = np.vectorize(lambda q: digamma(max(q, 1)))
        cmi = digamma(k) - \
            (vfunc(n_xz + 1) + vfunc(n_yz + 1) - vfunc(n_z + 1)).mean()
        ##
        return cmi
        ## end function for conditional mutual information (KSG)
    ##
    ## mutual coarse-grained information rate
    mcir_tau = lambda tau:(mi_ksg(x[:-tau], y[tau:]), mi_ksg(x[tau:], y[:-tau]))
    ## conditional coarse-grained information rate
    ccir_tau = lambda tau:(cmi_ksg(y[:-tau], x[tau:], x[:-tau]), \
        cmi_ksg(x[:-tau], y[tau:], y[:-tau]))
    ctir_xy = 0
    ctir_yx = 0
    ctir_vals = np.zeros((tau_max, 4))
    for ii in range(tau_max):
        tau_val = ii + 1
        if tau_threshold is not None:
            if mi_ksg(x[:-tau_val], x[tau_val:]) < tau_threshold or \
                    mi_ksg(y[:-tau_val], y[tau_val:]) < tau_threshold:
                break
        mcir_vals = mcir_tau(tau_val)
        ccir_vals = ccir_tau(tau_val)
        ctir_vals[ii, :2] = ccir_vals
        ctir_vals[ii, 2:] = mcir_vals
        ctir_yx += ccir_vals[0] - 0.5 * (mcir_vals[0] + mcir_vals[1])
        ctir_xy += ccir_vals[1] - 0.5 * (mcir_vals[0] + mcir_vals[1])
    ##
    ctir_xy = ctir_xy / tau_max
    ctir_yx = ctir_yx / tau_max
    return ctir_xy, ctir_yx
    ## end function for coarse-grained transinformation rate
##
##
##
##
##############################################################
##############################################################
##############################################################
## REGRESSION / PREDICTION
##
##############################################################
##############################################################
## EXTENDED GRANGER CAUSALITY
##
def extended_granger_causality(x, y, mx = 2, my = 2, L = 100, delta = 0.5, \
        min_k = 10, seed = 2212, metric = 'manhattan'):
    ##
    ## Extended Granger causality (Chen et al 2004) extends GC to non-linear
    ## multivariate time series, by dividing the embedding into a set of local
    ## neighbourhoods where a linear approximation should hold and ordinary
    ## least squares can be used for locally linear prediction. The initial set
    ## up is the same as in the previous index (and almost all others)
    ##
    pts = embedding(x, y, mx = mx, my = my, h = 1)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    if pts.shape[0] < min_k + 1:
        return np.nan, np.nan
    ##
    ## Using KDTree, find nearest min_k neighbours for each point
    xyTree = KDTree(pts[:, x_cols + y_cols], metric = metric)
    ## Only sample from points with a minimum of min_k neighbors in the cluster
    dists,_ = xyTree.query(pts[:, x_cols + y_cols], k = min_k + 1)
    if np.where(dists[:,-1] < delta)[0].size < L:
        return np.nan, np.nan
    ##
    ## Randomly select the L neighbourhood centers centers and find the indices
    ## of other points within each neighbourhood
    np.random.seed(seed)
    ind = np.random.choice(np.where(dists[:,-1] < delta)[0], size = L)
    idxs = xyTree.query_radius(pts[ind, :(mx + my)], delta)
    ##
    ## For each l in [1,...,L] we need to compute four variance terms for both
    ## EGC values, then average
    ## var has columns var_xy, var_x, var_yx, var_y
    var = np.zeros((L, 4))
    for ll in range(L):
        pts_ll = pts[idxs[ll]]
        ## By default, sm OLS doesn't include an intercept term! So add this in
        var[ll, 0] = sm.OLS(pts_ll[:, mx + my], \
            sm.add_constant(pts_ll[:, x_cols + y_cols])).fit().scale
        var[ll, 1] = sm.OLS(pts_ll[:, mx + my], \
            sm.add_constant(pts_ll[:, x_cols])).fit().scale
        var[ll, 2] = sm.OLS(pts_ll[:, mx + my + 1], \
            sm.add_constant(pts_ll[:, x_cols + y_cols])).fit().scale
        var[ll, 3] = sm.OLS(pts_ll[:, mx + my + 1], \
            sm.add_constant(pts_ll[:, y_cols])).fit().scale
    ## There is a problem when the variance of y is 0 and division by ~0
    ## for one local neighbourhood blows the estimate up, so avoid this by
    ## excluding these neighbourhoods
    ## Note this is more of a problem if the data is from some process with
    ## fixed precision e.g. to 1 d.p., where the data is essentially discrete
    ## A fix for this might be to randomise the data slightly
    var[var < 1e-15] = np.nan
    ##
    ## Average to give the index values, giving NaN if none of the variance
    ## terms worked.
    if np.all(np.isnan(var[:,2] / var[:,3])):
        egc_xy = np.nan
    else:
        egc_xy = 1 - np.nanmean(var[:,2] / var[:,3])
    ##
    if np.all(np.isnan(var[:,0] / var[:,1])):
        egc_yx = np.nan
    else:
        egc_yx = 1 - np.nanmean(var[:,0] / var[:,1])
    ##
    ## egc_xy is the transfer entropy of x given y e.g. EGC(X|Y)
    return egc_xy, egc_yx
    ## end function for extended granger causality
##
##
##
##
##############################################################
##############################################################
## NONLINEAR GRANGER CAUSALITY
##
## Fuzzy c-means clustering is used to find P centers for Gaussian radial basis
## functions in nonlinear granger causality (nlgc)
## Adapted from https://github.com/ITE-5th/fuzzy-clustering
class fuzzy_cmeans:
    def __init__(self, n_clusters = 10, centers = None, \
            max_iter = 500, m = 2, error = 1e-6, seed = 2212):
        super().__init__()
        self.n_clusters = n_clusters
        self.centers = centers
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.seed = seed
        self.u = None
        self.errors = np.zeros((0, 2))
        self.iter = 0
    ##
    def init_centers(self, x):
        n = x.shape[0]
        c = self.n_clusters
        centers = np.zeros((c, x.shape[1]))
        ii = 0
        np.random.seed(self.seed)
        centers[0, :] = x[np.random.choice(n, 1), :]
        p = float(2 / (self.m - 1))
        while ii < c - 1:
            ii += 1
            prob = cdist(x, centers[:(ii + 1), :]) ** p
            prob = prob.min(axis = 1)
            centers[ii, :] = x[np.random.choice(n, 1, p = prob / sum(prob)), :]
        return centers
    ##
    def fit(self, x):
        n = x.shape[0]
        c = self.n_clusters
        centers = self.centers
        if centers is None:
            centers = self.init_centers(x)
            ##
        u = self.update_u(x, centers)
        iter = 0
        while iter < self.max_iter:
            u_prev = u.copy()
            centers = self.update_centers(x, u)
            u = self.update_u(x, centers)
            iter += 1
            # Stopping rule
            self.iter = iter ##
            self.u = u ##
            self.centers = centers ##
            self.u_prev = u_prev ##
            self.errors = np.append(self.errors, [norm(u - u_prev), \
                norm((u - u_prev).flatten(), np.inf)]) ##
            if norm(u - u_prev) < self.error:
                break
        self.u = u
        self.centers = centers
        self.iter = iter
        return centers
    ##
    def update_centers(self, x, u):
        um = u ** self.m
        return (np.matmul(x.T, um) / np.sum(um, axis = 0)).transpose()
    ##
    def update_u(self, x, centers):
        p = float(2 / (self.m - 1))
        temp = cdist(x, centers) ** p
        sh = temp.shape[-1]
        denom = temp.reshape((x.shape[0], 1, -1)).repeat(sh, axis = 1)
        denom = temp[:, :, np.newaxis] / denom
        u = 1 / denom.sum(axis = 2)
        u[temp == 0] = 1
        return u
    ##
    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis = 0)
        u = self._predict(x, self.centers)
        return np.argmax(u, axis = -1)
##
##
##
##
## Non linear Granger causality
##
def nonlinear_granger_causality(x, y, mx, my, P = 50, sigma = 0.05, \
        seed = 2212, clustering = 'kmeans', decimal_threshold = 10, \
        fcm_max_iter = 500, fcm_error = 1e-6, min_P = None):
    ##
    ## This NLGC (Ancona et al 2004) is by far the most time intensive of
    ## the indices when using fuzzy c-means, so we recommend using k-means
    ## instead.
    ##
    pts = embedding(x, y, mx = mx, my = my, h = 1)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    ## The clustering fails if there are less unique points than there are
    ## clusters. This is a problem with some simulations of Henon maps,
    ## where there is synchronisation and/or periodic orbits of the attractor,
    ## and also in the case of rounding in the data.
    ## There were some instances where there were more than P unique points
    ## but the difference between some non-unique points was < 10e-15, so
    ## in reality there were < P clusters, hence a rounding to eliminate this
    if pts.shape[0] < P:
        return np.nan, np.nan
    if np.unique(pts[:, x_cols].round(decimals = decimal_threshold), \
            axis = 0).shape[0] <= P and min_P == True:
        return np.nan, np.nan
    if np.unique(pts[:, y_cols].round(decimals = decimal_threshold), \
            axis = 0).shape[0] <= P and min_P == True:
        return np.nan, np.nan
    ##
    ## Perform the fuzzy c-means or k-means clustering
    if clustering == 'cmeans':
        fcm_x = fuzzy_cmeans(n_clusters = P, max_iter = fcm_max_iter, \
            error = fcm_error, seed = seed)
        cx = fcm_x.fit(pts[:, x_cols])
        ##
        fcm_y = fuzzy_cmeans(n_clusters = P, max_iter = fcm_max_iter, \
            error = fcm_error, seed = seed)
        cy = fcm_y.fit(pts[:, y_cols])
    elif clustering == 'kmeans':
        ## This comes from scikit-cluster
        cx = KMeans(n_clusters = P, n_init = 10, \
            random_state = seed).fit(pts[:, x_cols]).cluster_centers_
        cy = KMeans(n_clusters = P, n_init = 10, \
            random_state = seed).fit(pts[:, y_cols]).cluster_centers_
    else:
        raise ValueError('Clustering must be "cmeans" or "kmeans"')
    ##
    ## RBF distance calculation
    def K(pts, centers, sigma):
        return np.exp(- cdist(pts, centers, 'sqeuclidean') / (2 * sigma ** 2))
    ##
    Phi_x = K(pts[:, x_cols], cx, sigma = sigma)
    Psi_y = K(pts[:, y_cols], cy, sigma = sigma)
    P_xy = np.hstack((Phi_x, Psi_y))
    ##
    ## Phi_x and Psi_y are used as predictors in a global nonlinear regression
    var_yx = sm.OLS(pts[:, mx + my + 1], sm.add_constant(P_xy)).fit().scale
    var_y = sm.OLS(pts[:, mx + my + 1], sm.add_constant(Psi_y)).fit().scale
    var_xy = sm.OLS(pts[:, mx + my], sm.add_constant(P_xy)).fit().scale
    var_x = sm.OLS(pts[:, mx + my], sm.add_constant(Phi_x)).fit().scale
    ##
    nlgc_xy = var_y - var_yx
    nlgc_yx = var_x - var_xy
    ##
    ## nlgc_xy is the transfer entropy of x given y e.g. NLGC(X|Y)
    return nlgc_xy, nlgc_yx
    ## end function for nonlinear granger causality
##
##
##
##
##############################################################
##############################################################
## PREDICTABILITY IMPROVEMENT
##
def predictability_improvement(x, y, mx = 2, my = 2, h = 1, R = 10, \
        metric = 'minkowski'):
    ##
    ## Predictability improvement (Feldmann and Bhattacharya 2004) is based on
    ## locally constant linear regression, a simpler version of GC measures.
    ## In place of x-current and y-current in pts, we potentially have a future
    ## value of of x and y using a prediction horizon. This is the only
    ## difference here from the other functions.
    ##
    pts = embedding(x, y, mx = mx, my = my, h = h)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    if pts.shape[0] < R + 1:
        return np.nan, np.nan
    ##
    ## We need three separate KDTrees for each subspace
    xyTree = KDTree(pts[:, x_cols + y_cols], metric = metric)
    xTree = KDTree(pts[:, x_cols], metric = metric)
    yTree = KDTree(pts[:, y_cols], metric = metric)
    ##
    ## There seems to be some inconsistencies if the distances between
    ## the point and its Rth neighbour is the same as the point and its
    ## (R + k)th neighbour for some k (in which case the query isn't always
    ## consistent on what the first R neighbours are - how does it choose
    ## between identical ones).
    ## Fix is to go for all points within the radius of the Rth neighbour
    dists_xy, _ = xyTree.query(pts[:, x_cols + y_cols], k = R + 1)
    dists_xy = dists_xy[:, R] + 1e-16
    idxs_xy = xyTree.query_radius(pts[:, x_cols + y_cols], dists_xy)
    ##
    dists_x, _ = xTree.query(pts[:, x_cols], k = R + 1)
    idxs_x = xTree.query_radius(pts[:, x_cols], dists_x[:, R] + 1e-16)
    ##
    dists_y, _ = yTree.query(pts[:, y_cols], k = R + 1)
    idxs_y = yTree.query_radius(pts[:, y_cols], dists_y[:, R] + 1e-16)
    ##
    ## This is the mean of the squares of the errors between actual horizon
    ## value and prediction based on horizon values of nearest neighbours
    def mse_fun(pts, col, idxs):
        ex = [np.mean(pts[ii, col] - pts[idxs[ii], col]) \
            for ii in range(len(idxs))]
        return np.square(ex).mean()
    ##
    mse_yx = mse_fun(pts, mx + my + 1, idxs_xy)
    mse_y = mse_fun(pts, mx + my + 1, idxs_y)
    mse_xy = mse_fun(pts, mx + my, idxs_xy)
    mse_x = mse_fun(pts, mx + my, idxs_x)
    ##
    pi_yx = mse_y - mse_yx
    pi_xy = mse_x - mse_xy
    ##
    ## Note: the output is reversed! To match the directionality of the other
    ## indices (which is reversed in the definition of the
    ## predictability improvement)
    ## pi_xy is the transfer entropy of x given y e.g. EGC(X|Y)
    return pi_yx, pi_xy
    ## end function for predictability improvement
##
##
##
##
##############################################################
##############################################################
##############################################################
## CROSS MAPPED
##
##############################################################
##############################################################
## SIMILARITY INDICES
##
def similarity_indices(x, y, mx = 2, my = 2, R1 = 10, R2 = 10, \
        N_max = None, metric = 'minkowski', seed = 2012):
    ##
    ## This computes two similar similarity indices S1 (Arhnold et al 1999) and
    ## S2 (Bhattacharya et al 2003) using R1 and R2 nearest neighbours
    ## respectively, for each point in the time series.
    ## These indices rely on the assumption that iterrelations between time
    ## series produce neighbourhood relations between the time series, that the
    ## indices of the nearest neighbours of one series match up with the indices
    ## of the nearest neighbours of the other when both systems are synchronised
    ##
    ## The initial set up is the same as in almost all other indices, but
    ## doesn't require x_{t + 1}, y_{t + 1}
    ##
    pts = embedding(x, y, mx = mx, my = my, h = 0)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    ## R1 and R2 are the number of nearest neighbours used for the indices
    ## S1 and S2 (this is important for computational efficiency)
    R = [np.min((R1, 2 * R2)), np.max((R1, 2 * R2)), np.max((R1, R2))]
    ##
    if pts.shape[0] < R[0] + 1:
        return np.nan, np.nan, np.nan, np.nan
    ##
    xTree = KDTree(pts[:, x_cols], metric = metric)
    yTree = KDTree(pts[:, y_cols], metric = metric)
    ##
    ## For the first index (Arnhold et al 1999), we need to calculate the
    ## squared euclidean distance between each point and all other points
    ## Computationally this is a bit of a pain.
    ## There are various options (KDTrees, sklearn's euclidean_distances,
    ## scipy pdist and squareform), but this vectorised approach using pdist
    ## appears to be the fastest, especially when n is large (~10 ** 5 points).
    ## There are also issues with memory if n is too large for some of these.
    ## For computational efficiency, we can randomly select a sample of N_max
    ## points and compute the distance metric between each point and this
    ## selection of points as an estimator for the dn_t_x.
    if N_max is None:
        met = 'sqeuclidean'
        dn_t_x = np.array([cdist(pts[ii, x_cols].reshape(1, -1), \
            pts[:, x_cols], metric = met).sum() for ii in range(n)]) / (n - 1)
        dn_t_y = np.array([cdist(pts[ii, y_cols].reshape(1, -1), \
            pts[:, y_cols], metric = met).sum() for ii in range(n)]) / (n - 1)
    else:
        np.random.seed(seed)
        ind = np.random.choice(n, N_max)
        met = 'sqeuclidean'
        dn_t_x = np.array([cdist(pts[ii, x_cols].reshape(1, -1), \
            pts[ind, :][:, x_cols], metric = met).mean() for ii in range(n)])
        dn_t_y = np.array([cdist(pts[ii, y_cols].reshape(1, -1), \
            pts[ind, :][:, y_cols], metric = met).mean() for ii in range(n)])
    ##
    ## S2 requires 2 * R2 nearest neighbours, so if there are fewer embedding
    ## vectors than this, the calculation is redundant, so split into two
    ## cases depending on if we need to calculate S2 or not (faster if not)
    ## We've already covered the case if there are fewer nearest neighbours
    ## than both R1 and 2 * R2, which returns NaN for all values
    ##
    if pts.shape[0] < R[1] + 1:
        ## We can only calculate S1, not enough nearest neighbours to calculate
        ## S2, this saves some computation
        _, idxs_x = xTree.query(pts[:, x_cols], k = R1 + 1)
        _, idxs_y = yTree.query(pts[:, y_cols], k = R1 + 1)
        ##
        dr_t_xy = np.array([cdist(pts[ii, x_cols].reshape(1, -1), \
            pts[:, x_cols][idxs_y[ii, 1:], :], metric = 'sqeuclidean') \
            for ii in range(n)]).reshape(n, -1)
        dr_t_yx = np.array([cdist(pts[ii, y_cols].reshape(1, -1), \
            pts[:, y_cols][idxs_x[ii, 1:], :], metric = 'sqeuclidean') \
            for ii in range(n)]).reshape(n, -1)
        ##
        ## It's tricky here if the denominator is equal to 0! The original paper
        ## by Arnhold et al [1999] and the review paper by Lungarella et al
        ## [2006] make no mention of this
        ## We have dr_t_x <= dr_t_xy but possibly not dn_t_x <= dr_t_xy
        ## So if the denominator == 0 when in Arnhold's original function
        ## (mean of dr_t_x / dr_t_xy), there is a 0 / 0. If X is constant,
        ## then X and Y are independent, so I think this should contribute 0 to
        ## the similarity indices.
        ind = dr_t_xy.mean(axis = 1) > 1e-15
        si_1_xy = np.sum(np.log(dn_t_x[ind] / dr_t_xy[ind, :].mean(axis = 1)))
        si_1_xy /= n
        ##
        ind = dr_t_yx.mean(axis = 1) > 1e-15
        si_1_yx = np.sum(np.log(dn_t_y[ind] / dr_t_yx[ind, :].mean(axis = 1)))
        si_1_yx /= n
        ##
        si_2_xy = np.nan
        si_2_yx = np.nan
    else:
        ## We need enough nearest neighbours for both S1 and S2
        dists_x, idxs_x = xTree.query(pts[:, x_cols], k = R[1] + 1)
        dists_y, idxs_y = yTree.query(pts[:, y_cols], k = R[1] + 1)
        ##
        ## For S2, the numerator requires only a subset of the R2 nearest nghbs
        dp_t_x = np.square(dists_x[:, R2 + 1:(2 * R2 + 1)]).mean(axis = 1)
        dp_t_y = np.square(dists_y[:, R2 + 1:(2 * R2 + 1)]).mean(axis = 1)
        ##
        ## This is needed for both S1 and S2, hence the finding the distances
        ## up to the R[2]-th nearest neighbour
        ## Again use cdist for computational efficiency (still slow though)
        dr_t_xy = np.array([cdist(pts[ii, x_cols].reshape(1, -1), \
            pts[:, x_cols][idxs_y[ii, 1:R[2] + 1], :], metric = 'sqeuclidean') \
            for ii in range(n)]).reshape(n, -1)
        dr_t_yx = np.array([cdist(pts[ii, y_cols].reshape(1, -1), \
            pts[:, y_cols][idxs_x[ii, 1:R[2] + 1], :], metric = 'sqeuclidean') \
            for ii in range(n)]).reshape(n, -1)
        ##
        ## As above (in if part of if/else) for division by 0 issues
        ind = dr_t_xy[:, :R1].mean(axis = 1) > 1e-15
        si_1_xy = \
            np.sum(np.log(dn_t_x[ind] / dr_t_xy[ind, :R1].mean(axis = 1))) / n
        ##
        ind = dr_t_yx[:, :R1].mean(axis = 1) > 1e-15
        si_1_yx = \
            np.sum(np.log(dn_t_y[ind] / dr_t_yx[ind, :R1].mean(axis = 1))) / n
        ##
        ind = dr_t_xy[:, :R2].mean(axis = 1) > 1e-15
        si_2_xy = np.sum(dp_t_x[ind] / dr_t_xy[ind, :R2].mean(axis = 1)) / n
        si_2_xy = np.mean(si_2_xy)
        ##
        ind = dr_t_yx[:, :R2].mean(axis = 1) > 1e-15
        si_2_yx = np.sum(dp_t_y[ind] / dr_t_yx[ind, :R2].mean(axis = 1)) / n
    ##
    ## si_1_xy is the transfer entropy of x given y e.g. SI1(X|Y)
    return si_1_xy, si_1_yx, si_2_xy, si_2_yx
    ## end function for similarity indices
##
##
##
##
##############################################################
##############################################################
## CONVERGENT CROSS MAPPING
##
def convergent_cross_mapping(x, y, mx = 2, my = 2, rho_tol = 0.05, \
        random = True, n_samples = 20, metric = 'minkowski', seed = 2212):
    ##
    ## Treated as in Clark et al [2015] e.g. convergence if increase in rho
    ## between minimum and maximum library size > rho_tol
    ##
    np.random.seed(seed)
    ##
    ## The initial set up is the same as in almost all other indices, but like
    ## similarity indices doesn't require x_{t + 1}, y_{t + 1}
    m = max(mx, my)
    pts = embedding(x, y, mx = mx, my = my, h = 0)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    if pts.shape[0] < m + 2:
        return np.nan, np.nan
    ##
    def cm_rho_fun(pts_x, y, m = m, metric = metric):
        ##
        xTree = KDTree(pts_x, metric = metric)
        ##
        ## dimension E = m, need m + 1 nearest neighbours for a bounding simplex
        ## (not including itself, hence k = m + 2)
        dists_x, idxs_x = xTree.query(pts_x, k = m + 2)
        u_x = np.exp(- dists_x[:, 1:] / dists_x[:, 1].reshape(-1, 1))
        null_ind = dists_x[:,1] == 0
        u_x[null_ind,:] = dists_x[null_ind, 1:] == 0
        ##
        w_x = u_x / u_x.sum(axis = 1).reshape(-1, 1)
        ##
        y_hat = (y[idxs_x[:, 1:]] * w_x).sum(axis = 1)
        ##
        cm_rho = np.corrcoef(y_hat, y)[0, 1]
        return cm_rho
        ##
    ##
    min_T = m + 2
    params = {'m': m, 'metric': metric}
    ##
    if pts.shape[0] < (n_samples // min_T):
        return np.nan, np.nan
    ##
    cm_xy_max = cm_rho_fun(pts[:, x_cols], pts[:, mx + my - 1], **params)
    cm_yx_max = cm_rho_fun(pts[:, y_cols], pts[:, mx - 1], **params)
    ##
    ## Number of samples for each library length
    if n_samples is not None:
        n_samples = np.min((n_samples, n // min_T))
    else:
        n_samples = n // min_T
    ##
    ## Computation
    for jj in range(n_samples):
        if random:
            ind = np.random.choice(n - min_T)
        else:
            ind = jj * min_T
        pts_jj = pts[ind:ind + min_T, :]
        cm_xy_min = cm_rho_fun(pts_jj[:, x_cols], pts_jj[:, mx + my - 1], **params)
        cm_yx_min = cm_rho_fun(pts_jj[:, y_cols], pts_jj[:, mx - 1], **params)
    ##
    cm_xy_min /= n_samples
    cm_yx_min /= n_samples
    ##
    if cm_yx_max - cm_yx_min < 0 or cm_yx_max < 0:
        ccm_yx = 0
    else:
        ccm_yx = cm_yx_max
    if cm_xy_max - cm_xy_min < 0 or cm_xy_max < 0:
        ccm_xy = 0
    else:
        ccm_xy = cm_xy_max
    ## ccm_xy is the transfer entropy of x given y e.g. CCM(X|Y) / CCM_{Y->X}
    ## Note if x causally influences y, x influences dynamics of y and knowledge
    ## of shadow manifold Ay can be used to estimate x. Hence
    ## CCM(X|Y) = rho(x^, x)
    return ccm_xy, ccm_yx
    ## end function for CCM
##
def convergent_cross_mapping_full(x, y, mx = 2, my = 2, rho_tol = 0.05, \
        exp_fit = False, n_T = 20, \
        plot = False, random = True, n_samples = 20, \
        metric = 'minkowski', seed = 2212):
    ##
    ## if output is anything other than 'exp_fit' (as in Monster et al [2016])
    ## or then it is treated as in Clark et al [2015] e.g. convergence if
    ## increase in rho between minimum and maximum library size > rho_tol
    ## n_T is the number of points on curve
    ## if n_samples = None, then the maximum number of samples are used
    ## plot = True shows a plot of the correlation rho against T values
    ##
    np.random.seed(seed)
    ##
    ## The initial set up is the same as in almost all other indices, but like
    ## similarity indices doesn't require x_{t + 1}, y_{t + 1}
    m = max(mx, my)
    pts = embedding(x, y, mx = mx, my = my, h = 0)
    n = pts.shape[0]
    ##
    x_cols = list(range(mx))
    y_cols = list(range(mx, mx + my))
    ##
    ##
    if pts.shape[0] == 0:
        return np.nan, np.nan
    ##
    ## One aim here is to make this function as minimally computationally
    ## intensive as possible whilst still accurate
    ## One way we can do that is to be smart about which values of T to include
    ## in the computation so that we can show convergence of the CCM without
    ## unnecessary computation
    ## How can we achieve that sensibly?
    ## (a) If we want to fit a curve (i.e. output = 'exp_fit'), then compute
    ##  rho for T = T_0,...,T_m, where T_m = n, T_0 = m + 2
    ##  We could space these T values out equidistantly but better
    ##  (computationally and for fitting) if more weight given to smaller
    ##  values of T, since the function may plateau (and so exponential curve
    ##  fit doesn't converge)
    ##  Not clear how to do this. For now we take T_x = k * a ** x
    ##  We need to find a suitable k and a, e.g.
    ##  a = (T_m - T_0) ** (1 / (n_T - 1)), k = (T_m - T_0) / (a ** n_T)
    ##  seems to work reasonably well (sort of like 2 ** n but optimally scaled)
    ## (b) If we just want to see if there is an increase between T_m = n and
    ##  T_0 = m + 2, and we don't want to plot, then only do these two values
    ##  Unless we have (rho_n - rho_0 > rho_tol), we assume convergence is not
    ##  satisfied and ccm = 0
    ##  Note: this is the strategy in Clark et al [2015], who stop here -
    ##  they use rho_tol = 0
    ##
    ##
    def cm_rho_fun(pts_x, y, m = m, metric = metric):
        ##
        xTree = KDTree(pts_x, metric = metric)
        ##
        ## dimension E = m, need m + 1 nearest neighbours for a bounding simplex
        ## (not including itself, hence k = m + 2)
        dists_x, idxs_x = xTree.query(pts_x, k = m + 2)
        ##
        u_x = np.exp(- np.square(dists_x[:, 1:] / dists_x[:, 1].reshape(-1, 1)))
        u_x = np.exp(- dists_x[:, 1:] / dists_x[:, 1].reshape(-1, 1))
        ##
        w_x = u_x / u_x.sum(axis = 1).reshape(-1, 1)
        ##
        y_hat = (y[idxs_x[:, 1:]] * w_x).sum(axis = 1)
        ##
        cm_rho = np.corrcoef(y_hat, y)[0, 1]
        return cm_rho
        ##
    ##
    def cm_rhos(pts, m = m, metric = metric):
        params = {'m': m, 'metric': metric}
        cm_xy = cm_rho_fun(pts[:, :mx], pts[:, mx + my - 1], **params)
        cm_yx = cm_rho_fun(pts[:, mx:], pts[:, mx - 1], **params)
        return cm_yx, cm_xy
        ##
    ##
    min_T = m + 2
    max_T = n
    ##
    ## Library lengths: list_T
    if plot is True or exp_fit is True:
        a = (max_T - min_T) ** (1 / (n_T - 1)) + 1e-8
        k = (max_T - min_T) / (a ** n_T)
        fun_T = lambda x: k * a ** x
        ##
        list_T = fun_T(np.arange(n_T) + 1) / fun_T(n_T) * (max_T - min_T)
        list_T = min_T + list_T.astype(int)
    else:
        list_T = np.array([min_T, max_T])
    ##
    ## Number of samples for each library length
    n_samples_in = n_samples
    n_samples = n // list_T
    if n_samples_in is not None:
        n_samples = \
            np.min((n_samples, np.repeat(n_samples_in, len(list_T))), axis = 0)
    ##
    ## Computation
    cm_vals = np.zeros((len(list_T), 2))
    for ii in range(len(list_T)):
        for jj in range(n_samples[ii]):
            if random:
                if n == list_T[ii]:
                    ind = 0
                else:
                    ind = np.random.choice(n - list_T[ii])
            else:
                ind = jj * list_T[ii]
            cm_vals[ii, :] += cm_rhos(pts[ind:ind + list_T[ii], :])
        cm_vals[ii, :] /= n_samples[ii]
    ##
    ## If exponential curve fit (as in Monster et al [2016]):
    if exp_fit is True:
        def exp_fun(x, gamma, p_0, p_inf):
            return (p_0 - p_inf) * np.exp(- gamma * x) + p_inf
        p0 = (1e-6, 0, 0.5)
        try:
            par_yx, _ = curve_fit(exp_fun, list_T, cm_vals[:, 0], p0 = p0)
            if exp_fun(list_T[-1], *par_yx) - \
                    exp_fun(list_T[0], *par_yx) < rho_tol:
                ccm_yx = 0
            else:
                ccm_yx = par_yx[-1]
        except:
            ccm_yx = np.nan
        try:
            par_xy, _ = curve_fit(exp_fun, list_T, cm_vals[:, 1], p0 = p0)
            ##
            if exp_fun(list_T[-1], *par_xy) - \
                    exp_fun(list_T[0], *par_xy) < rho_tol:
                ccm_xy = 0
            else:
                ccm_xy = par_xy[-1]
        except:
            ccm_xy = np.nan
    else:
        if cm_vals[-1, 0] - cm_vals[0, 0] < 0 or cm_vals[-1, 0] < 0:
            ccm_yx = 0
        else:
            ccm_yx = cm_vals[-1, 0]
        if cm_vals[-1, 1] - cm_vals[0, 1] < 0 or cm_vals[-1, 1] < 0:
            ccm_xy = 0
        else:
            ccm_xy = cm_vals[-1, 1]
    ##
    if plot is True:
        plt.figure()
        plt.plot(list_T, cm_vals[:, 1], 'bo')
        plt.plot(list_T, cm_vals[:, 0], 'ro')
        if exp_fit:
            xx = np.arange(1, max_T)
            try:
                plt.plot(xx, exp_fun(xx, *par_xy), 'b')
                plt.plot(xx, exp_fun(xx, *par_yx), 'r')
            except:
                pass
    ##
    ## ccm_xy is the transfer entropy of x given y e.g. CCM(X|Y) / CCM_{Y->X}
    ## Note if x causally influences y, x influences dynamics of y and knowledge
    ## of shadow manifold Ay can be used to estimate x. Hence
    ## CCM(X|Y) = rho(x^, x)
    return ccm_xy, ccm_yx
    ## end function for CCM
