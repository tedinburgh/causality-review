##############################################################
##
import numpy as np
import pandas as pd
import time
import argparse
##
## Causality functions
import causality_indices as ci
##
##
##
##
##############################################################
##############################################################
##############################################################
## SIMULATION
##
## Simulate data, returning N simulated values after discarding the first
## N_discard simulated values. e controls the degree of coupling in all cases.
def simulated_data(N, lambda_, map_type = 'linear_process', N_discard = 0, \
        init_random = True, seed = 0, params = None):
    ##
    np.random.seed(seed)
    ##
    ## Linear process: unidirectionally coupled autoregressive process of first
    ## order (Lungeralla et al 2006)
    if map_type == 'linear_process':
        if params is None:
            params = dict({'b_x': 0.8, 'b_y': 0.4, 'var_x': 0.2, 'var_y': 0.2})
        x = np.zeros((N_discard + N, ))
        y = np.zeros((N_discard + N, ))
        ## Random initialisations
        if init_random:
            x[0] = np.random.normal(scale = 0.1)
            y[0] = np.random.normal(scale = 0.1)
        ##
        for ii in range(N_discard + N - 1):
            x[ii + 1] = params['b_x'] * x[ii] + lambda_ * y[ii] + \
                np.random.normal(scale = np.sqrt(params['var_x']))
            y[ii + 1] = params['b_y'] * y[ii] + \
                np.random.normal(scale = np.sqrt(params['var_y']))
        return x[N_discard:], y[N_discard:]
    ##
    ## 1d ring lattice of unidirectionally coupled Ulam maps with periodic
    ## boundary conditions (Schreiber 2000), taking the two series in the
    ## lattice as outputs x and y
    elif map_type == 'ulam_lattice':
        if params is None:
            params = dict({'L': 100, 'fun': lambda x: 2 - x ** 2})
        x = np.zeros((N_discard + N, params['L']))
        if init_random:
            x[0, :] = np.random.normal(scale = 0.1, size = params['L'])
        ##
        for ii in range(N_discard + N - 1):
            for ll in range(1, params['L']):
                z = lambda_ * x[ii, ll - 1] + (1 - lambda_) * x[ii, ll]
                x[ii + 1, ll] = params['fun'](z)
            z = lambda_ * x[ii, -1] + (1 - lambda_) * x[ii, 0]
            x[ii + 1, 0] = params['fun'](z)
        return x[N_discard:, 1], x[N_discard:, 2], x[N_discard:, 3]
    ##
    ## Two unidirectionally coupled Henon maps (Schmitz 2000)
    elif map_type == 'henon_unidirectional':
        if params is None:
            params = dict({'a': 1.4, 'b_x': 0.3, 'b_y': 0.3})
        x = np.zeros((N_discard + N, ))
        y = np.zeros((N_discard + N, ))
        if init_random:
            x[0] = np.random.normal(scale = 0.1)
            y[0] = np.random.normal(scale = 0.1)
        ##
        for ii in range(N_discard + N - 2):
            x[ii + 2] = params['a'] - x[ii + 1] ** 2 + params['b_x'] * x[ii]
            y[ii + 2] = params['a'] - (lambda_ * x[ii + 1] + (1 - lambda_) * \
                y[ii + 1]) * y[ii + 1] + params['b_y'] * y[ii]
        return x[N_discard:], y[N_discard:]
    ##
    ## Bidirectionally coupled Henon maps (e.g. Wiesenfeldt et al 2001)
    elif map_type == 'henon_bidirectional':
        if params is None:
            params = dict({'a': 1.4, 'b_x': lambda_[2], 'b_y': lambda_[3], \
                'lambda_xy': lambda_[0], 'lambda_yx': lambda_[1]})
        x = np.zeros((N_discard + N, ))
        y = np.zeros((N_discard + N, ))
        if init_random:
            x[0] = np.random.normal(scale = 0.1)
            y[0] = np.random.normal(scale = 0.1)
        ##
        for ii in range(N_discard + N - 2):
            x[ii + 2] = params['a'] - x[ii + 1] ** 2 + params['b_x'] * x[ii] + \
                params['lambda_xy'] * (x[ii + 1] ** 2 - y[ii + 1] ** 2)
            y[ii + 2] = params['a'] - y[ii + 1] ** 2 + params['b_y'] * y[ii] + \
                params['lambda_yx'] * (y[ii + 1] ** 2 - x[ii + 1] ** 2)
        return x[N_discard:], y[N_discard:]
    ## end function simulated data
##
##
##
##
##############################################################
##############################################################
## PARAMETERS
##
## Henon bidirectional map has a different set of lambda_vals
n_runs = 10
lambda_vals = np.arange(0, 1 + 0.01, 0.01)
lambda_vals_hb = (np.arange(0, 0.41, 0.01), \
    np.arange(0, 0.41, 0.01), [0.1, 0.3], [0.1])
n_lambda = len(lambda_vals)
n_lambda_hb = np.prod([len(x) for x in lambda_vals_hb])
## Note: elements of lambda_vals_hb are c_xy, c_yx, b_x, b_y respectively
## Confusingly c_xy and b_y are for parameters for y and c_yx, b_x for x
## This is probably a slight mistake in setting this up but doesn't really
## matter as long as we're aware of this and which way around the results table
## should be (i.e. not accidentally flipping c_xy and c_yx)
lambda_ind = [(x, y, z, 0) for z in range(len(lambda_vals_hb[2])) \
    for y in range(len(lambda_vals_hb[1])) \
    for x in range(len(lambda_vals_hb[0]))]
indices_list = ['te', 'ete', 'te-ksg', 'ctir', \
    'egc', 'nlgc', 'pi', 'si1', 'si2', 'ccm']
##
## For input arguments when running the script (i.e. a subset of indices,
## verbose, logging time values)
parser = argparse.ArgumentParser( \
    description = 'Simulations for causality indices')
parser.add_argument('--simulations', '--sim', '-s', \
    default = 'lp', dest = 'sim', \
    help = 'Data simulations to be computed')
## list of indices to compute (transfer entropy, coarse-grained transinformation
## rate, nonlinear Granger causality, extended Granger causality,
## predictability improvement, similarity indices)
parser.add_argument('--indices', \
    default = 'te,ete,te-ksg,ctir,egc,nlgc,pi,si1,si2,ccm', \
    dest = 'indices', help = 'Causality indices to be computed')
parser.add_argument('--verbose', type = bool, default = True, \
    dest = 'verbose', help = 'Verbose')
parser.add_argument('--logt', type = bool, default = True, \
    dest = 'logt', help = 'Log time values')
##
args = parser.parse_args()
args.sim = args.sim.split()
verbose = args.verbose
indices = args.indices.split(',')
indices = [x for x in indices if x in indices_list]
logt = args.logt
##
xy_list = ['xy', 'yx']
inds = [x + '_' + y for x in indices for y in xy_list]
n_inds = len(indices)
inds_time = ['te' in indices or 'ete' in indices, \
    'te-ksg' in indices, 'ctir' in indices, 'egc' in indices, \
    'nlgc' in indices, 'pi' in indices, \
    'si1' in indices or 'si2' in indices, 'ccm' in indices]
n_time = sum(inds_time)
##
##
##
##
##############################################################
##############################################################
## INDICES COMPUTATION FUNCTION
##
## This function does the computation of all specified indices for a given x and
## y, where params is a dictionary of parameter values spanning all indices
def compute_indices(x, y, params, indices, logt = False, verbose = False):
    ##
    ## Initialise each index to NaN (if any index is not listed, then it
    ## won't be returned but easier to set it up here first)
    te = np.nan, np.nan
    teksg = np.nan, np.nan
    ctir = np.nan, np.nan
    egc = np.nan, np.nan
    nlgc = np.nan, np.nan
    pi = np.nan, np.nan
    si = np.nan, np.nan, np.nan, np.nan
    ccm = np.nan, np.nan
    ## if logt is True, then log the time taken to compute each index
    t_vals = []
    pa = params
    ##
    ## Transfer entropy (and extended transfer entropy) using histogram approach
    if 'te' in indices:
        t = time.time()
        te = ci.transfer_entropy(x, y, \
            mx = pa['mx'][pa['m_order'].index('te')], \
            my = pa['my'][pa['m_order'].index('te')], r = pa['r'], \
            effective = ('ete' in indices), shuffle = pa['shuffle'])
        te = te[:2 + 2 * ('ete' in indices)]
        ## Log time values
        if logt:
            t_vals.append(time.time() - t)
        ## Print output
        if verbose:
            print(', '.join([x for x in ['te', 'ete'] if x in indices]))
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Transfer entropy using KSG
    if 'te-ksg' in indices:
        t = time.time()
        teksg = ci.transfer_entropy_ksg(x, y, \
            mx = pa['mx'][pa['m_order'].index('te-ksg')], \
            my = pa['my'][pa['m_order'].index('te-ksg')], k = pa['k'])
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('te-ksg')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Coarse-grained transinformation rate
    if 'ctir' in indices:
        t = time.time()
        ctir = ci.coarse_grained_transinformation_rate(x, y, k = pa['k'], \
            tau_max = pa['tau_max'])
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('ctir')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Extended Granger causality
    if 'egc' in indices:
        t = time.time()
        egc = ci.extended_granger_causality(x, y, \
            mx = pa['mx'][pa['m_order'].index('egc')], \
            my = pa['my'][pa['m_order'].index('egc')], L = pa['L'], \
            delta = pa['delta'], min_k = pa['min_k'])
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('egc')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Nonlinear Granger causality
    if 'nlgc' in indices:
        t = time.time()
        nlgc = ci.nonlinear_granger_causality(x, y, \
            mx = pa['mx'][pa['m_order'].index('nlgc')], \
            my = pa['my'][pa['m_order'].index('nlgc')], P = pa['P'], \
            sigma = pa['sigma'], clustering = pa['clustering'])
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('nlcg')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Predictability improvement
    if 'pi' in indices:
        t = time.time()
        pi = ci.predictability_improvement(x, y, \
            mx = pa['mx'][pa['m_order'].index('pi')], \
            my = pa['my'][pa['m_order'].index('pi')], h = pa['h'], R = pa['R'])
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('pi')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Similarity indices
    if 'si1' in indices or 'si2' in indices:
        t = time.time()
        si = ci.similarity_indices(x, y, \
            mx = pa['mx'][pa['m_order'].index('si')], \
            my = pa['my'][pa['m_order'].index('si')], \
            R1 = pa['R1'], R2 = pa['R2'], N_max = pa['N_max'])
        if 'si2' not in indices:
            si = si[:2]
        elif 'si1' not in indices:
            si = si[2:]
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('si')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## Convergent cross mapping
    if 'ccm' in indices:
        t = time.time()
        ccm = ci.convergent_cross_mapping(x, y, \
            mx = pa['mx'][pa['m_order'].index('ccm')], \
            my = pa['my'][pa['m_order'].index('ccm')], \
            n_samples = pa['n_samples'], rho_tol = pa['rho_tol'])
        if logt:
            t_vals.append(time.time() - t)
        if verbose:
            print('ccm')
            print('Time: ' + str(np.round(time.time() - t, 3)))
    ##
    ## FIX THIS
    inds_bool = ['te' in indices or 'ete' in indices, \
        'te-ksg' in indices, 'ctir' in indices, 'egc' in indices, \
        'nlgc' in indices, 'pi' in indices, \
        'si1' in indices or 'si2' in indices, 'ccm' in indices]
    output = [(te, teksg, ctir, egc, nlgc, pi, si, ccm)[jj] \
        for jj in range(8) if inds_bool[jj]]
    output = [val for ind in output for val in ind]
    if logt:
        return output, t_vals
    else:
        return output
    ## end function for computing all indices
##
## Save and load to csv files, requires reshaping to a 2d array (from 3d array)
def save_reshape(results, shape, filename = 'results'):
    results = results.reshape(-1, shape[-1])
    pd.DataFrame(results).to_csv(filename + '.csv', header = None, index = None)
    results = results.reshape(shape)
##
def load_reshape(filename, shape):
    results = pd.read_csv(filename + '.csv', header = None)
    results = np.array(results).reshape(shape)
    return results
##
##
##
##
##############################################################
##############################################################
##############################################################
## PERFORMING SIMULATIONS
##
print('Starting: simulations')
print(time.ctime())
##
## Simulation parameters
## te, ete: mx, my, r, shuffle, (effective = True)
## te-ksg: mx, my, k, (metric = 'chebyshev', algorithm = 1)
## ctir: mx, my, k, max_iter, (tau_max = None, metric = 'chebyshev')
## egc: mx, my, L, delta, min_k (metric = 'manhattan')
## nlgc: mx, my, P, sigma, fcm_error, fcm_max_iter, clustering
## pi: mx, my, h, R, (metric = 'minkowski')
## si1, si2: mx, my, R1, R2, N_max, (metric = 'minkowski')
##
##
## Linear (Gaussian) process
if 'lp' in args.sim:
    ## Simulation parameters
    lp_params = {'m_order': ['te', 'ete', 'te-ksg', 'ctir', \
            'egc', 'nlgc', 'pi', 'si', 'ccm'], \
        'mx': [1, 1, 1, None, 2, 2, 1, 2, 2], \
        'my': [1, 1, 1, None, 2, 2, 1, 2, 2], \
        'r': 8, 'shuffle': 10, 'k': 4, 'tau_max': 20, \
        'L': 20, 'delta': 0.8, 'min_k': 20, 'P': 10, 'sigma': 0.05, \
        'clustering': 'kmeans', 'h': 1, 'R': 10, \
        'R1': 10, 'R2': 30, 'N_max': None, 'n_samples': 40, 'rho_tol': 0.05}
    ##
    lp_shape = (n_runs, n_lambda, 2 * n_inds), (n_runs, n_lambda, n_time)
    lp_results = np.zeros(lp_shape[0])
    lp_time = np.zeros(lp_shape[1])
    ci_args = {'indices': indices, 'logt': logt, 'verbose': verbose}
    print('Starting: linear process')
    ##
    for jj in range(n_lambda):
        for ii in range(n_runs):
            x, y = simulated_data(N = 10 ** 4, lambda_ = lambda_vals[jj], \
                map_type = 'linear_process', N_discard = 10 ** 4, \
                seed = ii * 101)
            lp_results[ii, jj, :], lp_time[ii, jj, :] = \
                compute_indices(x, y, lp_params, **ci_args)
        ##
        print('Completed run: ' + str(ii) + ' for linear process')
        save_reshape(lp_results, lp_shape[0], filename = 'lp_values')
        save_reshape(lp_time, lp_shape[1], filename = 'lp_time')
        print(time.ctime())
    ##
    save_reshape(lp_results, lp_shape[0], filename = 'lp_values')
    save_reshape(lp_time, lp_shape[1], filename = 'lp_time')
    print(time.ctime())
##
##
## Ulam lattice
if 'ul' in args.sim:
    ul_params = {'m_order': ['te', 'ete', 'te-ksg', 'ctir', \
            'egc', 'nlgc', 'pi', 'si', 'ccm'], \
        'mx': [1, 1, 1, None, 1, 1, 1, 1, 1], \
        'my': [1, 1, 1, None, 1, 1, 1, 1, 1], \
        'r': 8, 'shuffle': 10, 'k': 4, 'tau_max': 5, \
        'L': 100, 'delta': 0.5, 'min_k': 20, 'P': 50, 'sigma': 0.05, \
        'clustering': 'kmeans', 'h': 1, 'R': 1, \
        'R1': 20, 'R2': 20, 'N_max': None, 'n_samples': 40, 'rho_tol': 0.05}
    ##
    ul_shape = (n_runs, n_lambda, n_inds * 4), (n_runs, n_lambda, n_time * 2)
    ul_results = np.zeros(ul_shape[0])
    ul_time = np.zeros(ul_shape[1])
    ci_args = {'indices': indices, 'logt': logt, 'verbose': verbose}
    print('Starting: Ulam lattice')
    print(time.ctime())
    for n in range(2):
        kk = range(n * n_inds * 2, (n + 1) * n_inds * 2), \
            range(n * n_time, (n + 1) * n_time)
        for jj in range(n_lambda):
            for ii in range(n_runs):
                x, y, _ = simulated_data(N = 10 ** [3, 5][n], \
                    lambda_ = lambda_vals[jj], map_type = 'ulam_lattice', \
                    N_discard = 10 ** 5, seed = ii * 101)
                ul_params['delta'] = [0.5, 0.2][n]
                ul_results[ii, jj, kk[0]], ul_time[ii, jj, kk[1]] = \
                    compute_indices(x, y, ul_params, **ci_args)
            ##
            print('Completed run: ' + str(ii) + ' for Ulam lattice')
            save_reshape(ul_results, ul_shape[0], filename = 'ul_values')
            save_reshape(ul_time, ul_shape[1], filename = 'ul_time')
            print(time.ctime())
    ##
    save_reshape(ul_results, ul_shape[0], filename = 'ul_values')
    save_reshape(ul_time, ul_shape[1], filename = 'ul_time')
    print(time.ctime())
##
##
## Henon unidirectional map
if 'hu' in args.sim:
    hu_params = {'m_order': ['te', 'ete', 'te-ksg', 'ctir', \
            'egc', 'nlgc', 'pi', 'si', 'ccm'], \
        'mx': [1, 1, 1, None, 2, 2, 2, 2, 2], \
        'my': [1, 1, 1, None, 2, 2, 2, 2, 2], \
        'r': 8, 'shuffle': 10, 'k': 4, 'tau_max': 5, \
        'L': 100, 'delta': 0.5, 'min_k': 20, 'P': 50, 'sigma': 0.05, \
        'clustering': 'kmeans', 'h': 1, 'R': 1, \
        'R1': 20, 'R2': 20, 'N_max': None, 'n_samples': 40, 'rho_tol': 0.05}
    ##
    hu_shape = (n_runs, n_lambda, n_inds * 3), (n_runs, n_lambda, n_time * 3)
    hu_results = np.zeros(hu_shape[0])
    hu_time = np.zeros(hu_shape[1])
    ci_args = {'indices': indices, 'logt': logt, 'verbose': verbose}
    print('Starting: Henon unidirectional')
    ##
    for n in range(3):
        for jj in range(n_lambda): #n_lambda
            for ii in range(n_runs):
                x, y = simulated_data(N = 10 ** (n + 3), \
                    lambda_ = lambda_vals[jj], N_discard = 10 ** 5,
                    map_type = 'henon_unidirectional', seed = ii * 101)
                f = open('xy_current_hu.txt', 'w')
                np.savetxt(f, np.hstack((ii, jj, n, x, y)))
                f.close()
                ##
                hu_params['delta'] = [0.5, 0.3, 0.2][n]
                hu_params['P'] = [50, 50, 100][n]
                hu_params['N_max'] = [None, None, 10 ** 4][n]
                hu_ind_z = range(n * n_time, (n + 1) * n_time)
                c_output, hu_time[ii, jj, hu_ind_z] = \
                    compute_indices(x, y, hu_params, **ci_args)
                for kk in range(n_inds):
                    hu_results[ii, jj, n_inds * n + kk] = \
                        c_output[2 * kk] - c_output[2 * kk + 1]
            ##
            print('Completed run: ' + str(ii) + ' for Henon unidirectional map')
            save_reshape(hu_results, hu_shape[0], filename = 'hu_values')
            save_reshape(hu_time, hu_shape[1], filename = 'hu_time')
            print(time.ctime())
    ##
    save_reshape(hu_results, hu_shape[0], filename = 'hu_values')
    save_reshape(hu_time, hu_shape[1], filename = 'hu_time')
    print(time.ctime())
##
##
## Henon bidirectional map
if 'hb' in args.sim:
    hb_params = {'m_order': ['te', 'ete', 'te-ksg', 'ctir', \
            'egc', 'nlgc', 'pi', 'si', 'ccm'], \
        'mx': [1, 1, 1, None, 2, 2, 2, 2, 2], \
        'my': [1, 1, 1, None, 2, 2, 2, 2, 2], \
        'r': 8, 'shuffle': 10, 'k': 4, 'tau_max': 5, \
        'L': 100, 'delta': 0.6, 'min_k': 20, 'P': 10, 'sigma': 0.05, \
        'clustering': 'kmeans', 'h': 1, 'R': 1, \
        'R1': 20, 'R2': 100, 'N_max': None, 'n_samples': 40, 'rho_tol': 0.05}
    ##
    hb_shape = (n_runs, n_lambda_hb, n_inds), \
        (n_runs, n_lambda_hb, n_time)
    hb_results = np.zeros(hb_shape[0])
    hb_time = np.zeros(hb_shape[1])
    ci_args = {'indices': indices, 'logt': logt, 'verbose': verbose}
    print('Starting: Henon bidirectional')
    ##
    for jj in range(n_lambda_hb):
        for ii in range(n_runs):
            lambda_vals_jj = \
                [lambda_vals_hb[kk][lambda_ind[jj][kk]] for kk in range(4)]
            x, y = simulated_data(N = 10 ** 4, lambda_ = lambda_vals_jj, \
                map_type = 'henon_bidirectional', \
                N_discard = 10 ** 5, seed = ii * 101)
            ##
            c_output, hb_time[ii, jj, :] = \
                compute_indices(x, y, hb_params, **ci_args)
            for kk in range(n_inds):
                hb_results[ii, jj, kk] = c_output[2 * kk] - c_output[2 * kk + 1]
        print('Completed run: ' + str(ii) + ' for Henon bidirectional map')
        save_reshape(hb_results, hb_shape[0], filename = 'hb_values')
        save_reshape(hb_time, hb_shape[1], filename = 'hb_time')
        print(time.ctime())
    ##
    save_reshape(hb_results, hb_shape[0], filename = 'hb_values')
    save_reshape(hb_time, hb_shape[1], filename = 'hb_time')
    print(time.ctime())
##
##
##############################################################
##############################################################
##############################################################
## TRANSFORMATIONS
##
def transform(tup, **kwargs):
    ## Each transformation can only be repeated as a keyword argument,
    ## so if any need repeating for some reason, then the function can be called
    ## again e.g. the difference between:
    ##  transform(tup, scale_x = 3, log_x = True, scale_x = 2) ## not allowed
    ## and (allowed):
    ##  transform(transform(tup, scale_x = 3, log_x = True), scale_x = 2)
    ##
    funs = dict({
        ## substitute x in place of y (i.e. measure between x and x)
        'y_to_x': lambda tup: \
            (tup[0], tup[0]) if kwargs.get('y_to_x', False) else tup,
        ## scale x / y by constant value
        'scale_x': lambda tup: (tup[0] * kwargs.get('scale_x', 1), tup[1]),
        'scale_y': lambda tup: (tup[0], kwargs.get('scale_y', 1) * tup[1]),
        ## shift x / y by constant value
        'shift_x': lambda tup: (tup[0] + kwargs.get('shift_x', 0), tup[1]),
        'shift_y': lambda tup: (tup[0], kwargs.get('shift_y', 0) + tup[1]),
        ## round x / y to specified number of decimal places
        'round_x': lambda tup: \
            (np.round(tup[0], kwargs.get('round_x', 16)), tup[1]),
        'round_y': lambda tup: \
            (tup[0], np.round(tup[1], kwargs.get('round_y', 16))),
        ## set a selection of x / y to NaN
        'na_x': lambda tup: \
            (np.where(np.random.binomial(1, kwargs.get('na_x', 0) / 100, \
                size = len(tup[0])), np.nan, tup[0]), tup[1]),
        'na_y': lambda tup: \
            (tup[0], \
            np.where(np.random.binomial(1, kwargs.get('na_y', 0) / 100, \
                size = len(tup[1])), np.nan, tup[1])),
        ## normalise both x and y
        'normalise': lambda tup: ((tup[0] - tup[0].mean()) / tup[0].std(), \
            (tup[1] - tup[1].mean()) / tup[1].std()) \
                if kwargs.get('normalise', True) else tup,
        ## add Gaussian noise to x only
        'gaussian_x': lambda tup: \
            (tup[0] + np.random.normal(scale = kwargs.get('gaussian_x', 0), \
                size = len(tup[0])), tup[1]),
        ## add Gaussian noise to y only
        'gaussian_y': lambda tup: (tup[0], tup[1] + \
            np.random.normal(scale = kwargs.get('gaussian_y', 0), \
                size = len(tup[1])))
    })
    ##
    for kw in kwargs.keys():
        tup = funs[kw](tup)
    ##
    return tup
    ## end function for transformations
##
lambda_vals_tf = np.arange(0, 1 + 0.01, 0.01)
n_lambda_tf = len(lambda_vals_tf)
##
tf_list = np.array([ \
    {'y_to_x': True}, {'y_to_x': True, 'scale_x': 2}, \
    {'normalise': True}, {'scale_x': 10, 'scale_y': 1}, \
    {'scale_x': 1, 'scale_y': 10}, {'round_x': 1}, \
    {'round_y': 1}, {'round_x': 2, 'round_y': 2}, \
    {'na_x': 10, 'na_y': 0}, {'na_x': 0, 'na_y': 10}, \
    {'na_x': 10, 'na_y': 10}, {'na_x': 20, 'na_y': 20}, \
    {'gaussian_x': 0.1, 'gaussian_y': 0.1}, \
    {'gaussian_x': 1}, {'gaussian_y': 1}])
n_tf = len(tf_list)
tf_split = [2, 3, 4, 4, 3]
##
## Ulam lattice with transformations
if 'ult' in args.sim:
    ult_params = {'m_order': ['te', 'ete', 'te-ksg', 'ctir', \
            'egc', 'nlgc', 'pi', 'si', 'ccm'], \
        'mx': [1, 1, 1, None, 1, 1, 1, 1, 1], \
        'my': [1, 1, 1, None, 1, 1, 1, 1, 1], \
        'r': 8, 'shuffle': 10, 'k': 4, 'tau_max': 5, \
        'L': 100, 'delta': 0.5, 'min_k': 20, 'P': 50, 'sigma': 0.05, \
        'clustering': 'kmeans', 'h': 1, 'R': 1, \
        'R1': 20, 'R2': 20, 'N_max': None, 'n_samples': 40, 'rho_tol': 0.05}
    ##
    ult_shape = (n_runs, n_lambda_tf, 2 * n_inds, n_tf), \
        (n_runs, n_lambda_tf, n_time, n_tf)
    ult_results = np.zeros(ult_shape[0])
    ult_time = np.zeros(ult_shape[1])
    ci_args = {'indices': indices, 'logt': logt, 'verbose': verbose}
    print('Starting: ulam map (transformations)')
    ##
    for jj in range(len(lambda_vals_tf)):
        for ii in range(n_runs):
            x_sim, y_sim, _ = simulated_data(N = 10 ** 3, \
                lambda_ = lambda_vals_tf[jj], map_type = 'ulam_lattice', \
                N_discard = 10 ** 5, seed = ii * 101)
            for ll in range(n_tf):
                np.random.seed(ll * 102)
                x, y = transform((x_sim, y_sim), **tf_list[ll])
                ult_results[ii, jj, :, ll], ult_time[ii, jj, :, ll] = \
                    compute_indices(x, y, ult_params, **ci_args)
        print('Completed run: ' + str(ii) + ' for ulam map')
        save_reshape(ult_results, ult_shape[0], filename = 'ult_values')
        save_reshape(ult_time, ult_shape[1], filename = 'ult_time')
        print(time.ctime())
    ##
    save_reshape(ult_results, ult_shape[0], filename = 'ult_values')
    save_reshape(ult_time, ult_shape[1], filename = 'ult_time')
