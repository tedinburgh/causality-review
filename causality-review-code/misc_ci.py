##############################################################
##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap as lcm
from matplotlib import ticker
try:
    from palettable.colorbrewer import sequential as sq
    from palettable.colorbrewer import diverging as dv
    from palettable.colorbrewer import qualitative as ql
except Exception as err:
    print(err, ': please install module palettable')
##
##
##
##
##############################################################
##############################################################
##############################################################
## ANALYTICAL SOLUTIONS
## for information theoretic indices, linear (Gaussian) process
##
## Computes the analytical solution for transfer entropy in the case of
## the linear process simulation (see simulated_data_log.py for simulated and
## and the preprint for the working)
def te_gaussian(lambda_, b_x, b_y, var_x, var_y):
    c = (1 - b_y ** 2) * (1 - b_y * b_x) ** 2 * var_x ** 2
    numer = c + 2 * lambda_ ** 2 * (1 - b_x * b_y) * var_x * var_y + \
        lambda_ ** 4 * var_y ** 2
    denom = c + lambda_ ** 2 * (1 - b_x ** 2 * b_y ** 2) * var_x * var_y
    te_xy = 0
    te_yx = 0.5 * np.log(numer / denom)
    return te_xy, te_yx
    ##
##
##
## Similarly for coarse-grained transinformation rate in the case of
## the linear process simulation (see simulated_data_log.py for simulated and
## and the preprint for the working)
def ctir_gaussian(lambda_, b_x, b_y, var_x, var_y, tau_max):
    u = (1 - b_y ** 2) / var_y
    v = (1 - b_x ** 2)
    w = (1 - b_x * b_y)
    def b_psum(n):
        if n < 0:
            return 0
        else:
            out = np.sum([b_x ** ii * b_y ** (n - ii) for ii in range(n + 1)])
            return out
    ##
    cyyn = lambda n: (b_y ** n) / u
    cxxn = lambda n: (b_x ** n) * var_x / v + \
        lambda_ ** 2 / (u * v * w) * \
            (v * b_psum(n) + b_x ** (n + 1) * (b_x + b_y))
    cyxn = lambda n: lambda_ / (u * w) * (b_x ** n * b_y + w * b_psum(n - 1))
    cxyn = lambda n: lambda_ / (u * w) * b_y ** (n + 1)
    ##
    ctir_vals = np.zeros((tau_max, 4))
    for ii in range(tau_max):
        kk = ii + 1
        det_yx = cyyn(0) * cxxn(0) - cyxn(0) ** 2
        det_yxn = cyyn(0) * cxxn(0) - cyxn(kk) ** 2
        det_xyn = cyyn(0) * cxxn(0) - cxyn(kk) ** 2
        det_xxn = cxxn(0) ** 2 - cxxn(kk) ** 2
        det_yyn = cyyn(0) ** 2 - cyyn(kk) ** 2
        det_yxxn = cyyn(0) * cxxn(0) ** 2 + \
            2 * cxyn(0) * cxxn(kk) * cyxn(kk) - cxxn(0) * cyxn(kk) ** 2 - \
            cyyn(0) * cxxn(kk) ** 2 - cxxn(0) * cxyn(0) ** 2
        det_xyyn = cxxn(0) * cyyn(0) ** 2 + \
            2 * cyxn(0) * cyyn(kk) * cxyn(kk) - cyyn(0) * cxyn(kk) ** 2 - \
            cxxn(0) * cyyn(kk) ** 2 - cyyn(0) * cyxn(0) ** 2
        ctir_vals[ii, 0] = np.log((det_yx * det_yyn) / (det_xyyn * cyyn(0)))
        ctir_vals[ii, 1] = np.log((det_yx * det_xxn) / (det_yxxn * cxxn(0)))
        ctir_vals[ii, 2] = np.log((cxxn(0) * cyyn(0)) / det_xyn)
        ctir_vals[ii, 3] = np.log((cxxn(0) * cyyn(0)) / det_yxn)
    ctir_vals *= 0.5
    ctir_xy = (2 * ctir_vals[:,0] - ctir_vals[:,2] - ctir_vals[:,3]).mean() / 2
    ctir_yx = (2 * ctir_vals[:,1] - ctir_vals[:,2] - ctir_vals[:,3]).mean() / 2
    return ctir_xy, ctir_yx
    ##
##
##
##
##
##############################################################
##############################################################
##############################################################
## PLOTTING FUNCTIONS
## (to match Lungeralla et al 2006)
##
##
## Save and load to csv files, requires a reshaping to a 2d array (from 3d)
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
## Plot #1:
## Used for simulations of linear processes and Ulam lattice
def sim_plot1(mean_vals, std_vals, lambda_vals, ylabs, ylims = None, \
        analytic_solutions = None, nrows = 3, ncols = 4, labelpads = None, \
        figpad = None, skip_ax = list(), figsize = None, cols = None, \
        filename = 'ci_figure1'):
    ##
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    ## If we want to skip an axis (in order to group certain indices then use
    ## skip_ax argument to remove this from rowcol)
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_inds = mean_vals.shape[1]
    n_ax = np.min((nrows * ncols, n_inds))
    if labelpads == None:
        labelpads = [None] * n_inds
    ##
    with PdfPages(filename + '.pdf') as pdf:
        ##
        if figsize is None:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True)
        else:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, \
                sharex = True, figsize = figsize)
        ##
        for ii in range(n_ax):
            ax_temp = ax[rowcol_show[ii]]
            ##
            ## Add in error bars for one standard deviation (before means)
            for jj in range(n_lambda):
                for kk in range(2):
                    ax_temp.plot([lambda_vals[jj], lambda_vals[jj]], \
                        mean_vals[jj, ii, kk] + \
                            np.array([-1, 1]) * std_vals[jj, ii, kk], \
                        c = 'black', lw = 0.1)
            ##
            if cols is None:
                ## Default: blue for x>y, red for y>x
                cols = ['blue', 'red', 'darkblue', 'darkred']
            ##
            ax_temp.plot(lambda_vals, mean_vals[:, ii, 0], \
                c = cols[0], lw = 1.8, label = r'$i_{X\rightarrow Y}$')
            ax_temp.plot(lambda_vals, mean_vals[:, ii, 1], \
                c = cols[1], lw = 1.8, label = r'$i_{Y\rightarrow X}$')
            if analytic_solutions is not None:
                as_temp = analytic_solutions[ii]
                if as_temp is not None:
                    ax_temp.plot(lambda_vals, as_temp[:, 0], c = cols[2], \
                        lw = 1.8, linestyle = 'dashed', \
                        label = r'$i_{X\rightarrow Y}$: Analytic solution')
                    ax_temp.plot(lambda_vals, as_temp[:, 1], c = cols[3], \
                        lw = 1.8, linestyle = 'dashed',
                        label = r'$i_{Y\rightarrow X}$: Analytic solution')
            ax_temp.set_ylabel(ylabs[ii].upper(), labelpad = labelpads[ii])
            ## If ylims specified as an (n_ind, 2) array then include this
            if ylims is not None:
                if np.any(ylims[ii,:] is not None):
                    ax_temp.set_ylim(ylims[ii, 0], ylims[ii, 1])
            ## x labels only on the bottom row
            if rowcol_show[ii][0] == nrows - 1:
                ax_temp.set_xlabel(r'Coupling ${\lambda}$')
            ##
        ##
        ## Set 'axis = off' for remaining axes
        for ii in range(n_ax + len(skip_ax), nrows * ncols):
            ax[rowcol[ii]].axis('off')
        for ax_ind in skip_ax:
            ax[rowcol[ax_ind]].axis('off')
        ##
        ## Add axis to the bottom right plot
        if ii < nrows * ncols:
            ax_legend = ax[rowcol[-1]]
            label_params = ax[rowcol[0]].get_legend_handles_labels()
            ax_legend.legend(*label_params, fontsize = 'medium', loc = 'center')
        else:
            ax_temp.legend(fontsize = 'x-small')
        ##
        if figpad is None:
            plt.tight_layout()
        else:
            plt.tight_layout(pad = figpad[0], \
                h_pad = figpad[1], w_pad = figpad[2])
        pdf.savefig(fig)
        plt.close()
    ## end function sim_plot1
##
##
## Plot #2:
## Used for simulations of Henon unidirectional maps using the indices
## C_i = i_xy - i_yx where i is any of the causality measures
def sim_plot2(mean_vals, std_vals, lambda_vals, ylabs, ylims = None, \
        nrows = 3, ncols = 4, skip_ax = list(), figpad = None, figsize = None, \
        cols = None, linestyles = None, labelpads = None, \
        filename = 'ci_figure2'):
    ##
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_inds = mean_vals.shape[1]
    n_ax = np.min((nrows * ncols, n_inds))
    if labelpads == None:
        labelpads = [None] * n_inds
    ##
    with PdfPages(filename + '.pdf') as pdf:
        ##
        if figsize is None:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True)
        else:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, \
                sharex = True, figsize = figsize)
        ##
        for ii in range(n_ax):
            ax_temp = ax[rowcol_show[ii]]
            ##
            for jj in range(n_lambda):
                for kk in range(3):
                    ax_temp.plot([lambda_vals[jj], lambda_vals[jj]], \
                        mean_vals[jj, ii, kk] + \
                            np.array([-1, 1]) * std_vals[jj, ii, kk], \
                        c = 'black', lw = 0.1)
            ## Showing simulation results for different lengths of input
            if cols is None:
                ## Default: blue for 10^3, red for 10^4, green for 10^5
                cols = ['blue', 'red', 'green']
            if linestyles is None:
                linestyles = ['dotted', 'dashed', 'solid']
            label_str = r'i$_{X\rightarrow Y}$ - i$_{Y\rightarrow X}$, '
            label_str_add = [r'$T = 10^3$', r'$T = 10^4$', r'$T = 10^5$']
            ##
            for kk in range(3):
                ax_temp.plot(lambda_vals, mean_vals[:, ii, kk], \
                    label = label_str + label_str_add[kk], \
                    c = cols[kk], linestyle = linestyles[kk], lw = 1.8)
            # ax_temp.plot(lambda_vals, mean_vals[:, ii, 0], \
            #     label = label_str + r'$T = 10^3$', \
            #     c = cols[0], linestyle = linestyles[0], lw = 1.5)
            # ax_temp.plot(lambda_vals, mean_vals[:, ii, 1], \
            #     label = label_str + r'$T = 10^4$', \
            #     c = cols[1], linestyle = linestyles[1], lw = 1.5)
            # ax_temp.plot(lambda_vals, mean_vals[:, ii, 2], \
            #     label = label_str + r'$T = 10^5$', \
            #     c = cols[2], linestyle = linestyles[2], lw = 1.5)
            ax_temp.set_ylabel(ylabs[ii].upper(), labelpad = labelpads[ii])
            ##
            ## If ylims specified as an (n_ind, 2) array then include this
            if ylims is not None:
                ax_temp.set_ylim(ylims[ii, 0], ylims[ii, 1])
            if rowcol_show[ii][0] == nrows - 1:
                ax_temp.set_xlabel(r'Coupling ${\lambda}$')
            ## x labels only on the bottom row
        ##
        ## Set 'axis = off' for remaining axes
        for ii in range(n_ax + len(skip_ax), nrows * ncols):
            ax[rowcol[ii]].axis('off')
        for ax_ind in skip_ax:
            ax[rowcol[ax_ind]].axis('off')
        ##
        ## Add axis to the bottom right plot
        if ii < nrows * ncols:
            ax_legend = ax[rowcol[-1]]
            label_params = ax_temp.get_legend_handles_labels()
            ax_legend.legend(*label_params, fontsize = 'medium', loc = 'center')
        else:
            ax_temp.legend(fontsize = 'x-small')
        ##
        if figpad is None:
            plt.tight_layout()
        else:
            plt.tight_layout(pad = figpad[0], \
                h_pad = figpad[1], w_pad = figpad[2])
        pdf.savefig(fig)
        plt.close()
    ## end function sim_plot2
##
##
## Plot #3:
## Used for simulations of Henon bidirectional maps using the indices
## C_i = i_xy - i_yx where i is any of the causality measures
## This is a set of heatmaps rather than plots
def sim_plot3(vals, lambda_vals, titles, vlims = None, transpose = True, \
        vlim_percentiles = [1, 99], nrows = 3, ncols = 4, skip_ax = list(), \
        cmap = None, figpad = None, figsize = None, filename = 'ci_figure3'):
    ##
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_inds = vals.shape[2]
    n_ax = np.min((nrows * ncols, n_inds))
    extent = np.min(lambda_vals[0]), np.max(lambda_vals[0]), \
        np.min(lambda_vals[1]), np.max(lambda_vals[1])
    ##
    with PdfPages(filename + '.pdf') as pdf:
        ##
        if figsize is None:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, \
                sharex = True, sharey = True)
        else:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, \
                sharex = True, sharey = True, figsize = figsize)
        ##
        for ii in range(n_ax):
            ##
            ax_temp = ax[rowcol_show[ii]]
            ##
            if vlims is None:
                vlims = np.zeros((n_ax, 2))
                ## cut vlim so extreme values don't influence the colour scale
                percentiles = np.nanpercentile(vals[:, :, ii], vlim_percentiles)
                vlims[ii, :] = np.array([-1, 1]) * np.max(np.abs(percentiles))
                # vlim = np.max(np.abs(np.nanpercentile(results_temp,[0, 100])))
            if transpose is True:
                z = vals[:, :, ii].T
            else:
                z = vals[:, :, ii]
            if cmap is None:
                cmap = 'RdYlGn'
            im = ax_temp.imshow(z, origin = 'lower', cmap = cmap, \
                vmin = vlims[ii, 0], vmax = vlims[ii, 1], extent = extent)
            ##
            ## Set title as the indices (rather than as ylabels as in
            ## the other plotting functions)
            if titles is not None:
                ax_temp.set_title(titles[ii].upper())
                # ax_temp.set_title(titles[ii].upper(), fontsize = 14)
            ##
            # plt.setp(ax_temp.get_xticklabels(), fontsize = 10)
            # plt.setp(ax_temp.get_yticklabels(), fontsize = 10)
            ##
            if rowcol_show[ii][0] == nrows - 1:
                ax_temp.set_xlabel(r'$\lambda_{xy}$')
                # ax_temp.set_xlabel(r'$\lambda_{xy}$', fontsize = 12)
            if rowcol_show[ii][1] == 0:
                ax_temp.set_ylabel(r'$\lambda_{yx}$')
                # ax_temp.set_ylabel(r'$\lambda_{yx}$', fontsize = 12)
            ##
            ## Add a colourbar to each subplot
            fm = ticker.ScalarFormatter()
            fm.set_powerlimits((-3, 3))
            cbar = ax_temp.figure.colorbar(im, ax = ax_temp, \
                fraction = 0.046, pad = 0.04, format = fm)
            # cbar.ax.tick_params(labelsize = 10)
            ##
        ## Add axis to the bottom right plot
        for ii in range(n_ax + len(skip_ax), nrows * ncols):
            ax[rowcol[ii]].axis('off')
        for ax_ind in skip_ax:
            ax[rowcol[ax_ind]].axis('off')
        ##
        if figpad is None:
            plt.tight_layout()
        else:
            plt.tight_layout(pad = figpad[0], \
                h_pad = figpad[1], w_pad = figpad[2])
        pdf.savefig(fig)
        plt.close()
    ## end function sim_plot3
##
##
def sim_plot4(mean_vals, std_vals, lambda_vals, ylabs, ylims = None, \
        cols = None, tf_names = None, nrows = 3, ncols = 4, labelpads = None, \
        figpad = None, skip_ax = list(), figsize = None, linestyles = None, \
        yticks = None, filename = 'ci_figure4'):
    ##
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    ## If we want to skip an axis (in order to group certain indices then use
    ## skip_ax argument to remove this from rowcol)
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_inds = mean_vals.shape[1]
    n_xy = mean_vals.shape[2]
    n_tf = mean_vals.shape[3]
    n_ax = np.min((nrows * ncols, n_inds))
    if labelpads is None:
        labelpads = [None] * n_inds
    if cols is None:
        cols = np.array(['blue'] * n_tf * n_xy)
        cols = cols.reshape((n_tf, n_xy), order = 'F')
    if tf_names is None:
        tf_names = np.array(['Result' + str(x) for x in np.arange(n_tf * n_xy)])
        tf_names = tf_names.reshape(n_tf, n_xy)
    ##
    with PdfPages(filename + '.pdf') as pdf:
        ##
        if figsize is None:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharex = True)
        else:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, \
                sharex = True, figsize = figsize)
        ##
        for ii in range(n_ax):
            ax_temp = ax[rowcol_show[ii]]
            ##
            for kk in range(n_xy):
                for ll in range(n_tf):
                    ll = n_tf - ll - 1
                    ##
                    ## Add in error bars for one std (before means)
                    for jj in range(n_lambda):
                        ax_temp.plot([lambda_vals[jj], lambda_vals[jj]], \
                            mean_vals[jj, ii, kk, ll] + np.array([-1, 1]) * \
                                std_vals[jj, ii, kk, ll], \
                            c = 'black', lw = 0.1)
                    ##
                    if linestyles is None:
                        ls = 'solid'
                    else:
                        ls = linestyles[ll]
                    ax_temp.plot(lambda_vals, mean_vals[:, ii, kk, ll], \
                        c = cols[kk, ll, :], lw = 1.5, ls = ls, \
                        label = tf_names[ll, kk])
                ##
            ##
            ax_temp.set_ylabel(ylabs[ii].upper(), labelpad = labelpads[ii])
            ## If ylims specified as an (n_ind, 2) array then include this
            if ylims is not None:
                if np.any(ylims[ii,:] is not None):
                    ax_temp.set_ylim(ylims[ii, 0], ylims[ii, 1])
            if yticks is not None:
                if yticks[ii] is not None:
                    ax_temp.set_yticks(yticks[ii])
            ## x labels only on the bottom row
            if rowcol_show[ii][0] == nrows - 1:
                ax_temp.set_xlabel(r'Coupling ${\lambda}$')
            ##
        ##
        ## Set 'axis = off' for remaining axes
        for ii in range(n_ax + len(skip_ax), nrows * ncols):
            ax[rowcol[ii]].axis('off')
        for ax_ind in skip_ax:
            ax[rowcol[ax_ind]].axis('off')
        ##
        ## Add axis to the bottom right plot
        if ii < nrows * ncols:
            ax_legend = ax[rowcol[-1]]
            handles, labels = ax[rowcol[0]].get_legend_handles_labels()
            ax_legend.legend(handles[::-1], labels[::-1], \
                fontsize = 'small', loc = 'center')
        else:
            ax_temp.legend(fontsize = 'x-small')
        ##
        if figpad is None:
            plt.tight_layout()
        else:
            plt.tight_layout(pad = figpad[0], \
                h_pad = figpad[1], w_pad = figpad[2])
        pdf.savefig(fig)
        plt.close()
    ## end function sim_plot4
##
##
##
##
##############################################################
##############################################################
##############################################################
## Supplementary figure plots
##
n_runs = 10
lambda_vals = np.arange(0, 1 + 0.01, 0.01)
## NOTE: lambda_vals_hb is different here than from simulated_data_log.py (only
## the first two elements of the tuple)
lambda_vals_hb = (np.arange(0, 0.41, 0.01), np.arange(0, 0.41, 0.01))
n_lambda = len(lambda_vals)
n_lambda_hb = np.prod([len(x) for x in lambda_vals_hb])
indices = ['te', 'ete', 'te-ksg', 'ctir', 'egc', 'nlgc', 'pi', 'si1']
indices = indices + ['si2', 'ccm']
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
file_dir = 'simulation-data/'
plot_dir = 'figures/'
##
indices_plot1 = ['TE (H)', 'ETE (H)', 'TE (KSG)', 'CTIR', 'EGC', 'NLGC']
indices_plot1 = indices_plot1 + ['PI', r'SI$^2$', r'SI$^3$', 'CCM']
indices_plot2 = ['TE (H)', 'ETE (H)', 'TE (KSG)', 'CTIR', 'EGC', 'NLGC']
indices_plot2 = indices_plot2 + ['PI', 'EGC*', r'SI$^2$', r'SI$^3$', 'CCM']
plot_params = {'lambda_vals': lambda_vals, 'skip_ax': [7], \
    'ylabs': indices_plot1, 'figsize': [10, 7], 'figpad': [1, 0.1, 0.01]}
plot_params_egc = {'lambda_vals': lambda_vals, 'skip_ax': list(), \
    'ylabs': indices_plot2, 'figsize': [10, 7], 'figpad': [1, 0.1, 0.01]}
plot_params_hb = {'lambda_vals': lambda_vals_hb, \
    'skip_ax': [7], 'transpose': True, 'cmap': dv.RdYlBu_11_r.mpl_colormap, \
    'titles': indices_plot1, 'figsize': [11, 7], 'figpad': [1, 0.1, 0.01]}
##
##############################################################
## LINEAR PROCESS
lp_shape = (n_runs, n_lambda, 2 * n_inds), (n_runs, n_lambda, n_time)
lp_results = load_reshape(file_dir + 'lp_values', lp_shape[0])
lp_time = load_reshape(file_dir + 'lp_time', lp_shape[1])
##
lp_params_gaussian = {'b_x': 0.8, 'b_y': 0.4, 'var_x': 0.2, 'var_y': 0.2}
lp_te_gaussian = [te_gaussian(x, **lp_params_gaussian) for x in lambda_vals]
lp_te_gaussian = np.array(lp_te_gaussian)
lp_ctir_gaussian = np.array([ \
    ctir_gaussian(x, **lp_params_gaussian, tau_max = 20) for x in lambda_vals])
lp_as = [lp_te_gaussian, lp_te_gaussian, lp_te_gaussian, lp_ctir_gaussian, \
    None, None, None, None, None, None]
##
lp_cols = [ql.Paired_4.mpl_colors[x] for x in list([1, 3, 0, 2])]
lp_means = np.nanmean(lp_results, axis = 0).reshape(n_lambda, n_inds, 2)
lp_std = np.nanstd(lp_results, axis = 0).reshape(n_lambda, n_inds, 2)
labelpads = [None, None, None, -2, None, None, None, None, None, None]
##
sim_plot1(mean_vals = lp_means, std_vals = lp_std, **plot_params, \
    labelpads = labelpads, analytic_solutions = lp_as, cols = lp_cols, \
    filename = plot_dir + 'lp_figure')
##
##############################################################
## ULAM LATTICE
ul_shape = (n_runs, n_lambda, 4 * n_inds), (n_runs, n_lambda, 2 * n_time)
ul_results = load_reshape(file_dir + 'ul_values', ul_shape[0])
ul_time = load_reshape(file_dir + 'ul_time', ul_shape[1])
##
ul_means = np.nanmean(ul_results, axis = 0).reshape(n_lambda, 2 * n_inds, 2)
ul_std = np.nanstd(ul_results, axis = 0).reshape(n_lambda, 2 * n_inds, 2)
##
## PLOT 1: T = 10 ** 3
ul_means1 = ul_means[:,:10,:]
ul_std1 = ul_std[:,:10,:]
##
ylim1 = np.array([[-0.1, 1.25], [-0.1, 1.25], [-0.2, 4.5], [-14.5, 2], \
    [-0.25, 1.05], [-0.1, 2.1], [-0.15, 1.15], \
    [-3.95, 7.25], [None, None], [-0.05, 1.05]])
labelpads1 = [None, None, None, -2, -2, None, -1, None, None, None]
##
ul_cols = [ql.Paired_4.mpl_colors[x] for x in list([1, 3])]
sim_plot1(mean_vals = ul_means1, std_vals = ul_std1, **plot_params, \
    cols = ul_cols, ylims = ylim1, labelpads = labelpads1, \
    filename = plot_dir + 'ul_figure1a')
##
## PLOT 2: T = 10 ** 5
ind_exclude = [16, 17, 18, 19, 81, 82, 83, 84]
ind_include = [ii for ii in range(n_lambda) if ii not in ind_exclude]
ul_means2 = np.zeros((n_lambda, n_inds + 1, 2))
ul_means2[:,:7,:] = ul_means[:,10:17,:]
ul_means2[:,8:,:] = ul_means[:,17:,:]
ul_means2[:,7,:] = ul_means[:,14,:]
ul_means2[ind_exclude,7,:] = np.nan
##
ul_std2 = np.zeros((n_lambda, n_inds + 1, 2))
ul_std2[:,:7,:] = ul_std[:,10:17,:]
ul_std2[:,8:,:] = ul_std[:,17:,:]
ul_std2[:,7,:] = ul_std[:,14,:]
ul_std2[ind_exclude,7,:] = np.nan
##
ylim2 = np.array([[-0.1, 1.25], [-0.1, 1.25], [-0.2, 4.5], [-14.5, 2], \
    [None, None], [-0.1, 2.1], [-0.15, 1.15], [-0.25, 1.05], \
    [-3.95, 7.25], [None, None], [-0.05, 1.05]])
labelpads2 = [None, None, None, -5, -2, None, None, -2, -3, -9.5, None]
##
sim_plot1(mean_vals = ul_means2, std_vals = ul_std2, **plot_params_egc, \
    cols = ul_cols, ylims = ylim2, labelpads = labelpads2, \
    filename = plot_dir + 'ul_figure2b')
##
##############################################################
## HENON UNIDIRECTIONAL
hu_shape = (n_runs, n_lambda, 3 * n_inds), (n_runs, n_lambda, 3 * n_time)
hu_results = load_reshape(file_dir + 'hu_values', hu_shape[0])
hu_time = load_reshape(file_dir + 'hu_time', hu_shape[1])
##
hu_reshape = (n_lambda, n_inds, 3)
hu_means = np.nanmean(hu_results, axis = 0).reshape(hu_reshape, order = 'F')
hu_std = np.nanstd(hu_results, axis = 0).reshape(hu_reshape, order = 'F')
##
hu_means1 = np.zeros((n_lambda, n_inds + 1, 3))
hu_means1[:,:7,:] = hu_means[:,:7,:]
hu_means1[:,8:,:] = hu_means[:,7:,:]
hu_means1[:,7,:] = hu_means[:,4,:]
hu_means1[70:,7,:] = np.nan
##
hu_std1 = np.zeros((n_lambda, n_inds + 1, 3))
hu_std1[:,:7,:] = hu_std[:,:7,:]
hu_std1[:,8:,:] = hu_std[:,7:,:]
hu_std1[:,7,:] = hu_std[:,4,:]
hu_std1[70:,7,:] = np.nan
##
labelpads = [None, None, None, None, -4, -6, 1, None, None, None, None]
hu_cols = ql.Dark2_3.mpl_colors[:3]
hu_ls = [(0, (1, 0)), (0, (1, 1)), (0, (5, 2))]
##
sim_plot2(mean_vals = hu_means1, std_vals = hu_std1, **plot_params_egc, \
    cols = hu_cols, linestyles = hu_ls, labelpads = labelpads, \
    filename = plot_dir + 'hu_figure')
##
##############################################################
## HENON BIDIRECTIONAL
hb_shape = (n_runs, n_lambda_hb * 2, n_inds), (n_runs, n_lambda_hb * 2, n_time)
hb_results = load_reshape(file_dir + 'hb_values', hb_shape[0])
hb_time = load_reshape(file_dir + 'hb_time', hb_shape[1])
##
hb_means = np.nanmean(hb_results, axis = 0)
n_lambda_hb1 = len(lambda_vals_hb[0])
hb_means1 = hb_means[:n_lambda_hb,:].reshape(n_lambda_hb1, n_lambda_hb1, n_inds)
hb_means2 = hb_means[n_lambda_hb:,:].reshape(n_lambda_hb1, n_lambda_hb1, n_inds)
##
def hb_vlims(vals, percentiles1 = [5, 95], percentiles2 = [1, 99]):
    n_inds = vals.shape[2]
    vlims = np.zeros((n_inds, 2))
    for ii in range(n_inds):
        percentiles = np.nanpercentile(vals[:, :, ii], percentiles1)
        vlim = np.max(np.abs(percentiles) + np.diff(percentiles))
        minmax = np.abs(np.nanpercentile(vals[:, :, ii], percentiles2))
        vlims[ii, :] = np.min((np.max(minmax), vlim)) * np.array([-1, 1])
    return vlims
##
def hb_vlims(vals, percentiles = [5, 95]):
    n_inds = vals.shape[2]
    vlims = np.zeros((n_inds, 2))
    for ii in range(n_inds):
        # percentiles = np.nanpercentile(vals[:, :, ii], percentiles)
        # vlim = np.max(np.abs(percentiles) + np.diff(percentiles))
        minmax = np.abs(np.nanpercentile(vals[:, :, ii], percentiles))
        vlims[ii, :] = np.min(np.max(minmax)) * np.array([-1, 1])
    return vlims
##
vlims1 = hb_vlims(hb_means1, percentiles = [1, 99])
vlims2 = hb_vlims(hb_means2, percentiles = [5, 95])
sim_plot3(vals = hb_means1, **plot_params_hb, vlims = vlims1, \
    filename = plot_dir + 'hb_figure1a')
sim_plot3(vals = hb_means2, **plot_params_hb, vlims = vlims2, \
    filename = plot_dir + 'hb_figure2a')
##
##
##############################################################
## ULAM LATTICE, TRANSFORMATIONS
tf_list = np.array([{'y_to_x': True}, {'y_to_x': True, 'scale_x': 2}, \
    {'normalise': True}, {'scale_x': 10, 'scale_y': 1}, \
    {'scale_x': 1, 'scale_y': 10}, {'round_x': 1}, \
    {'round_y': 1}, {'round_x': 2, 'round_y': 2}, \
    {'na_x': 10, 'na_y': 0}, {'na_x': 0, 'na_y': 10}, \
    {'na_x': 10, 'na_y': 10}, {'na_x': 20, 'na_y': 20}, \
    {'gaussian_x': 0.1, 'gaussian_y': 0.1}, \
    {'gaussian_x': 1}, {'gaussian_y': 1}])
n_tf = len(tf_list)
tf_split = [2, 3, 4, 4, 3]
tf_split1 = [0, 2, 5, 9, 13, 16]
##
ult_shape = (n_runs, n_lambda, 2 * n_inds, n_tf), \
    (n_runs, n_lambda, n_time, n_tf)
ult_results = load_reshape(file_dir + 'ult_values', ult_shape[0])
# ult_results = ult_results[:,:,:,np.append(np.arange(2,15), np.arange(2))]
ult_split = [0, 2, 3, 3, 4, 3]
##
ult_means = np.nanmean(ult_results, axis = 0).reshape(n_lambda, n_inds, 2, n_tf)
ult_std = np.nanstd(ult_results, axis = 0).reshape(n_lambda, n_inds, 2, n_tf)
##
ul_means1n = ul_means1[:,:,:,np.newaxis]
ul_std1n = ul_std1[:,:,:,np.newaxis]
##
def ult_cols(n, type = 'diverging', cbrewer = None, seq = None):
    if type == 'diverging':
        if seq is None:
            seq = np.linspace(0, 1, 2 * n)
            seq1 = seq[:n]
            seq2 = seq[n:]
        seq1 = np.linspace(seq[0], seq[1], n)
        seq2 = np.flip(np.linspace(seq[2], seq[3], n))
        if cbrewer is None:
            cbrewer = dv.RdBu_11_r
        cols = np.stack([lcm(cbrewer.mpl_colormap(seq1)).colors, \
            lcm(cbrewer.mpl_colormap(seq2)).colors])
    elif type == 'qualitative':
        if cbrewer is None:
            cbrewer = ql.Set1_9
        if n > 9:
            raise ValueError('Max value of n for type "qualitative" is 9')
        cols = np.array(cbrewer.colors)[np.newaxis, :n] / 256
    return cols
##
ixy_str = r'i$_{X\rightarrow Y}$'
iyx_str = r'i$_{Y\rightarrow X}$'
ls = [(0, (1, 0)), (0, (1, 1)), (0, (5, 2)), (0, (5, 2, 1, 2))]
##
def ult_ylims(ult_means, ult_std, pad = 0.05):
    ymax = np.nanmax(ult_means + ult_std, axis = (0, 2, 3))
    ymin = np.nanmin(ult_means - ult_std, axis = (0, 2, 3))
    ymax = ymax + pad * (ymax - ymin)
    ymin = ymin + pad * (ymin - ymax)
    return np.stack((ymin, ymax), axis = 1)
##
ul_means_ylim = ul_means.copy()
ul_means_ylim[[17,18,81,82],14,:] = np.nan
ul_shape1 = n_lambda, n_inds, 2, 2
ul_means_ylim = ul_means_ylim.reshape(ul_shape1, order = 'f').swapaxes(2,3)
ul_std_ylim = ul_std.reshape(ul_shape1, order = 'f').swapaxes(2,3)
ylims1 = ult_ylims( \
    np.insert(ult_means[:,:,:,:13], [0], ul_means_ylim, axis = 3), \
    np.insert(ult_std[:,:,:,:13], [0], ul_std_ylim, axis = 3))
ult_means_ylim = ult_means.copy()
ult_means_ylim[:,5:7,:,:3] = np.nan
ylims2 = ult_ylims( \
    np.insert(ult_means_ylim[:,:,:,:13], [0], ul_means_ylim, axis = 3),  \
    np.insert(ult_std[:,:,:,:13], [0], ul_std_ylim, axis = 3))
##
##
##
ult_means1 = np.insert(ult_means[:,:,:,:3], [0], ul_means1n, axis = 3)
ult_std1 = np.insert(ult_std[:,:,:,:3], [0], ul_std1n, axis = 3)
tf_names1 = np.array([ixy_str, iyx_str, \
    ixy_str + ': Standardised', iyx_str + ': Standardised',
    r'i$_{10X\rightarrow Y}$', r'i$_{Y\rightarrow 10X}$',
    r'i$_{X\rightarrow 10Y}$', r'i$_{10Y\rightarrow X}$']).reshape(-1, 2)
labelpads = [None, None, None, -8, None, -2, -4, -4, -4, None]
sim_plot4(ult_means1, ult_std1, **plot_params, tf_names = tf_names1, \
    cols = ult_cols(4, seq = [0.1, 0.3, 0.7, 0.9]), linestyles = ls[:4], \
    labelpads = labelpads, ylims = ylims1, filename = plot_dir + 'ult_figure1a')
##
ult_means2 = np.insert(ult_means[:,:,:,3:6], [0], ul_means1n, axis = 3)
ult_std2 = np.insert(ult_std[:,:,:,3:6], [0], ul_std1n, axis = 3)
tf_names2 = np.array([ixy_str, iyx_str,
    ixy_str + ': X to 1dp', iyx_str + ': X to 1dp',
    ixy_str + ': Y to 1dp', iyx_str + ': Y to 1dp',
    ixy_str + ': X, Y to 2dp', iyx_str + ': X, Y to 2dp']).reshape(-1, 2)
labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
sim_plot4(ult_means2, ult_std2, **plot_params, tf_names = tf_names2, \
    cols = ult_cols(4, seq = [0.1, 0.3, 0.7, 0.9]), linestyles = ls[:4], \
    labelpads = labelpads, ylims = ylims2, filename = plot_dir + 'ult_figure2')
##
ult_means3 = np.insert(ult_means[:,:,:,8:10], [0], ul_means1n, axis = 3)
ult_std3 = np.insert(ult_std[:,:,:,8:10], [0], ul_std1n, axis = 3)
tf_names3 = np.array([ixy_str, iyx_str,
    ixy_str + ': X, Y 10% NA', iyx_str + ': X, Y 10% NA',
    ixy_str + ': X, Y 20% NA', iyx_str + ': X, Y 20% NA']).reshape(-1, 2)
# r'i$_{X\rightarrow Y}$: X 10% NA', r'i$_{Y\rightarrow X}$: X 10% NA', \
# r'i$_{X\rightarrow Y}$: Y 10% NA', r'i$_{Y\rightarrow X}$: Y 10% NA', \
labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
sim_plot4(ult_means3, ult_std3, **plot_params, tf_names = tf_names3, \
    cols = ult_cols(3, seq = [0.1, 0.3, 0.7, 0.9]), linestyles = ls[:3], \
    labelpads = labelpads, ylims = ylims2, filename = plot_dir + 'ult_figure3')
##
ult_means4 = np.insert(ult_means[:,:,:,10:13], [0], ul_means1n, axis = 3)
ult_std4 = np.insert(ult_std[:,:,:,10:13], [0], ul_std1n, axis = 3)
tf_names4 = np.array([ixy_str, iyx_str, \
    ixy_str + r': $\sigma^2_G$ = 0.1', \
    iyx_str + r': $\sigma^2_G$ = 0.1', \
    ixy_str + r': $\sigma^2_{G,X}$ = 1', \
    iyx_str + r': $\sigma^2_{G,X}$ = 1', \
    ixy_str + r': $\sigma^2_{G,Y}$ = 1', \
    iyx_str + r': $\sigma^2_{G,Y}$ = 1']).reshape(-1, 2)
yticks = [None, np.arange(0, 1.1, step = 0.2), None, \
    np.arange(-12.5, 1, step = 2.5), np.arange(-8, 3, step = 2), None, None, \
    None, None, np.arange(0, 1.1, step = 0.2)]
labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
sim_plot4(ult_means4, ult_std4, **plot_params, tf_names = tf_names4, \
    cols = ult_cols(4, seq = [0.1, 0.3, 0.7, 0.9]), linestyles = ls[:4], \
    labelpads = labelpads, ylims = ylims2, yticks = yticks, \
    filename = plot_dir + 'ult_figure4')
##
ult_means5 = ult_means[:,:,0,13:].reshape(n_lambda, n_inds, 1, 2)
ult_std5 = ult_std[:,:,0,13:].reshape(n_lambda, n_inds, 1, 2)
tf_names5 = np.array([r'X$\rightarrow$X', r'X$\rightarrow$2X']).reshape(-1, 1)
labelpads = [-4, None, None, None, -3, -2, -2, -2, -2, None]
sim_plot4(ult_means5, ult_std5, **plot_params, tf_names = tf_names5, \
    cols = ult_cols(2, type = 'qualitative'), linestyles = ls[:2], \
    labelpads = labelpads, filename = plot_dir + 'ult_figure5a')
##
tf_names6 = np.array([ixy_str + r': $T = 10^3$', iyx_str + r': $T = 10^3$', \
    ixy_str + r': $T = 10^5$', iyx_str + r': $T = 10^5$']).reshape(-1, 2)
labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
sim_plot4(ul_means_ylim, ul_std_ylim, **plot_params, tf_names = tf_names6, \
    cols = ult_cols(2, seq = [0.1, 0.3, 0.7, 0.9]), linestyles = ls[:2], \
    labelpads = labelpads, ylims = ylims2, filename = plot_dir + 'ult_figure6')
##
##
ind_exclude = [16, 17, 18, 19, 81, 82, 83, 84]
ind_include = [ii for ii in range(n_lambda) if ii not in ind_exclude]
ultd = ult_results[:,:,range(0, 2 * n_inds, 2),:] - \
    ult_results[:,:,range(1, 2 * n_inds, 2),:]
ultd = ultd[:,ind_include,:,:]
ultd1 = ult_results[:,:,list(range(0, 2 * n_inds, 2)) + \
    list(range(1, 2 * n_inds, 2)),13:]
ultd1 = ultd1[:,ind_include,:,:]
uld = ul_results[:,:,range(0, 4 * n_inds, 2)] - \
    ul_results[:,:,range(1, 4 * n_inds, 2)]
uld1 = uld[:,ind_include,:n_inds]
uld2 = uld[:,ind_include,n_inds:]
ult_table = np.zeros((n_inds, 2 * n_tf + 2))
denom = np.nanmean(np.abs(np.nanmean(uld1, axis = 0)), axis = 0)
ult_table[:,0] = np.nanmean(uld1, axis = (0, 1))
ult_table[:,1] = np.nanmean((uld1 - uld2), axis = (0, 1)) / denom
ult_table[:,2:8] = np.nanmean(uld1[:,:,:,np.newaxis] - ultd[:,:,:,:6], \
    axis = (0, 1)) / denom[:,np.newaxis]
ult_table[:,8:13] = np.nanmean(uld1[:,:,:,np.newaxis] - ultd[:,:,:,8:13], \
    axis = (0, 1)) / denom[:,np.newaxis]
ult_table[:,13] = np.nanmean(ultd1[:,:,:n_inds, 0], axis = (0, 1))
ult_table[:,14] = np.nanmean(ultd1[:,:,:n_inds, 1], axis = (0, 1))
ult_table[:,15] = np.nanmean(ultd1[:,:,n_inds:, 1], axis = (0, 1))
ult_table[:,16] = np.nanmean(np.nanstd(uld1, axis = 0), axis = 0)
ult_table[:,17] = np.nanmean(np.nanstd(uld2, axis = 0), axis = 0) \
    / ult_table[:,16]
ult_table[:,18:24] = np.nanmean(np.nanstd(ultd[:,:,:,:6], axis = 0), axis = 0) \
    / ult_table[:,16,np.newaxis]
ult_table[:,24:29] = np.nanmean(np.nanstd(ultd[:,:,:,8:13], \
    axis = 0), axis = 0) / ult_table[:,16,np.newaxis]
ult_table[:,29] = np.nanmean(np.nanstd(ultd1[:,:,:n_inds,0],axis = 0), axis = 0)
ult_table[:,30] = np.nanmean(np.nanstd(ultd1[:,:,:n_inds,1],axis = 0), axis = 0)
ult_table[:,31] = np.nanmean(np.nanstd(ultd1[:,:,n_inds:,1],axis = 0), axis = 0)
str_table = ''
for ii in range(n_inds):
    str_table = str_table + indices[ii] + '\n'
    for jj in range(2 * n_tf + 2):
        str_table = str_table + str(np.round(ult_table[ii,jj], 3))
        if jj in [0,7,15,16,23]:
            str_table = str_table + '\n'
        else:
            str_table = str_table + ' & '
    str_table = str_table[:-2] + '\n\n\n'
##
##
ult_all = ultd[:,:,:,list(range(6)) + list(range(8,13))]
ult_all = np.insert(ult_all, [0], np.stack((uld1, uld2), axis = 3), axis = 3)
ult_corr_array = np.zeros((n_tf - 2, n_tf - 2, n_inds))
for ii in range(n_inds):
    z = ult_all[:,:,ii,:]
    pe_corr = pd.DataFrame(z.reshape(-1, n_tf - 2)).corr(method = 'pearson')
    sp_corr = pd.DataFrame(z.reshape(-1, n_tf - 2)).corr(method = 'spearman')
    ult_corr_array[:, :, ii] = np.array(pe_corr)
    mask = np.triu_indices(n_tf - 2)
    ult_corr_array[:, :, ii][mask] = np.array(sp_corr)[mask]
ult_corr_ylabs = np.array([r'$T = 10^3$', r'$T = 10^5$', 'Stand.', \
    r'$D_{10X\rightarrow Y}$', r'$D_{X\rightarrow 10Y}$', r'$X$ to 1dp', \
    r'$Y$ to 1dp', r'$X$, $Y$ to 2dp', '10% NA', '20% NA', \
    r'$\sigma^2_G$ = 0.1', r'$\sigma^2_{G,X}$ = 1', \
    r'$\sigma^2_{G,Y}$ = 1'])
##
##############################################################
##############################################################
##############################################################
## CORRELATIONS BETWEEN INDICES
##
jjs = np.hstack((np.arange(10) * n_lambda, \
    n_lambda_hb + n_lambda * 9, n_lambda_hb * 2 + n_lambda * 9))
all_results = np.zeros((n_runs, 9 * n_lambda + 2 * n_lambda_hb, n_inds))
all_results[:,jjs[0]:jjs[1],:] = lp_results[:,:,range(0,2 * n_inds,2)]
all_results[:,jjs[1]:jjs[2],:] = lp_results[:,:,range(1,2 * n_inds,2)]
all_results[:,jjs[2]:jjs[3],:] = ul_results[:,:,range(0,2 * n_inds,2)]
all_results[:,jjs[3]:jjs[4],:] = ul_results[:,:,range(1,2 * n_inds,2)]
all_results[:,jjs[4]:jjs[5],:] = ul_results[:,:,range(2 * n_inds,4 * n_inds,2)]
all_results[:,jjs[5]:jjs[6],:] = \
    ul_results[:,:,range(2 * n_inds + 1,4 * n_inds,2)]
all_results[:,jjs[6]:jjs[7],:] = hu_results[:,:,range(0,n_inds)]
all_results[:,jjs[7]:jjs[8],:] = hu_results[:,:,range(n_inds,2 * n_inds)]
all_results[:,jjs[8]:jjs[9],:] = hu_results[:,:,range(2 * n_inds,3 * n_inds)]
all_results[:,jjs[9]:jjs[11],:] = hb_results
##
## Remove anomalous results (as above)
ind_exclude = [16, 17, 18, 19, 81, 82, 83, 84]
all_results[:,jjs[2] + ind_exclude, :] = np.nan
all_results[:,jjs[3] + ind_exclude, :] = np.nan
all_results[:,jjs[4] + ind_exclude, :] = np.nan
all_results[:,jjs[5] + ind_exclude, :] = np.nan
# all_results[:,jjs[4] + [17, 18, 81, 82], 4] = np.nan
# all_results[:,jjs[5] + [17, 18, 81, 82], 4] = np.nan
# all_results[:,jjs[4] + 17:jjs[4] + 19,4] = np.nan
# all_results[:,jjs[4] + 81:jjs[4] + 83,4] = np.nan
# all_results[:,jjs[5] + 17:jjs[5] + 19,4] = np.nan
# all_results[:,jjs[5] + 81:jjs[5] + 83,4] = np.nan
all_results[:,jjs[6] + 70:jjs[7],4] = np.nan
all_results[:,jjs[7] + 70:jjs[8],4] = np.nan
all_results[:,jjs[8] + 70:jjs[9],4] = np.nan
# all_results[:,np.isnan(all_results).all(axis = 0).any(axis = 1),:] = np.nan
z = all_results[:,jjs[9]:jjs[11],:]
z[:,np.isnan(z).all(axis = 0).any(axis = 1),:] = np.nan
all_results[:,jjs[9]:jjs[11],:] = z
all_results[:,np.isnan(all_results).all(axis = 2).any(axis = 0),:] = np.nan
for ii in range(10):
    z = all_results[:,jjs[9]:jjs[10],ii]
    percentiles = np.nanpercentile(np.nanmean(z, axis = 0), [5, 95])
    vlim = np.max(np.abs(percentiles) + np.diff(percentiles))
    z[z > vlim] = np.nan
    z[z < -vlim] = np.nan
    all_results[:,jjs[9]:jjs[10],ii] = z
    z = all_results[:,jjs[10]:jjs[11],ii]
    percentiles = np.nanpercentile(np.nanmean(z, axis = 0), [5, 95])
    vlim = np.max(np.abs(percentiles) + np.diff(percentiles))
    z[z > vlim] = np.nan
    z[z < -vlim] = np.nan
    all_results[:,jjs[10]:jjs[11],ii] = z
##
##
sims = [r'LP, $T = 10^4,~i_{X \rightarrow Y}$', \
    r'LP, $T = 10^4,~i_{Y \rightarrow X}$', \
    r'UL, $T = 10^3,~i_{X \rightarrow Y}$', \
    r'UL, $T = 10^3,~i_{Y \rightarrow X}$', \
    r'UL, $T = 10^5,~i_{X \rightarrow Y}$', \
    r'UL, $T = 10^5,~i_{Y \rightarrow X}$', \
    r'HU, $T = 10^3,~D_{X \rightarrow Y}$', \
    r'HU, $T = 10^4,~D_{X \rightarrow Y}$', \
    r'HU, $T = 10^5,~D_{X \rightarrow Y}$', \
    r'HB(I), $T = 10^4,~D_{X \rightarrow Y}$', \
    r'HB(NI), $T = 10^4,~D_{X \rightarrow Y}$', \
    r'Mean, $D_{X \rightarrow Y}$']
corr_array = np.zeros((n_inds, n_inds, len(jjs)))
for ii in range(len(jjs) - 1):
    z = all_results[:,jjs[ii]:jjs[ii + 1],:]
    c = pd.DataFrame(z.reshape(-1,n_inds)).corr(method = 'pearson')
    c = np.array(c)
    sp_corr = \
        np.array(pd.DataFrame(z.reshape(-1,n_inds)).corr(method = 'spearman'))
    c[np.triu_indices(n_inds)] = sp_corr[np.triu_indices(n_inds)]
    corr_array[:,:,ii] = c
corr_array_mean = np.zeros((n_inds, n_inds, 8))
for ii in range(3):
    z = all_results[:,jjs[2 * ii]:jjs[2 * ii + 1],:] - \
        all_results[:,jjs[2 * ii + 1]:jjs[2 * ii + 2],:]
    c = pd.DataFrame(z.reshape(-1,10)).corr(method = 'pearson')
    c = np.array(c)
    sp_corr = np.array(pd.DataFrame(z.reshape(-1,10)).corr(method = 'spearman'))
    c[np.triu_indices(n_inds)] = sp_corr[np.triu_indices(n_inds)]
    corr_array_mean[:,:,ii] = c
corr_array_mean[:,:,3:8] = corr_array[:,:,6:11]
corr_array[:,:,-1] = corr_array_mean[:,:,:-1].mean(axis = 2)
##
##
def corr_plots(corr_array, skip_ax = list(), ylabs = None, titles = None, \
        nrows = 3, ncols = 4, indices_groups = None, figsize = None, \
        cmap = None, figpad = None, fontsize = None, filename = 'corr_plot'):
    n_plots = corr_array.shape[2]
    n_x = corr_array.shape[0]
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_ax = np.min((nrows * ncols, n_plots))
    with PdfPages(filename + '.pdf') as pdf:
        if figsize is None:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols)
        else:
            fig, ax = plt.subplots(nrows = nrows, ncols = ncols, \
                figsize = figsize)
        for ii in range(n_ax):
            ##
            ax_temp = ax[rowcol_show[ii]]
            if cmap is None:
                cmap = 'RdYlGn'
            im = ax_temp.imshow(corr_array[:,:,ii], origin = 'upper', \
                    vmin = -1, vmax = 1, cmap = cmap)
            group_x = [-0.5, n_x - 0.5]
            ax_temp.plot(group_x, group_x, color = 'k', lw = 1.5)
            if indices_groups is not None:
                groups_cs = np.cumsum(indices_groups)
                for kk in range(len(groups_cs)):
                    y_vals = np.repeat(groups_cs[kk], 2) - 0.5
                    ax_temp.plot(group_x, y_vals, color = 'k', lw = 1)
                    ax_temp.plot(y_vals, group_x, color = 'k', lw = 1)
            ##
            ## Set title as the indices (rather than as ylabels as in
            ## the other plotting functions)
            if titles is not None:
                if fontsize is None:
                    ax_temp.set_title(titles[ii])
                else:
                    ax_temp.set_title(titles[ii], fontsize = fontsize[0])
            ##
            if (rowcol_show[ii][0] == nrows - 1) and (ylabs is not None):
                ax_temp.set_xticks(np.arange(n_x))
                ax_temp.set_xticklabels([x for x in ylabs])
                if fontsize is None:
                    plt.setp(ax_temp.get_xticklabels())
                else:
                    plt.setp(ax_temp.get_xticklabels(), fontsize = fontsize[1])
                ax_temp.tick_params('x', labelrotation = 90)
            else:
                ax_temp.tick_params('x', bottom = False, labelbottom = False)
            if (rowcol_show[ii][1] == 0) and (ylabs is not None):
                ax_temp.set_yticks(np.arange(n_x))
                ax_temp.set_yticklabels([x for x in ylabs])
                if fontsize is None:
                    plt.setp(ax_temp.get_yticklabels())
                else:
                    plt.setp(ax_temp.get_yticklabels(), fontsize = fontsize[1])
            else:
                ax_temp.tick_params('y', left = False, labelleft = False)
            ##
        ##
        ## Add axis to the bottom right plot
        for ii in range(n_ax + len(skip_ax), nrows * ncols):
            ax[rowcol[ii]].axis('off')
        for ax_ind in skip_ax:
            ax[rowcol[ax_ind]].axis('off')
        ##
        if figpad is None:
            plt.tight_layout()
        else:
            plt.tight_layout(pad = figpad[0], \
                h_pad = figpad[1], w_pad = figpad[2])
        ##
        plt.subplots_adjust(right = 0.95)
        cax = plt.axes([0.95, 0.1, 0.02, 0.85])
        cbar = plt.colorbar(im, cax = cax)
        if fontsize is not None:
            cax.tick_params(labelsize = fontsize[2])
        cbar.set_ticks([-1, 0, 1])
        ##
        pdf.savefig(fig)
        plt.close()
    return
##
corr_plots(corr_array, titles = sims, ylabs = indices_plot1, \
    indices_groups = [4, 3, 3], figsize = [10, 7], figpad = [0.6, 0.5, 0.1], \
    fontsize = [12, 9, 7], cmap = dv.PRGn_11.mpl_colormap, \
    filename = plot_dir + 'corr_plots')
##
##
def ult_corr_plot(corr_array, xlabs = None, ylabs = None, x_groups = None, \
        y_groups = None, figsize = None, cmap = None, figpad = None, \
        fontsize = None, filename = 'ult_corr_plot'):
    n_x = corr_array.shape[1]
    n_y = corr_array.shape[0]
    with PdfPages(filename + '.pdf') as pdf:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize = figsize)
        ##
        if cmap is None:
            cmap = 'RdYlGn'
        im = ax.imshow(corr_array, origin = 'upper', \
            vmin = -1, vmax = 1, cmap = cmap)
        x_lims = [-0.5, n_x - 0.5]
        y_lims = [-0.5, n_y - 0.5]
        if x_groups is not None:
            x_groups_cs = np.cumsum(x_groups)
            for kk in range(len(x_groups_cs)):
                x_vals = np.repeat(x_groups_cs[kk], 2) - 0.5
                ax.plot(x_vals, y_lims, color = 'k', lw = 1)
        if y_groups is not None:
            y_groups_cs = np.cumsum(y_groups)
            for kk in range(len(y_groups_cs)):
                y_vals = np.repeat(y_groups_cs[kk], 2) - 0.5
                ax.plot(x_lims, y_vals, color = 'k', lw = 1)
        ##
        ax.set_xticks(np.arange(n_x))
        ax.set_xticklabels([x for x in xlabs])
        if fontsize is not None:
            plt.setp(ax.get_xticklabels(), fontsize = fontsize[1])
        ax.tick_params('x', labelrotation = 90)
        ##
        ax.set_yticks(np.arange(n_y))
        ax.set_yticklabels([x for x in ylabs])
        if fontsize is not None:
            plt.setp(ax.get_yticklabels(), fontsize = fontsize[1])
        ## Add a colourbar to each subplot
        cbar = ax.figure.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
        if fontsize is not None:
            cbar.ax.tick_params(labelsize = fontsize[2])
        cbar.set_ticks([-1, 0, 1])
        ##
        if figpad is None:
            plt.tight_layout()
        else:
            plt.tight_layout(pad = figpad[0], \
                h_pad = figpad[1], w_pad = figpad[2])
        ##
        pdf.savefig(fig)
        plt.close()
    return
##
ult_corr_plot(ult_corr_array[0,:,:], ylabs = ult_corr_ylabs, \
    xlabs = indices_plot1, y_groups = [2, 3, 3, 2, 3], x_groups = [4, 3, 3], \
    figsize = [5, 5], fontsize = [12, 9, 7], figpad = [0.6, 0.1, 0.1], \
    cmap = dv.PRGn_11.mpl_colormap, filename = plot_dir + 'ult_corr_plots')
##
##
##############################################################
##############################################################
##############################################################
## COMPUTATION TIMES
##
all_time = np.zeros((n_time * 2, 8))
all_time[:n_time,0] = np.nanmean(lp_time, axis = (0, 1))
all_time[:n_time,1] = np.nanmean(ul_time[:,:,:n_time], axis = (0, 1))
all_time[:n_time,2] = np.nanmean(ul_time[:,:,n_time:], axis = (0, 1))
all_time[:n_time,3] = np.nanmean(hu_time[:,:,:n_time], axis = (0, 1))
all_time[:n_time,4] = np.nanmean(hu_time[:,:,n_time:(2 * n_time)],axis = (0, 1))
all_time[:n_time,5] = np.nanmean(hu_time[:,:,(2 * n_time):], axis = (0, 1))
all_time[:n_time,6] = np.nanmean(hb_time[:,:n_lambda_hb,:], axis = (0, 1))
all_time[:n_time,7] = np.nanmean(hb_time[:,n_lambda_hb:,:], axis = (0, 1))
all_time[n_time:,0] = np.nanstd(lp_time, axis = (0, 1))
all_time[n_time:,1] = np.nanstd(ul_time[:,:,:n_time], axis = (0, 1))
all_time[n_time:,2] = np.nanstd(ul_time[:,:,n_time:], axis = (0, 1))
all_time[n_time:,3] = np.nanstd(hu_time[:,:,:n_time], axis = (0, 1))
all_time[n_time:,4] = np.nanstd(hu_time[:,:,n_time:(2 * n_time)], axis = (0, 1))
all_time[n_time:,5] = np.nanstd(hu_time[:,:,(2 * n_time):], axis = (0, 1))
all_time[n_time:,6] = np.nanstd(hb_time[:,:n_lambda_hb,:], axis = (0, 1))
all_time[n_time:,7] = np.nanstd(hb_time[:,n_lambda_hb:,:], axis = (0, 1))
## order is EGC, NLGC, PI, ETE, TE-KSG, CTIR, SI, CCM
all_time = all_time[[3,4,5,0,1,2,6,7,11,12,13,8,9,10,14,15],:]
for jj in range(8):
    print('\n\n')
    for ii in range(n_time):
        print(str(np.round(all_time[ii,jj], 3)) + ' (' + \
            str(np.round(all_time[ii + n_time,jj], 3)) + ')')
str_time_table = ''
for ii in range(n_time):
    for jj in range(8):
        str_time_table = str_time_table + str(np.round(all_time[ii,jj], 3)) + \
            ' (' + str(np.round(all_time[ii + n_time,jj], 3)) + ') & '
    str_time_table = str_time_table[:-2] + '\n\n\n'
print(str_time_table)
##
##############################################################
## Transfer entropy from idtxl package
##
## idtxl package also calculates transfer entropy
# import numpy as np
# from idtxl.estimators_jidt import (JidtDiscreteMI, JidtDiscreteTE,
#                                    JidtKraskovMI, JidtKraskovTE)
# ##
# settings_h = {}
# settings_h['history_target'] = 1
# settings_h['history_source'] = 1
# settings_h['source_target_delay'] = 1
# settings_h['discretise_method'] = 'equal'
# settings_h['n_discrete_bins'] = 8
# settings_h['noise_level'] = 0
# ##
# settings_ksg = {}
# settings_ksg['history_target'] = 1
# settings_ksg['history_source'] = 1
# settings_ksg['source_target_delay'] = 1
# settings_ksg['algorithm_num'] = 1
# settings_ksg['kraskov_k'] = 4
# settings_ksg['noise_level'] = 0
# settings_ksg['normalise'] = False
# settings_ksg['local_values'] = False
# ##
# est_ksg = JidtKraskovTE(settings_ksg)
# te_ksg = est_ksg.estimate(x, y)
# est_h = JidtDiscreteTE(settings_h)
# te_h = est_h.estimate(x, y)
##
## NOTE: kmeans vs cmeans computational burden!
## kmeans: shape (100, 3), clusters 50, time 0.27243233360350133
## kmeans: shape (1000, 3), clusters 50, time 1.325215889001265
## kmeans: shape (10000, 3), clusters 50, time 49.92137239077128
## kmeans: shape (999998, 3), clusters 50, time 962.7682773035951
## cmeans: shape (100, 3), clusters 50, time 0.5133263552095741
## cmeans: shape (1000, 3), clusters 50, time 25.340735886571927
## cmeans: shape (10000, 3), clusters 50, time 189.5237478673924
## cmeans: shape (999998, 3), clusters 50, time 221623.170088802
