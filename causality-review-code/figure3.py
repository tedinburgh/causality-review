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
file_dir = '../simulation-data/'
plot_dir = '../figures/'
##
indices_plot1 = ['TE (H)', 'ETE (H)', 'TE (KSG)', 'CTIR', 'EGC', 'NLGC']
indices_plot1 = indices_plot1 + ['PI', r'SI$^{(1)}$', r'SI$^{(2)}$', 'CCM']
indices_plot2 = ['TE (H)', 'ETE (H)', 'TE (KSG)', 'CTIR', 'EGC', 'NLGC', 'PI']
indices_plot2 = indices_plot2 + ['EGC*', r'SI$^{(1)}$', r'SI$^{(2)}$', 'CCM']
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
