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
import argparse
##
##
##
##
##############################################################
##############################################################
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
##
##
##############################################################
##############################################################
##############################################################
## PLOTTING FUNCTIONS
## (to match Lungeralla et al 2006)
##
## Plot #1: (Figure 3)
## Used for simulations of linear processes
def sim_plot1(mean_vals, std_vals, lambda_vals, ylabs, ylims = None, \
        analytic_solutions = None, nrows = 3, ncols = 4, labelpads = None, \
        figpad = None, skip_ax = list(), figsize = None, cols = None, \
        filetype = 'eps', filename = 'ci_figure1'):
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
    with PdfPages(filename + '.pdf', keep_empty = False) as pdf:
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
        if filetype == 'eps':
            plt.savefig(filename + '.eps', format = 'eps', dpi = 300)
        elif filetype == 'pdf':
            pdf.savefig(fig)
        plt.close()
    ## end function sim_plot1
##
##
## Plot #2: (Figure 4b)
## Used for simulations of Henon unidirectional maps using the indices
## C_i = i_xy - i_yx where i is any of the causality measures
def sim_plot2(mean_vals, std_vals, lambda_vals, ylabs, ylims = None, \
        nrows = 3, ncols = 4, skip_ax = list(), figpad = None, figsize = None, \
        cols = None, linestyles = None, labelpads = None, \
        filetype = 'eps', filename = 'ci_figure2'):
    ##
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_inds = mean_vals.shape[1]
    n_ax = np.min((nrows * ncols, n_inds))
    if labelpads == None:
        labelpads = [None] * n_inds
    ##
    with PdfPages(filename + '.pdf', keep_empty = False) as pdf:
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
        if filetype == 'eps':
            plt.savefig(filename + '.eps', format = 'eps', dpi = 300)
        elif filetype == 'pdf':
            pdf.savefig(fig)
        plt.close()
    ## end function sim_plot2
##
##
## Plot #3: (Figure 5)
## Used for simulations of Henon bidirectional maps using the indices
## C_i = i_xy - i_yx where i is any of the causality measures
## This is a set of heatmaps rather than plots
def sim_plot3(vals, lambda_vals, titles, vlims = None, transpose = True, \
        vlim_percentiles = [1, 99], nrows = 3, ncols = 4, skip_ax = list(), \
        cmap = None, figpad = None, figsize = None, \
        filetype = 'eps', filename = 'ci_figure3'):
    ##
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_inds = vals.shape[2]
    n_ax = np.min((nrows * ncols, n_inds))
    extent = np.min(lambda_vals[0]), np.max(lambda_vals[0]), \
        np.min(lambda_vals[1]), np.max(lambda_vals[1])
    ##
    with PdfPages(filename + '.pdf', keep_empty = False) as pdf:
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
        if filetype == 'eps':
            plt.savefig(filename + '.eps', format = 'eps', dpi = 300)
        elif filetype == 'pdf':
            pdf.savefig(fig)
        plt.close()
    ## end function sim_plot3
##
##
## Plot #4: (Figure 4a, S2, S3)
## Used for simulations of Ulam lattice (allows multiple experiments on the
## same subplot)
def sim_plot4(mean_vals, std_vals, lambda_vals, ylabs, ylims = None, \
        cols = None, tf_names = None, nrows = 3, ncols = 4, labelpads = None, \
        figpad = None, skip_ax = list(), figsize = None, linestyles = None, \
        yticks = None, filetype = 'eps', filename = 'ci_figure4'):
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
    with PdfPages(filename + '.pdf', keep_empty = False) as pdf:
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
        if filetype == 'eps':
            plt.savefig(filename + '.eps', format = 'eps', dpi = 300)
        elif filetype == 'pdf':
            pdf.savefig(fig)
        plt.close()
    ## end function sim_plot4
##
##
def corr_plots(corr_array, skip_ax = list(), ylabs = None, titles = None, \
        nrows = 3, ncols = 4, indices_groups = None, figsize = None, \
        cmap = None, figpad = None, fontsize = None, \
        filetype = 'eps', filename = 'corr_plot'):
    n_plots = corr_array.shape[2]
    n_x = corr_array.shape[0]
    rowcol = [(x, y) for x in range(nrows) for y in range(ncols)]
    rowcol_show = [rowcol[ii] for ii in range(len(rowcol)) if ii not in skip_ax]
    n_ax = np.min((nrows * ncols, n_plots))
    with PdfPages(filename + '.pdf', keep_empty = False) as pdf:
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
        if filetype == 'eps':
            plt.savefig(filename + '.eps', format = 'eps', dpi = 300)
        elif filetype == 'pdf':
            pdf.savefig(fig)
        plt.close()
    return
    ## end function corr_plots
##
##
def corr_transforms_plot(corr_array, xlabs = None, ylabs = None, \
        x_groups = None, y_groups = None, figsize = None, cmap = None, \
        figpad = None, fontsize = None, \
        filetype = 'eps', filename = 'corr_transforms_plot'):
    ##
    n_x = corr_array.shape[1]
    n_y = corr_array.shape[0]
    ##
    with PdfPages(filename + '.pdf', keep_empty = False) as pdf:
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
        if filetype == 'eps':
            plt.savefig(filename + '.eps', format = 'eps', dpi = 300)
        elif filetype == 'pdf':
            pdf.savefig(fig)
        plt.close()
    return
##
##
##
##
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
##############################################################
##############################################################
##############################################################
## Argument parsing
## For input arguments when running the script (i.e. a subset of indices,
## verbose, logging time values)
parser = argparse.ArgumentParser( \
    description = 'Simulations for causality indices')
parser.add_argument('--figure', '--fig', default = 'all', dest = 'fig', \
    help = 'Figures to be generated')
## list of indices to compute (transfer entropy, coarse-grained transinformation
## rate, nonlinear Granger causality, extended Granger causality,
## predictability improvement, similarity indices)
parser.add_argument('--table', '--tab', default = 'all', dest = 'tab', \
    help = 'Tables to be printed')
##
args = parser.parse_args()
figures = args.fig
if figures == 'all':
    figures = ['lp', 'ul', 'hu', 'hb', 'ul-scaling', 'ul-rounding', \
        'ul-missing', 'ul-gaussian', 'corr-all', 'corr-ul-transforms']
else:
    figures = [figures]
tables = args.tab
if tables == 'all':
    tables = ['ul-transforms', 'computational-times']
else:
    tables = [tables]
##
##
##############################################################
##############################################################
##############################################################
## Shared figure values
##
file_dir = 'simulation-data/'
plot_dir = 'figures/'
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
lp_shape = (n_runs, n_lambda, 2 * n_inds), (n_runs, n_lambda, n_time)
ul_shape = (n_runs, n_lambda, 4 * n_inds), (n_runs, n_lambda, 2 * n_time)
hu_shape = (n_runs, n_lambda, 3 * n_inds), (n_runs, n_lambda, 3 * n_time)
hb_shape = (n_runs, n_lambda_hb * 2, n_inds), (n_runs, n_lambda_hb * 2, n_time)
##
##
##############################################################
##############################################################
##############################################################
## Correlations between methods (Figure 1)
##
if 'corr-all' in figures:
    jjs = np.hstack((np.arange(10) * n_lambda, \
        n_lambda_hb + n_lambda * 9, n_lambda_hb * 2 + n_lambda * 9))
    try:
        lp_results = load_reshape(file_dir + 'lp_values', lp_shape[0])
        ul_results = load_reshape(file_dir + 'ul_values', ul_shape[0])
        hu_results = load_reshape(file_dir + 'hu_values', hu_shape[0])
        hb_results = load_reshape(file_dir + 'hb_values', hb_shape[0])
    except Exception as err:
        print('At least one *_values.csv missing from ' + file_dir)
        raise
    ##
    all_results = np.zeros((n_runs, 9 * n_lambda + 2 * n_lambda_hb, n_inds))
    all_results[:,jjs[0]:jjs[1],:] = lp_results[:,:,range(0, 2 * n_inds, 2)]
    all_results[:,jjs[1]:jjs[2],:] = lp_results[:,:,range(1, 2 * n_inds, 2)]
    all_results[:,jjs[2]:jjs[3],:] = ul_results[:,:,range(0, 2 * n_inds, 2)]
    all_results[:,jjs[3]:jjs[4],:] = ul_results[:,:,range(1, 2 * n_inds, 2)]
    all_results[:,jjs[4]:jjs[5],:] = \
        ul_results[:,:,range(2 * n_inds, 4 * n_inds, 2)]
    all_results[:,jjs[5]:jjs[6],:] = \
        ul_results[:,:,range(2 * n_inds + 1, 4 * n_inds, 2)]
    all_results[:,jjs[6]:jjs[7],:] = hu_results[:,:,range(0, n_inds)]
    all_results[:,jjs[7]:jjs[8],:] = hu_results[:,:,range(n_inds, 2 * n_inds)]
    all_results[:,jjs[8]:jjs[9],:] = \
        hu_results[:,:,range(2 * n_inds, 3 * n_inds)]
    all_results[:,jjs[9]:jjs[11],:] = hb_results
    ##
    ## Remove anomalous results (as above)
    ind_exclude = [16, 17, 18, 19, 81, 82, 83, 84]
    all_results[:,jjs[2] + ind_exclude, :] = np.nan
    all_results[:,jjs[3] + ind_exclude, :] = np.nan
    all_results[:,jjs[4] + ind_exclude, :] = np.nan
    all_results[:,jjs[5] + ind_exclude, :] = np.nan
    all_results[:,jjs[6] + 70:jjs[7],4] = np.nan
    all_results[:,jjs[7] + 70:jjs[8],4] = np.nan
    all_results[:,jjs[8] + 70:jjs[9],4] = np.nan
    ##
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
    corr_array = np.zeros((n_inds, n_inds, len(jjs)))
    for ii in range(len(jjs) - 1):
        z = all_results[:,jjs[ii]:jjs[ii + 1],:]
        c = pd.DataFrame(z.reshape(-1, n_inds)).corr(method = 'pearson')
        c = np.array(c)
        sp_corr = pd.DataFrame(z.reshape(-1, n_inds)).corr(method = 'spearman')
        sp_corr = np.array(sp_corr)
        c[np.triu_indices(n_inds)] = sp_corr[np.triu_indices(n_inds)]
        corr_array[:,:,ii] = c
    ##
    corr_array_mean = np.zeros((n_inds, n_inds, 8))
    for ii in range(3):
        z = all_results[:,jjs[2 * ii]:jjs[2 * ii + 1],:] - \
            all_results[:,jjs[2 * ii + 1]:jjs[2 * ii + 2],:]
        c = pd.DataFrame(z.reshape(-1, n_inds)).corr(method = 'pearson')
        c = np.array(c)
        sp_corr = pd.DataFrame(z.reshape(-1, n_inds)).corr(method = 'spearman')
        sp_corr = np.array(sp_corr)
        c[np.triu_indices(n_inds)] = sp_corr[np.triu_indices(n_inds)]
        corr_array_mean[:,:,ii] = c
    corr_array_mean[:,:,3:8] = corr_array[:,:,6:11]
    corr_array[:,:,-1] = corr_array_mean[:,:,:-1].mean(axis = 2)
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
    ##
    corr_plots(corr_array, titles = sims, ylabs = indices_plot1, \
        indices_groups = [4, 3, 3], figsize = [10, 7], \
        figpad = [0.6, 0.5, 0.1], fontsize = [12, 9, 7], \
        cmap = dv.PRGn_11.mpl_colormap, filename = plot_dir + 'corr_plots')
##
##############################################################
## LINEAR PROCESS (FIGURE 3)
if 'lp' in figures:
    try:
        lp_results = load_reshape(file_dir + 'lp_values', lp_shape[0])
    except Exception as err:
        print('lp_values.csv missing from ' + file_dir)
        raise
    ##
    lp_params_gaussian = {'b_x': 0.8, 'b_y': 0.4, 'var_x': 0.2, 'var_y': 0.2}
    lp_te_gaussian = [te_gaussian(x, **lp_params_gaussian) for x in lambda_vals]
    lp_te_gaussian = np.array(lp_te_gaussian)
    lp_ctir_gaussian = [ctir_gaussian(x, **lp_params_gaussian, tau_max = 20) \
        for x in lambda_vals]
    lp_ctir_gaussian = np.array(lp_ctir_gaussian)
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
## HENON UNIDIRECTIONAL (Figure 4b)
if 'hu' in figures:
    try:
        hu_results = load_reshape(file_dir + 'hu_values', hu_shape[0])
    except Exception as err:
        print('hu_values.csv missing from ' + file_dir)
        raise
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
## HENON BIDIRECTIONAL (Figure 5)
if 'hb' in figures:
    try:
        hb_results = load_reshape(file_dir + 'hb_values', hb_shape[0])
    except Exception as err:
        print('hb_values.csv missing from ' + file_dir)
        raise
    ##
    hb_means = np.nanmean(hb_results, axis = 0)
    n_lambda_hb1 = len(lambda_vals_hb[0])
    hb_means1 = hb_means[:n_lambda_hb,:]
    hb_means1 = hb_means1.reshape(n_lambda_hb1, n_lambda_hb1, n_inds)
    hb_means2 = hb_means[n_lambda_hb:,:]
    hb_means2 = hb_means2.reshape(n_lambda_hb1, n_lambda_hb1, n_inds)
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
            minmax = np.abs(np.nanpercentile(vals[:, :, ii], percentiles))
            vlims[ii, :] = np.min(np.max(minmax)) * np.array([-1, 1])
        return vlims
    ##
    vlims1 = hb_vlims(hb_means1, percentiles = [1, 99])
    vlims2 = hb_vlims(hb_means2, percentiles = [5, 95])
    sim_plot3(vals = hb_means1, **plot_params_hb, vlims = vlims1, \
        filename = plot_dir + 'hb_figure1')
    sim_plot3(vals = hb_means2, **plot_params_hb, vlims = vlims2, \
        filename = plot_dir + 'hb_figure2')
##
##
##############################################################
## ULAM LATTICE (Figure 4a)
##
def ul_cols(n, type = 'diverging', cbrewer = None, seq = None):
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
ul_ls = [(0, (1, 0)), (0, (1, 1)), (0, (5, 2)), (0, (5, 2, 1, 2))]
##
def ul_ylims(means, std, pad = 0.05):
    ymax = np.nanmax(means + std, axis = (0, 2, 3))
    ymin = np.nanmin(means - std, axis = (0, 2, 3))
    ymax = ymax + pad * (ymax - ymin)
    ymin = ymin + pad * (ymin - ymax)
    return np.stack((ymin, ymax), axis = 1)
##
##
if 'ul' in figures:
    try:
        ul_results = load_reshape(file_dir + 'ul_values', ul_shape[0])
    except Exception as err:
        print('ul_values.csv missing from ' + file_dir)
        raise
    ##
    ul_shape1 = (n_lambda, n_inds, 2, 2)
    ##
    ul_means = np.nanmean(ul_results, axis = 0).reshape(n_lambda, 2 * n_inds, 2)
    ul_std = np.nanstd(ul_results, axis = 0).reshape(n_lambda, 2 * n_inds, 2)
    ##
    ul_means[[17,18,81,82],14,:] = np.nan
    ul_means = ul_means.reshape(ul_shape1, order = 'f').swapaxes(2,3)
    ul_std = ul_std.reshape(ul_shape1, order = 'f').swapaxes(2,3)
    ##
    ylims = ul_ylims(ul_means, ul_std)
    ul_names = np.array([ixy_str + r': $T = 10^3$', iyx_str + r': $T = 10^3$', \
        ixy_str + r': $T = 10^5$', iyx_str + r': $T = 10^5$']).reshape(-1, 2)
    labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
    sim_plot4(ul_means, ul_std, **plot_params, tf_names = ul_names, \
        cols = ul_cols(2, seq = [0.1, 0.3, 0.7, 0.9]), \
        linestyles = ul_ls[:2], labelpads = labelpads, ylims = ylims, \
        filename = plot_dir + 'ul_figure')
##
##
##############################################################
## ULAM LATTICE, TRANSFORMATIONS (Figure S2, S3)
##
tf_list = np.array([{'normalise': True}, \
    {'scale_x': 10, 'scale_y': 1}, {'scale_x': 1, 'scale_y': 10},
    {'round_x': 1}, {'round_y': 1}, {'round_x': 2, 'round_y': 2}, \
    {'na_x': 10, 'na_y': 0}, {'na_x': 20, 'na_y': 20}, \
    {'gaussian_x': 0.1, 'gaussian_y': 0.1}, \
    {'gaussian_x': 1}, {'gaussian_y': 1}])
n_tf = len(tf_list)
##
ult_shape = (n_runs, n_lambda, 2 * n_inds, n_tf), \
    (n_runs, n_lambda, n_time, n_tf)
##
ul_transforms = ['ul-scaling', 'ul-rounding', 'ul-missing', 'ul-gaussian']
ul_transforms = ul_transforms + ['corr-ul-transforms']
if any(x in figures for x in ul_transforms):
    try:
        ul_results = load_reshape(file_dir + 'ul_values', ul_shape[0])
        ult_results = load_reshape(file_dir + 'ult_values', ult_shape[0])
    except Exception as err:
        print('\nul_values.csv or ult_values.csv missing from ' + file_dir)
        raise
    ##
    ul_shape1 = (n_lambda, n_inds, 2, 2)
    ##
    ul_means = np.nanmean(ul_results, axis = 0).reshape(n_lambda, 2 * n_inds, 2)
    ul_std = np.nanstd(ul_results, axis = 0).reshape(n_lambda, 2 * n_inds, 2)
    ##
    ul_means[[17,18,81,82],14,:] = np.nan
    ul_means = ul_means.reshape(ul_shape1, order = 'f').swapaxes(2,3)
    ul_std = ul_std.reshape(ul_shape1, order = 'f').swapaxes(2,3)
    ##
    ult_means = np.nanmean(ult_results, axis = 0)
    ult_means = ult_means.reshape(n_lambda, n_inds, 2, n_tf)
    ult_std = np.nanstd(ult_results, axis = 0)
    ult_std = ult_std.reshape(n_lambda, n_inds, 2, n_tf)
    ##
    ul_means1n = ul_means[:,:,:,:1]
    ul_std1n = ul_std[:,:,:,:1]
##
if 'ul-scaling' in figures:
    ylims = ul_ylims(np.insert(ult_means, [0], ul_means, axis = 3), \
        np.insert(ult_std, [0], ul_std, axis = 3))
    ult_means1 = np.insert(ult_means[:,:,:,:3], [0], ul_means1n, axis = 3)
    ult_std1 = np.insert(ult_std[:,:,:,:3], [0], ul_std1n, axis = 3)
    tf_names = np.array([ixy_str, iyx_str, \
        ixy_str + ': Standardised', iyx_str + ': Standardised',
        r'i$_{10X\rightarrow Y}$', r'i$_{Y\rightarrow 10X}$',
        r'i$_{X\rightarrow 10Y}$', r'i$_{10Y\rightarrow X}$']).reshape(-1, 2)
    labelpads = [None, None, None, -8, None, -2, -4, -4, -4, None]
    sim_plot4(ult_means1, ult_std1, **plot_params, tf_names = tf_names, \
        cols = ul_cols(4, seq = [0.1, 0.3, 0.7, 0.9]), \
        linestyles = ul_ls[:4], labelpads = labelpads, ylims = ylims, \
        filename = plot_dir + 'ul_scaling_figure')
##
if 'ul-rounding' in figures:
    ult_means_ylim = ult_means.copy()
    ult_means_ylim[:,5:7,:,1:3] = np.nan
    ylims = ul_ylims(np.insert(ult_means_ylim, [0], ul_means, axis = 3), \
        np.insert(ult_std, [0], ul_std, axis = 3))
    ult_means2 = np.insert(ult_means[:,:,:,3:6], [0], ul_means1n, axis = 3)
    ult_std2 = np.insert(ult_std[:,:,:,3:6], [0], ul_std1n, axis = 3)
    tf_names = np.array([ixy_str, iyx_str,
        ixy_str + ': X to 1dp', iyx_str + ': X to 1dp',
        ixy_str + ': Y to 1dp', iyx_str + ': Y to 1dp',
        ixy_str + ': X, Y to 2dp', iyx_str + ': X, Y to 2dp']).reshape(-1, 2)
    labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
    sim_plot4(ult_means2, ult_std2, **plot_params, tf_names = tf_names, \
        cols = ul_cols(4, seq = [0.1, 0.3, 0.7, 0.9]), \
        linestyles = ul_ls[:4], labelpads = labelpads, ylims = ylims, \
        filename = plot_dir + 'ul_rounding_figure')
##
if 'ul-missing' in figures:
    ult_means_ylim = ult_means.copy()
    ult_means_ylim[:,5:7,:,1:3] = np.nan
    ylims = ul_ylims(np.insert(ult_means_ylim, [0], ul_means, axis = 3), \
        np.insert(ult_std, [0], ul_std, axis = 3))
    ult_means3 = np.insert(ult_means[:,:,:,6:8], [0], ul_means1n, axis = 3)
    ult_std3 = np.insert(ult_std[:,:,:,6:8], [0], ul_std1n, axis = 3)
    tf_names = np.array([ixy_str, iyx_str,
        ixy_str + ': X, Y 10% NA', iyx_str + ': X, Y 10% NA',
        ixy_str + ': X, Y 20% NA', iyx_str + ': X, Y 20% NA']).reshape(-1, 2)
    labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
    sim_plot4(ult_means3, ult_std3, **plot_params, tf_names = tf_names, \
        cols = ul_cols(3, seq = [0.1, 0.3, 0.7, 0.9]), \
        linestyles = ul_ls[:3], labelpads = labelpads, ylims = ylims, \
        filename = plot_dir + 'ul_missing_figure')
##
if 'ul-gaussian' in figures:
    ult_means_ylim = ult_means.copy()
    ult_means_ylim[:,5:7,:,1:3] = np.nan
    ylims = ul_ylims(np.insert(ult_means_ylim, [0], ul_means, axis = 3), \
        np.insert(ult_std, [0], ul_std, axis = 3))
    ult_means4 = np.insert(ult_means[:,:,:,8:], [0], ul_means1n, axis = 3)
    ult_std4 = np.insert(ult_std[:,:,:,8:], [0], ul_std1n, axis = 3)
    tf_names = np.array([ixy_str, iyx_str, \
        ixy_str + r': $\sigma^2_G$ = 0.1', \
        iyx_str + r': $\sigma^2_G$ = 0.1', \
        ixy_str + r': $\sigma^2_{G,X}$ = 1', \
        iyx_str + r': $\sigma^2_{G,X}$ = 1', \
        ixy_str + r': $\sigma^2_{G,Y}$ = 1', \
        iyx_str + r': $\sigma^2_{G,Y}$ = 1']).reshape(-1, 2)
    yticks = [None, np.arange(0, 1.1, step = 0.2), None, \
        np.arange(-12.5, 1, step = 2.5), \
        np.arange(-8, 3, step = 2), None, None, \
        None, None, np.arange(0, 1.1, step = 0.2)]
    labelpads = [None, None, None, -8, None, None, None, -4, -4, None]
    sim_plot4(ult_means4, ult_std4, **plot_params, tf_names = tf_names, \
        cols = ul_cols(4, seq = [0.1, 0.3, 0.7, 0.9]), \
        linestyles = ul_ls[:4], labelpads = labelpads, ylims = ylims, \
        yticks = yticks, filename = plot_dir + 'ul_gaussian_figure')
##
##
if 'corr-ul-transforms' in figures:
    ind_exclude = [16, 17, 18, 19, 81, 82, 83, 84]
    ind_include = [ii for ii in range(n_lambda) if ii not in ind_exclude]
    ##
    ult_D = ult_results[:,:,range(0, 2 * n_inds, 2),:] - \
        ult_results[:,:,range(1, 2 * n_inds, 2),:]
    ult_D = ult_D[:,ind_include,:,:]
    ##
    ul_D = ul_results[:,:,range(0, 4 * n_inds, 2)] - \
        ul_results[:,:,range(1, 4 * n_inds, 2)]
    ul_D = np.stack((ul_D[:,:,:n_inds], ul_D[:,:,n_inds:]), axis = 3)
    ul_D = ul_D[:,ind_include,:,:]
    ##
    ult_D = np.insert(ult_D, [0], ul_D, axis = 3)
    ult_corr_array = np.zeros((n_tf + 2, n_tf + 2, n_inds))
    for ii in range(n_inds):
        z = ult_D[:,:,ii,:]
        pe_corr = pd.DataFrame(z.reshape(-1, n_tf + 2)).corr(method = 'pearson')
        sp_corr = \
            pd.DataFrame(z.reshape(-1, n_tf + 2)).corr(method = 'spearman')
        ult_corr_array[:, :, ii] = np.array(pe_corr)
        mask = np.triu_indices(n_tf - 2)
        ult_corr_array[:, :, ii][mask] = np.array(sp_corr)[mask]
    ##
    ult_corr_ylabs = np.array([r'$T = 10^3$', r'$T = 10^5$', 'Stand.', \
        r'$D_{10X\rightarrow Y}$', r'$D_{X\rightarrow 10Y}$', r'$X$ to 1dp', \
        r'$Y$ to 1dp', r'$X$, $Y$ to 2dp', '10% NA', '20% NA', \
        r'$\sigma^2_G$ = 0.1', r'$\sigma^2_{G,X}$ = 1', \
        r'$\sigma^2_{G,Y}$ = 1'])
    ##
    corr_transforms_plot(ult_corr_array[0,:,:], ylabs = ult_corr_ylabs, \
        xlabs = indices_plot1, y_groups = [2, 3, 3, 2, 3], \
        x_groups = [4, 3, 3], figsize = [5, 5], fontsize = [12, 9, 7], \
        figpad = [0.6, 0.1, 0.1], cmap = dv.PRGn_11.mpl_colormap, \
        filename = plot_dir + 'corr_transforms_plots')
##
##
##############################################################
##############################################################
##############################################################
## TABLES
##
##############################################################
## Ulam lattice transformations table (Table III)
##
if 'ul-transforms' in tables:
    try:
        ul_results = load_reshape(file_dir + 'ul_values', ul_shape[0])
        ult_results = load_reshape(file_dir + 'ult_values', ult_shape[0])
    except Exception as err:
        print('ul_values.csv or ult_values.csv missing from ' + file_dir)
        raise
    ##
    ind_exclude = [16, 17, 18, 19, 81, 82, 83, 84]
    ind_include = [ii for ii in range(n_lambda) if ii not in ind_exclude]
    ##
    ult_D = ult_results[:,:,range(0, 2 * n_inds, 2),:] - \
        ult_results[:,:,range(1, 2 * n_inds, 2),:]
    ult_D = ult_D[:,ind_include,:,:]
    ##
    ul_D = ul_results[:,:,range(0, 4 * n_inds, 2)] - \
        ul_results[:,:,range(1, 4 * n_inds, 2)]
    ul_D = np.stack((ul_D[:,:,:n_inds], ul_D[:,:,n_inds:]), axis = 3)
    ul_D = ul_D[:,ind_include,:,:]
    ##
    ult_D = np.insert(ult_D, [0], ul_D, axis = 3)
    ##
    ult_table = np.zeros((n_inds, n_tf + 2, 2))
    denom = np.nanmean(np.abs(np.nanmean(ul_D[:,:,:,:1], axis = 0)), axis = 0)
    ult_table[:,0,0] = np.nanmean(ult_D[:,:,:,0], axis = (0, 1))
    ult_table[:,1:,0] = \
        np.nanmean(ult_D[:,:,:,:1] - ult_D[:,:,:,1:], axis = (0, 1))
    ult_table[:,1:,0] /= denom
    ult_table[:,0,1] = np.nanmean(np.nanstd(ul_D[:,:,:,0], axis = 0), axis = 0)
    ult_table[:,1:,1] = \
        np.nanmean(np.nanstd(ult_D[:,:,:,1:], axis = 0), axis = 0)
    ult_table[:,1:,1] /= ult_table[:,:1,1]
    ult_table = np.round(ult_table, 3)
    inds_order = [4,5,6,0,1,2,3,7,8,9]
    ult_table = ult_table[inds_order,:,:]
    bold_mean = np.abs(ult_table[:,:,0]).argmin(axis = 0)
    bold_std = np.abs(ult_table[:,:,1] - 1).argmin(axis = 0)
    bold_std[1] = np.abs(ult_table[:,1,1]).argmin(axis = 0)
    ##
    str_table = ''
    for ii in range(n_inds):
        str_table += indices_plot1[inds_order[ii]]
        str_table += r' & $\langle\mu\rangle$ = '
        str_table += str('%.3f' % ult_table[ii,0,0]) + r' & $f(\mu,\hat{\mu})$'
        for jj in range(1, n_tf + 2):
            str_table += ' & '
            if bold_mean[jj] == ii:
                str_table += r'\textbf{'
            str_table += str('%.3f' % ult_table[ii,jj,0])
            if bold_mean[jj] == ii:
                str_table += '}'
        str_table += r' \\' + '\n' + r' & $\langle\sigma\rangle$ = '
        str_table += str('%.3f' % ult_table[ii,0,1])
        str_table += r' & $g(\sigma,\hat{\sigma})$'
        for jj in range(1, n_tf + 2):
            str_table += ' & '
            if bold_std[jj] == ii:
                str_table += r'\textbf{'
            str_table += str('%.3f' % ult_table[ii,jj,1])
            if bold_std[jj] == ii:
                str_table += '}'
        str_table += r' \\' + '\n'
    txt_table = open(plot_dir + 'ul-transforms.txt', 'w')
    txt_table.write(str_table)
    txt_table.close()
##
##
##############################################################
##############################################################
##############################################################
## Computation times (Table S.II)
##
if 'computational-times' in tables:
    try:
        lp_time = load_reshape(file_dir + 'lp_time', lp_shape[1])
        ul_time = load_reshape(file_dir + 'ul_time', ul_shape[1])
        hu_time = load_reshape(file_dir + 'hu_time', hu_shape[1])
        hb_time = load_reshape(file_dir + 'hb_time', hb_shape[1])
    except Exception as err:
        print('At least one *_time.csv missing from ' + file_dir)
        raise
    ##
    time_table = np.zeros((n_time, 8, 2))
    time_table[:,0,0] = np.nanmean(lp_time, axis = (0, 1))
    time_table[:,1,0] = np.nanmean(ul_time[:,:,:n_time], axis = (0, 1))
    time_table[:,2,0] = np.nanmean(ul_time[:,:,n_time:], axis = (0, 1))
    time_table[:,3,0] = np.nanmean(hu_time[:,:,:n_time], axis = (0, 1))
    time_table[:,4,0] = \
        np.nanmean(hu_time[:,:,n_time:(2 * n_time)],axis = (0, 1))
    time_table[:,5,0] = np.nanmean(hu_time[:,:,(2 * n_time):], axis = (0, 1))
    time_table[:,6,0] = np.nanmean(hb_time[:,:n_lambda_hb,:], axis = (0, 1))
    time_table[:,7,0] = np.nanmean(hb_time[:,n_lambda_hb:,:], axis = (0, 1))
    time_table[:,0,1] = np.nanstd(lp_time, axis = (0, 1))
    time_table[:,1,1] = np.nanstd(ul_time[:,:,:n_time], axis = (0, 1))
    time_table[:,2,1] = np.nanstd(ul_time[:,:,n_time:], axis = (0, 1))
    time_table[:,3,1] = np.nanstd(hu_time[:,:,:n_time], axis = (0, 1))
    time_table[:,4,1] = \
        np.nanstd(hu_time[:,:,n_time:(2 * n_time)], axis = (0, 1))
    time_table[:,5,1] = np.nanstd(hu_time[:,:,(2 * n_time):], axis = (0, 1))
    time_table[:,6,1] = np.nanstd(hb_time[:,:n_lambda_hb,:], axis = (0, 1))
    time_table[:,7,1] = np.nanstd(hb_time[:,n_lambda_hb:,:], axis = (0, 1))
    time_table = np.round(time_table, 3)
    ##
    indices_time = ['ETE (H)', 'TE (KSG)', 'CTIR', 'EGC', 'NLGC', 'PI']
    indices_time = indices_time + [r'SI$^{(1,2)}$', 'CCM']
    ## order is EGC, NLGC, PI, ETE (H), TE (KSG), CTIR, SI, CCM
    inds_order = [3,4,5,0,1,2,6,7]
    time_table = time_table[inds_order,:,:]
    bold_mean = np.abs(time_table[:,:,0]).argmin(axis = 0)
    bold_std = np.abs(time_table[:,:,1]).argmin(axis = 0)
    ##
    str_table = ''
    for ii in range(n_time):
        str_table += indices_time[inds_order[ii]]
        for jj in range(8):
            str_table += ' & '
            if bold_mean[jj] == ii:
                str_table += r'\textbf{'
            str_table += str('%.3f' % time_table[ii,jj,0])
            if bold_mean[jj] == ii:
                str_table += '}'
            str_table += ' ('
            if bold_std[jj] == ii:
                str_table += r'\textbf{'
            str_table += str('%.3f' % time_table[ii,jj,1])
            if bold_std[jj] == ii:
                str_table += '}'
            str_table += ') '
        str_table += r' \\' + '\n'
    txt_table = open(plot_dir + 'computational-times.txt', 'w')
    txt_table.write(str_table)
    txt_table.close()
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
