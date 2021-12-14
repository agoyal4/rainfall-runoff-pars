# -*- coding: utf-8 -*-
"""
Color plot module

Created on Mon Mar  9 14:51:56 2020

@author: abhishek
"""
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

font = {'family' : 'Arial',
        'size'   : 6}

plt.rc('font', **font)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['axes.linewidth'] = 0.5 #set the value globally
plt.rcParams['axes.labelweight'] = 'normal'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def colorplot(t, x, y, z, event_no, ifsave):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(7, 4), dpi=300)
    maxlim = np.round(np.max(z[:,:,1:np.size(z,axis=2)-1]), 1)
    levels = np.arange(0, 2, 0.1)
    print(maxlim)
    ncontours = int(maxlim * 10 + 2)
    for idx, ax in enumerate(axes.flat):
    #    ax.set_axis_off()
        if idx >= len(t):
            fig.delaxes(ax)
            continue
        im = ax.contourf(x, y, z[:,:,idx], levels=levels[0:ncontours], vmin=0, cmap='Greys')
        # print(im)
        ax.set_xticks(np.arange(-5, 6, 2))
        ax.set_yticks(np.arange(0, 6, 1))
        ax.xaxis.set_tick_params(which='major', length=1.5, width=0.5, direction='out')
        ax.yaxis.set_tick_params(which='major', length=1.5, width=0.5, direction='out')
        ax.set_xlabel('$\mu_y$')
        ax.set_ylabel('$\sigma_y$')
        ax.set_title(r't = %.1f hr' % (t[idx]), size=8, weight='normal')
    
    fig.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.8, wspace=0.5, hspace=0.7)
    
    for c in im.collections:
        c.set_edgecolor(None)
    
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8   
    cb_ax = fig.add_axes([0.83, 0.3, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel('PDF')
    # set the colorbar ticks and tick labels
    cbar.set_ticks(levels[0:ncontours])
    # cbar.set_ticks(np.arange(0, maxlim+.05, 0.1))
    # cbar.ax.set_tickparams(labelsize=1.5, width=0.5)
    # cbar.set_ticklabels(np.linspace(0, maxlim+0.05, num=ncontours))
    # cbar.set_ticklabels(np.round(np.arange(0, maxlim-0.09, 0.1), 1))
    fig.suptitle(str("Event: cal_{}".format(event_no)), x=0.46, y=0.99, size=8, weight='bold')
    if ifsave == 1:
        fig.savefig(str("D:\\Research\\Hydraulic Conductivity\\problem_3\\figures\\cal_with_prior\\cal_{}.pdf".format(event_no)), dpi=300)

def colorplot_single(x, y, z, ifsave, kwargs):
    fig, ax = plt.subplots(figsize=(5,4), dpi=300)
    im = ax.contourf(x, y, z, cmap='Greys')
    ax.set_aspect('auto')
    ax.set_xticks(np.arange(-5, 6, 2))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.xaxis.set_tick_params(which='major', length=1.5, width=0.5, direction='out')
    ax.yaxis.set_tick_params(which='major', length=1.5, width=0.5, direction='out')
    ax.set_xlabel('$\mu_y$')
    ax.set_ylabel('$\sigma_y$')
    ax.set_ylim(0, 5)
    ax.set_title('Prior using %s data' % kwargs)
    cbar = fig.colorbar(im)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel('PDF')
    plt.tight_layout()
    if ifsave == 1:
        fig.savefig(str("D:\\Research\\Hydraulic Conductivity\\problem_3\\figures\\cal_with_prior\\prior_%s.png" %kwargs), dpi=300)    
        
def colorplot_conflation(x, y, z, ifsave, kwargs):
    # hdrmuy, hdrsigy = hdrparams
    fig, ax = plt.subplots(figsize=(5,4), dpi=300)
    im = ax.contourf(x, y, z, levels=10, cmap='coolwarm')
    # ax.scatter(hdrmuy, hdrsigy, s=5, c='r', marker='o', alpha=0.1)
    # print(im.levels.shape)
    ax.set_aspect('auto')
    ax.set_xticks(np.arange(-5, 6, 2))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.xaxis.set_tick_params(which='major', length=1.5, width=0.5, direction='out')
    ax.yaxis.set_tick_params(which='major', length=1.5, width=0.5, direction='out')
    ax.set_xlabel('$\mu_Y$', fontsize=10)
    ax.set_ylabel('$\sigma_Y$', fontsize=10)
    ax.set_ylim(0, 5)
    # ax.set_title('Conflation using %s data' % kwargs, fontsize=12)
    # for c in im.collections:
    #     c.set_edgecolor(None)
    
    cbar = fig.colorbar(im)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel('PDF', fontsize=10)
    plt.tight_layout()
    if ifsave == 1:
        fig.savefig(r"D:\Research\2021\CONFERENCES\EWRI 2021\conflation.png", dpi=600)
        





import seaborn as sns
from matplotlib.ticker import MultipleLocator      
def val_plot(data, r2, ifsave):
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7.3,9.0), dpi=200)
    for idx in range(len(data)):
        val = data[idx]
        col = 0
        if (idx % 2) == 1:
            col = 1
        ax[idx//2, col].plot(val[0], val[1], 'd', markersize=3, 
          color='r', label='Observed', zorder=3)
        ax[idx//2, col].plot(val[0], val[4], '-', color='k', lw=1.0, label='Most likely model')
        ax[idx//2, col].fill_between(val[0], val[2], val[3], 
          color=sns.color_palette('PRGn', 10)[8], alpha=0.5, label='95% credible region', lw=0)
        ax[idx//2, col].legend(loc='upper right', frameon=False)
        ax[idx//2, col].set_xlabel('Time (hr)')
        ax[idx//2, col].set_ylabel('$E[I(t)]$ (mm/hr)')
        ax[idx//2, col].set_title(r'Event: val_%d' % (idx+1), fontsize=9, weight='bold')
        ax[idx//2, col].set_xlim([-0.1*np.max(val[0])/8.0, np.ceil(np.max(val[0])) + 0.1*np.max(val[0])/8.0])
        if idx == 0:
            ymax = 16
        elif idx == 1:
            ymax = 12
        else:
            ymax = 10
        ax[idx//2, col].set_ylim([-0.4*ymax/10.0, ymax + 0.4*ymax/10.0])
        # ax[idx//2, col].text(0.05, 0.9, r'$R^2 = {:0.3f}$'.format(r2[idx]), transform=ax[idx//2, col].transAxes, fontdict=font)
        ax[idx//2, col].xaxis.set_major_locator(MultipleLocator(1))
        ax[idx//2, col].xaxis.set_minor_locator(MultipleLocator(0.5))
        if idx == 0:
            ax[idx//2, col].yaxis.set_major_locator(MultipleLocator(4))
        ax[idx//2, col].xaxis.set_tick_params(which='major', length=3, width=0.5, direction='in')
        ax[idx//2, col].xaxis.set_tick_params(which='minor', length=1.5, width=0.5, direction='in')
        ax[idx//2, col].yaxis.set_tick_params(which='major', length=3, width=0.5,direction='in')
    fig.tight_layout()
    if ifsave == 1:
        fig.savefig(str("D:\\Research\\Hydraulic Conductivity\\problem_3\\figures\\cal_with_prior\\validation.pdf"), dpi=300)