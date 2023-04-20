#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from turtle import color
from fit_routine_oiii import WLAX, Lines, lines, oiii_doublet, c, z, oiii_wratio
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class OIII(Lines):
    def __init__(self, name: str, rest, crange, vrange, cube: NDArray, varc: NDArray, contspec: NDArray):
        super().__init__(name, rest, crange, vrange, cube, varc, contspec)
        self.CUBE = cube
        self.fitcube = np.zeros((6, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((4, self.cube_x, self.cube_y))

    def plot_spe(self, ax_in, i, j):
        self.spec = self.spaxel(self.subcube, i, j)
        self.varspec = self.spaxel(self.errcube, i, j)
        self.remove_nan()
        
        lranges = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])
        fit_spec = self.baseline_subtraction()
        

        popt = list(self.fitcube[0:4, i, j])
        fitsnr = self.fitcube[4, i, j]
        chi = self.fitcube[5, i, j]
        uncertainty = list(self.fiterrcube[:, i, j])
        
        quicksnr = sum(fit_spec)/np.sqrt(sum(self.varspec))

        base_cont_lvl = self.get_cont_lvl(self.basespec)
        line_cont_lvl = self.get_cont_lvl(self.spec)
        #aC = (line_cont_lvl/base_cont_lvl) * self.basespec

        #plot
        ax_in[0].step(self.wlax, fit_spec, where='mid', color='black')
        ax_in[0].step(self.wlax[lranges], fit_spec[lranges], where='mid', color='#1f77b4')
        ax_in[0].axhline(y=np.median(fit_spec[~lranges]), color = 'green', label = 'Median of baseline')
        ax_in[0].plot(self.wlax, oiii_doublet(self.wlax, *popt), color='Orange')
        ax_in[0].axhline(y=np.median(fit_spec[lranges]), color='r', label='Median around line ranges')
        ax_in[0].errorbar(self.wlax,fit_spec,yerr=np.sqrt(self.varspec),color='#1f77b4',linestyle='')
        ax_in[0].set_title(f"(i,j): ({j+1}, {i+1}), quicksnr = {quicksnr}, fitsnr = {fitsnr}, chi = {chi}")
        ax_in[0].legend(loc='upper right')

        linewlax = WLAX[620:770]
        linespec = self.CUBE[620:770,i,j]
        
        ax_in[1].step(linewlax, linespec, where='mid')     
        ax_in[1].axvspan(popt[1]-1.18*popt[2], popt[1]+1.18*popt[2], color='g', alpha=0.3)
        ax_in[1].axvspan(oiii_wratio*(popt[1]-1.18*popt[2]), oiii_wratio*(popt[1]+1.18*popt[2]), color='g', alpha=0.3)
        ax_in[1].set_title(f"Original Spectrum at ({j+1},{i+1})")
        

