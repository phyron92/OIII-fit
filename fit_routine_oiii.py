from ast import Call
from opcode import hascompare
from typing import Callable, Iterable, List, Tuple, Union, Optional
from numpy.typing import NDArray
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.optimize import curve_fit
from tqdm import tqdm
from copy import deepcopy
from inspect import signature
import matplotlib.pyplot as plt 
import numpy as np
import time

def AIR(VAC):
    return VAC / (1.0 + 2.735182e-4 + 131.4182 / VAC**2 + 2.76249E8 / VAC**4)

lines = {}
#rest, crop ranges, vrange for line ranges, no.arguments
lines['SII'] = [(6718.29, 6732.67), (6700., 6865.), (2000.,2000.)]
lines['OIII'] = [(4960.3, 5008.24), (4940., 5240.), (2000.,2000.)] #crange = (4950., 5200.), skyline on left ~4900
lines['OI6300'] = [(6302.046, 6302.046), (6250., 6500.), (1500., 1500.)]
lines['OI'] = [(6302.046, 6365.536), (6250., 6500.), (1000.,1000.)]
lines['Ha+NII'] = [(6549.86, 6564.61, 6585.27), (6400., 6720.), (2000., 2000.)]
lines['Hb'] = [(4862.68, 4862.68), (4800., 5000.), (1000., 1000.)]

#General functions
#Redshift
-->z = 0.1283
#Wavelength to Velocity scale
c = 299792 #km/s
def lambda2vel(xLambda, restLambda):
    k = (xLambda / restLambda) ** 2
    return c *  (k - 1) / (k + 1) 

def vel2lambda(vel, restLambda):
    k = vel / c
    return restLambda * ((1+k)/(1-k))**0.5

def chisq(fit_func: Callable, wlax: NDArray, fit_spec: NDArray, errspec: NDArray, popt: List[float]):
    return np.sum((fit_spec - fit_func(wlax, *popt))**2. / errspec**2.) / (np.size(fit_spec) - np.size(popt))


sii_wratio = lines['SII'][0][1]/lines['SII'][0][0]
def sii_doublet(x, h1, h2, lmbda1, lmbda2, sig, c):
    return (h1/np.sqrt(2*np.pi)/sig)*np.exp(-(x-lmbda1)**2/2/sig**2) +            (h2/np.sqrt(2*np.pi)/sig)*np.exp(-(x-lmbda2)**2/2/sig**2) + c
sii_doublet = np.vectorize(sii_doublet)

oiii_wratio = lines['OIII'][0][1]/lines['OIII'][0][0]
def oiii_doublet(x, h1, lmbda1, sig, c):
    return (h1/np.sqrt(2*np.pi)/sig/3)*np.exp(-(x-lmbda1)**2/2/sig**2) +            (h1/np.sqrt(2*np.pi)/sig)*np.exp(-(x-oiii_wratio*lmbda1)**2/2/sig**2) + c
oiii_doublet = np.vectorize(oiii_doublet)

oi_wratio = lines['OI'][0][1]/lines['OI'][0][0]
def oi_doublet(x, h1, h2, lmbda1, sig, c):
    return (h1/np.sqrt(2*np.pi)/sig)*np.exp(-(x-lmbda1)**2/2/sig**2) +            (h2/np.sqrt(2*np.pi)/sig)*np.exp(-(x-oi_wratio*lmbda1)**2/2/sig**2) + c
oi_doublet = np.vectorize(oi_doublet)

#Ratio to Ha line
niil_wratio = lines['Ha+NII'][0][0]/lines['Ha+NII'][0][1]
niir_wratio = lines['Ha+NII'][0][2]/lines['Ha+NII'][0][1]

def hamod1(x, h_nii, h_ha, lmbda1, sig):
    return (h_nii/3/np.sqrt(2*np.pi)/sig)*np.exp(-(x-niil_wratio*lmbda1)**2/2/sig**2) +            (h_ha/np.sqrt(2*np.pi)/sig)*np.exp(-(x-lmbda1)**2/2/sig**2) +            (h_nii/np.sqrt(2*np.pi)/sig)*np.exp(-(x-niir_wratio*lmbda1)**2/2/sig**2) 
hamod1 = np.vectorize(hamod1)

def hamod2(x, h_nii1, h_ha1, lmbda1, sig1, h_nii2, h_ha2, lmbda2, sig2):
    return (h_nii1/3/np.sqrt(2*np.pi)/sig1)*np.exp(-(x-niil_wratio*lmbda1)**2/2/sig1**2) +            (h_ha1/np.sqrt(2*np.pi)/sig1)*np.exp(-(x-lmbda1)**2/2/sig1**2) +            (h_nii1/np.sqrt(2*np.pi)/sig1)*np.exp(-(x-niir_wratio*lmbda1)**2/2/sig1**2)            +            (h_nii2/3/np.sqrt(2*np.pi)/sig2)*np.exp(-(x-niil_wratio*lmbda2)**2/2/sig2**2) +            (h_ha2/np.sqrt(2*np.pi)/sig2)*np.exp(-(x-lmbda2)**2/2/sig2**2) +            (h_nii2/np.sqrt(2*np.pi)/sig2)*np.exp(-(x-niir_wratio*lmbda2)**2/2/sig2**2) 
hamod2 = np.vectorize(hamod2)

def single(x, h, lmbda, sig, c):
    return (h/np.sqrt(2*np.pi)/sig)*np.exp(-(x-lmbda)**2/2/sig**2) + c
single = np.vectorize(single)

-->WLAX = 4750.45361328125 + np.arange(3682) * 1.25
class Lines:
    def __init__(self, name: str, rest: Union[Tuple[float, float],Tuple[float, float, float]], crange: Tuple[float, float], 
                 vrange: Tuple[float, float], cube: NDArray, varc: NDArray, contspec: NDArray, quickrej_threshold: float = 0.2):

        self.name = name
        self.rest = rest
        self.obs = tuple(np.array(rest)*(1+z))
        self.cranges = crange
        self.lranges = self.lrange(vrange)
        self.quickrej_threshold = quickrej_threshold

        self.wlax = self.get_slice(WLAX)
        self.subcube = self.get_slice(cube)
        self.errcube = self.get_slice(varc)
        self.basespec = self.get_slice(contspec)
        self.spec = None
        self.varspec = None


        self.fitcube = None
        self.fiterrcube = None

        self.cube_x = cube.shape[1]
        self.cube_y = cube.shape[2]
        self.rejcube = np.full((3, self.cube_x, self.cube_y), np.nan)

        self.mask = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])

    #Load in previous fit cubes for inspections
    def load_fitcubes(self, FITCUBE_DIR: str) -> None:
        fithdul =  fits.open(FITCUBE_DIR)
        fitcube_z = len(fithdul)
        FITCUBE = np.zeros((fitcube_z, self.cube_x, self.cube_y))
        for k in range(fitcube_z):
            for j in range(self.cube_x):
                for i in range(self.cube_y):
                    FITCUBE[k,j,i] = fithdul[k].data[j,i]
        if self.fitcube is not None and self.fiterrcube is not None:
            self.fitcube = FITCUBE[:self.fitcube.shape[0]]
            self.fiterrcube = FITCUBE[self.fitcube.shape[0]:]

    def spaxel(self, cube_in: NDArray, i: int, j: int) -> NDArray:
        return np.array(cube_in[:,i,j])
    
    #Slice an array according to the desired wavelength range
    def get_slice(self, arr_in: NDArray) -> NDArray:
-->        mask = (WLAX[620] < WLAX) & (WLAX < WLAX[770])
        return arr_in[mask]
    
    #Return the lrange used in curve_fit
    def lrange(self, vrange:Tuple[float, float]) -> Tuple[float, float]:
        if type(self.rest) is tuple:
            lleft = vel2lambda(lambda2vel(self.obs[0], self.rest[0])-vrange[0], self.rest[0])
            lright = vel2lambda(lambda2vel(self.obs[-1], self.rest[-1])+vrange[1], self.rest[-1])
        else:
            lleft = vel2lambda(lambda2vel(self.obs, self.rest)-vrange[0], self.rest)
            lright = vel2lambda(lambda2vel(self.obs, self.rest)+vrange[1], self.rest)
        return lleft, lright
    
    #Return the continuum level of subspec
    def get_cont_lvl(self, subspec: NDArray) -> float:
        lranges = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])
        return np.average(subspec[~lranges])
    
    #Return a baseline subtracted spectrum ready for fitting
    def baseline_subtraction(self) -> NDArray:
        base_cont_lvl = self.get_cont_lvl(self.basespec)
        line_cont_lvl = self.get_cont_lvl(self.spec)
        return self.spec - (line_cont_lvl/base_cont_lvl) * self.basespec
    
    #Remove nan values in the spectrum
    def remove_nan(self) -> None:
        self.spec[np.isnan(self.spec)] = 50
        self.varspec[np.isnan(self.varspec)] = 0.

    #Return arrays ready for curve_fit    
    def get_fit_spaxel(self, i: int, j: int) -> Tuple[NDArray, NDArray]:
        self.spec = self.spaxel(self.subcube, i, j)
        self.varspec = self.spaxel(self.errcube, i, j)
        
        if all(np.isnan(self.spec)):
            self.set_to_nan(i,j)
            return None, None
        
        self.remove_nan()
        fit_spec = self.baseline_subtraction()
        
        if sum(fit_spec)/np.sqrt(sum(self.varspec)) < self.quickrej_threshold:
            self.set_to_nan(i,j)
            return None, None
        
        return fit_spec, deepcopy(np.sqrt(self.varspec))
    
    #Set the fit results of a spaxel to nan
    def set_to_nan(self, i: int, j: int) -> None:
        self.fitcube[:,i,j] = np.nan
        self.fiterrcube[:,i,j] = np.nan
    
    #Implemented in child class
    def plot_spe(self):
        print("No implementation yet!")
        return None
        ...
    
    #Plot the spectra of a list of spaxels
    def plot_eval(self, pixels: Iterable[Tuple[int, int]], save: bool = False, fname: Optional[str] = None):
        fig, axes = plt.subplots(len(pixels), 2, figsize=(19.5, 5*len(pixels)))
        counter = 0
        for pix in pixels:
            i, j = pix
            self.plot_spe(axes[counter], j-1, i-1)
            counter += 1
        if save:
            plt.savefig("../plots/"+fname+".jpg")

if __name__ == "__main__":
    print(AIR(6302.046))
    print(6302.046*1.01)

