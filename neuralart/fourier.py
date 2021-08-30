#functions for Fourier Transformation

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift


def matriceImage(matrice,gamma,rgb):

    '''
    Generate a Fast Fourier Transformation from matrice

    '''
    s = matrice.shape
    a=1.0/gamma;
    norm=matrice.max()
    m = np.power(matrice/norm,a)
    im = np.zeros((s[0],s[1],3),dtype=float)
    im[:,:,0] = rgb[0]*m
    im[:,:,1] = rgb[1]*m
    im[:,:,2] = rgb[2]*m
    return im

def matriceImageLog(matrice,rgb):

    '''
    Generate a Fast Fourier Transformation from matrice

    '''
    s = matrice.shape
    m = np.log10(1+matrice)
    min = m.min()
    max = m.max()
    m = (m-min)/(max-min)
    im = np.zeros((s[0],s[1],3),dtype=float)
    im[:,:,0] = rgb[0]*m
    im[:,:,1] = rgb[1]*m
    im[:,:,2] = rgb[2]*m
    return im

def plotSpectre(image,Lx,Ly):

    '''
    Plot function of FFT

    '''

    (Ny,Nx,p) = image.shape
    fxm = Nx*1.0/(2*Lx)
    fym = Ny*1.0/(2*Ly)
    plt.imshow(image,extent=[-fxm,fxm,-fym,fym])

    plt.xlabel("fx")
    plt.ylabel("fy")
