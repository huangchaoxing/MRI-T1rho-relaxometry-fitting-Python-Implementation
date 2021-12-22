# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:44:06 2021

@author: Chaoxing Huang, DIIR, CUHK
Implementation was based on  mannual  and matlab implementation Professor Thierry Blu, 
EE,CUHK
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_image (nifty_path):
    '''
    

    Parameters
    ----------
    nifty_path : STRING
        THE FILE DIR OF YOUR NIFFTY IMAGE FILE

    Returns
    -------
    scaled_image : NUMPY ARRAY
        THE IMAGE !

    '''
    img = nib.load(nifty_path)

    images = img.get_fdata()
    slope = img.dataobj.slope
    inter = img.dataobj.inter
    scaled_image = slope * images + inter

    return scaled_image

def compute_J(b,x,y):
    '''
    

    Parameters
    ----------
    b : np array
        H X W
    x : np array, time of spin-lock
        NTSL X h x w
    y : np array, multiple dynamic scans
        NTSL X h x w

    Returns
    -------
    J : np array, cost function value corresponds to every pixels 
        h  x w 

    '''
    # Implementation accroding to Thierry's manuual 
    f1 = np.mean(y*np.exp(-b*x),axis=0)
    alpha1 = np.mean(np.exp(-b*x),axis=0)
    alpha2 = np.mean(np.exp(-2*b*x),axis=0)
    f0 = np.mean(y,axis=0)
        
    a = (f1-alpha1*f0)/(alpha2-alpha1**2)
    L = np.abs(y-a*np.exp(-b*x))
    J = np.sum(L**2,axis=0)
    
    return J

def fit(x,y):
    '''
    

    Parameters
    ----------
    x : np array, time of spin-lock
        NTSL X h x w
    y : np array, multiple dynamic scans
        NTSL X h x w

    Returns T1rho
    -------
    
    '''
    
    bmin=1e-5*np.ones_like(y[0])   
    bmax = 1e5*np.ones_like(y[0])   
    eps = 1e-9
    count =0
    while(1):
        # Dichotomic algorithm. 
        b = (bmin+bmax)/2
        J = compute_J(b, x, y)
        #Jmin = compute_J(bmin,x,y)
        #Jmax = compute_J(bmax,x,y)
        dJ = compute_J(b+eps, x, y)-compute_J(b, x, y)
        pos_idx = np.where(dJ>=0)
        neg_idx = np.where(dJ<0)
        bmax[pos_idx]=b[pos_idx]
        bmin[neg_idx]=b[neg_idx]
        
        interval_length = np.abs(bmax-bmin)
        count+=1
        if np.linalg.norm(interval_length)<1e-3:
            break
        elif count>1000:
            print('take too long,break!')
            break
    return 1/(b+1e-9)   # we are computing t1rho, we need to invert r1rho to t1rho

if __name__=='__main__':
    
    image_list = ['data/00001.nii','data/00002.nii','data/00003.nii','data/00004.nii']
    
    y = []
    for image in image_list:
        y.append(load_image(image))
    y = np.array(y)
    
    x = np.array([0,0.01,0.03,0.05])
    x = x.reshape((4,1,1))
    x = x*np.ones_like(y)
    t1rho = fit(x,y)
    plt.figure(1)
    plt.imshow(t1rho,vmin=0.02,vmax=0.06,cmap='jet')
    plt.colorbar()
    