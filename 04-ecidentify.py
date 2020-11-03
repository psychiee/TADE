#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:17:43 2018

@author: wskang
"""
import sys, os
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
#from astropy.modeling import models, fitting 
from scipy.optimize import curve_fit
#from numpy.polynomial.chebyshev import chebfit, chebval
from specutil import read_params, cr_reject, find_emission, smooth

#=============================================================
ECID_FILE = 'compFLI.ecid'
ECID_SPEC = 'compFLI.ec.fits'
# OPEN template information of emissions (from ecidentify)
dat = np.genfromtxt(ECID_FILE)
eap, eord = np.array(dat[:,0], int), np.array(dat[:,1], int)
epix, ewfit, ew0, eflag = dat[:,2], dat[:,3], dat[:,4], dat[:,7]
eordset = list(set(eord))
eordset.sort(reverse=True)
# OPEN template spectrum 
hdu = fits.open(ECID_SPEC)[0]
espec = hdu.data
ex = np.arange(espec.shape[1])+1
#==============================================================

par = read_params()
os.chdir(par['WORKDIR'])

EID_OUTPUT = par['EIDFILE']
ORD_START = int(par['EIDSTART'])
SCALE, SHIFT = float(par['EIDSCALE']), float(par['EIDSHIFT']) 
THRES, GPIX = int(par['EIDTHRES']), int(par['EIDGPIX'])
SPEC_PLOT = bool(int(par['EIDPLOT']))
TEST_PLOT = 0 

#print sys.argv
if len(sys.argv) > 1: 
    COMP_SPEC = sys.argv[1]
else:
    COMP_SPEC = 'comp1.ec.fits'  

print ('COMP_SPEC=%s' %(COMP_SPEC,))
print ('THRES=%.1f, GPIX=%.1f' %(THRES,GPIX))
print ('SPEC_PLOT=%i, TEST_PLOT=%i' %(SPEC_PLOT,TEST_PLOT))

# OPEN the FITS file of new spectrum 
hdu = fits.open(COMP_SPEC)[0]
spec = hdu.data 
#spec = spec[RANGE[0]:RANGE[1],:]
sx = np.arange(spec.shape[1])+1
# DEFINE the name of spectrum 
FID = os.path.splitext(COMP_SPEC)[0]
# OPEN the line information of new spectrum 
fout = open(EID_OUTPUT, 'w')

# LOOP for aperture of new spectrum 
for j in range(0,spec.shape[0]):
    sord = ORD_START - j
    print ('#AP', j+1, ' #ORDER', sord)
    # FIND the order & aperture in the template 
    if eordset.count(sord) == 0: continue
    eapnum = eordset.index(sord)
    
    NSTRONG = 40
    # EXTRACT the aperture in the template
    erow = espec[eapnum,:]
    ecx, ecy = find_emission(ex, erow, thres=3*THRES, width=GPIX)
    if len(ecx) > NSTRONG: 
        ss = np.argsort(ecy)
        ecx, ecy = ecx[ss[-NSTRONG:]], ecy[ss[-NSTRONG:]]
    # EXTRACT the aperture in new spectrum 
    srow = spec[j,:]
    ssig = np.std(srow)/5
    scx, scy = find_emission(sx, srow, thres=3*THRES, width=GPIX)
    if len(scx) > NSTRONG: 
        ss = np.argsort(scy)
        scx, scy = scx[ss[-NSTRONG:]], scy[ss[-NSTRONG:]] 
    # FITTING the new spectrum with template spectrum 
    # FIND the factors of SCALE, SHIFT

    def conv(x, *p):
        a, b = p 
        ox = ex*a + b 
        oy = np.interp(x, ox, erow)
        oy[oy < THRES] = 0
        oy = oy / np.max(oy)
        oy = smooth(oy, width=GPIX)
        return oy     
    srow[srow < THRES] = 0 
    srow2 = srow / np.max(srow)
    srow2 = smooth(srow2, width=GPIX)
    c, cov = curve_fit(conv, sx, srow2, [SCALE, SHIFT], \
                       np.zeros_like(sx)+0.01, \
                       bounds=([SCALE*0.8,SHIFT-50],[SCALE*1.2,SHIFT+50]))
    a, b = c 
    print ('#factors by curve_fit')
    print ('  SCALE   SHIFT')
    print ('%7.4f %7.2f' % (a, b))
    
    # MATCHING the strong emissions 
    fdx, fsx, fex = [], [], [] 
    for ix in ecx:
        dx = scx - (ix*a+b)
        mm = np.argmin(abs(dx))
        fsx.append(scx[mm])
        fex.append(ix)
        fdx.append(dx)
    fsx, fex = np.array(fsx), np.array(fex)
    
    # CALIBRATE the template with matching emissions
    cnum = len(fex)
    print ('#matching lines:', cnum)
    if cnum == 0: 
        print ('NO LINES')
        continue
    if len(fsx) > 2: 
        # FIND the correct matching lines by sigma-clipping
        print (' N pts    dx    sig')
        n = 1   
        while n < 10:
            a, b = np.polyfit(fex, fsx, 1)
            dx = fsx - (fex*a+b)
            sig = np.std(dx)
            print( '%2d %3d %5.1f %6.2f' % \
              (n, len(fex), max(dx)-min(dx), sig))
            cond = abs(dx) < sig
            fex, fsx = fex[cond], fsx[cond]
            if len(fex) < 10: break
            if sig < 1.0: break
            if cnum == len(fex): break
            cnum = len(fex)
            n += 1
    # PRINT the scaling factors between new and template
    print ('SCALE SHIFT   sig')
    print ('%5.3f %5.1f %5.3f' % (a, b, np.std(fsx-(fex*a+b))))
    
    '''
    fig, ax = plt.subplots(num=2, figsize=(7,7))
    ax.plot(fex, fsx, 'r+')
    ax.plot(fex, fex*a+b, 'k--', alpha=0.7, lw=2)
    ax.grid()
    fig.savefig(FID+'-FIT%02d.png' % (sord,))
    fig.clf()
    '''
    # MODIFY the default factors
    SCALE, SHIFT = a, b
    
    # PLOT the matching result of spectrum
    if SPEC_PLOT:
        fig, ax = plt.subplots(num=99, figsize=(25,10))
        ax.plot(ex*a+b, erow, 'g-', lw=4, alpha=0.5, label='TEMPLATE')
        ax.plot(sx, srow, 'b-', lw=1, label=FID)
        ax.plot(ecx*a+b, np.zeros_like(ecx)-9000, 'g|', \
                mew=7, ms=20, alpha=0.5)
        ax.plot(scx, np.zeros_like(scx)-5000, 'b|', \
                mew=3, ms=20, alpha=0.8)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Relative intensity')
        ax.set_ylim(-15000,200000)
        ax.set_xlim(min(sx), max(sx))        
        ax.set_title('AP%02d ORD%02d Transform Result of ThAr' % (j+1, sord))
        ax.grid()
        ax.legend()
        fig.savefig(FID+'-CORR%02d.png' % (j+1,))
        fig.clf()
    # FIND the position of emission in a spectrum
    vv, = np.where(eord == sord)
    # LOOP of lines
    for vidx in vv:
        # read the ecidentify results 
        i0 = epix[vidx] # pixel position 
        wv0 = ew0[vidx] # wavelenth of ThAr emission 
        flg = eflag[vidx] # flag for calibration fitting 
        # transform pixel position of template 
        i1 = i0*a+b
        ex1 = ex*a+b
        # define crop range
        xmin, xmax = i1-3*GPIX, i1+3*GPIX
        vv0, = np.where((ex1 < xmax) & (ex1 > xmin))
        vv1, = np.where((sx < xmax) & (sx > xmin))
        if len(vv1) == 0: continue
        # correct base level of emission 
        cbox0 = erow[vv0] - np.min(erow[vv0])
        cbox1 = srow[vv1] - np.min(srow[vv1])
        # convert into same coordinates
        cx0, cx1 = ex1[vv0], sx[vv1] 
        max0, max1 = max(cbox0), max(cbox1)
        ymax = np.max([max1, max0])
        # find peaks in new spec
        peaks, _ = find_emission(cx1, cbox1, thres=THRES, width=GPIX)
        if len(peaks) > 0:
            mm = np.argmin(abs(peaks-i1))
            mpeak = peaks[mm]
            mdx = mpeak - i1
            # matching with the template emission line
            if abs(mdx) > GPIX: 
                print ('SKIP: %i %.2f' % (i0, mdx))
                find_flag = False
            else:
                # save the result into the text file 
                fstr = '%2d %2d %10.4f %8.3f %8.3f %8.5f %i' % \
                   (j+1, sord, wv0, i0, mpeak, mdx, flg)
                fout.write(fstr+'\n')
                find_flag = True
                
        if len(peaks) == 0: continue
        #if TEST_PLOT == False: continue 
        if find_flag == True: continue
        if not os.path.exists('test'):
            os.mkdir('test')
            
        # plot the emissions
        fig, ax = plt.subplots(num=98, figsize=(7,5))
        ax.plot(cx1, cbox1, 'bo-', label=FID)
        ax.plot(cx0, cbox0, 'go-', lw=2, ms=7, alpha=0.5, label='TEMPLATE')
        ax.plot([peaks, peaks], [0, ymax], 'b--', alpha=0.5)
        # mark the center position of new
        #if find_flag:
        #    ax.plot([mpeak, mpeak], [0, ymax], 'b-', lw=9, alpha=0.3)
        # mark the center of template 
        ax.plot([i1,i1],[0,ymax], 'g-', lw=9, alpha=0.3)
        ax.set_xlim(xmin,xmax)
        ax.grid()
        ax.legend(loc='upper right')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Relative intensity')
        ax.set_title('AP%02d - Profile of ThAr %.4f' % (j+1, wv0))
        fig.savefig('test/AP%02d-PEAK%04d' % (j+1,i0))
        fig.clf()
fout.close()    
plt.close('all')

    
    


