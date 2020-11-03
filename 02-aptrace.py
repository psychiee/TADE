#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:38:25 2018

@author: wskang
"""
import sys, os
import numpy as np 
import matplotlib.pyplot as plt 
#from scipy.signal import find_peaks_cwt
from numpy.polynomial.chebyshev import chebfit, chebval
from astropy.io import fits
from specutil import read_params, cr_reject, find_emission

par = read_params()
os.chdir(par['WORKDIR'])
NAP = int(par['NAP'])
STARTY = int(par['APTSTART'])
THRES = float(par['APTTHRES'])
#RANGE = np.array(par['APTRANGE'].split(','), int)
# CUTTING parameters
dW, dH = int(par['APTDW']), int(par['APTDH'])
FIT_PLOT = bool(par['APTPLOT'])
APT_FILE = par['APTFILE']
APT_ORDER = int(par['APTORDER'])

# READ the flat file name 
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'iflat.fits'

print ('NAP = ', NAP)
print ('STARTY = ', STARTY)
print ('THRES = ', THRES)
print ('dW, dH = ', dW, dH)
print ('PLOT = ', FIT_PLOT)
print ('FILE = ', fname )

# OPEN the flat spectrum image 
hdu = fits.open(fname)[0]
fidx = os.path.splitext(fname)[0]
dat = hdu.data
# TRIM the data image 
#dat = dat[RANGE[0]:RANGE[1],:]
# SET the shape and mid-point
H, W = dat.shape
halfW = int(W/2) 
# READ the middle column/row in the image 
ys = np.arange(H)
cols = np.median(dat[:,(halfW-dW):(halfW+dW)], axis=1)
# REMOVE cosmic rays 
fcols = cr_reject(cols)#, nsigma=7, npix=5)
# FIND emission for apertures in the middle
cys0, cpeaks0 = find_emission(ys, fcols, thres=THRES, width=dH)
#cys0 = find_peaks_cwt(fcols, np.arange(2,dH+1))
#cpeaks0 = fcols[cys0]
# CHECK the found peaks
NPEAKS = len(cys0)
cys, cpeaks = [], []
for k in range(NPEAKS):
    # CHECK start Y-pixel position 
    if cys0[k] < STARTY: 
        print ('CUTOFF: %i, %i' % (cys0[k], cpeaks0[k]))
        continue 
    # CHECK peak values w.r.t. nearby peaks
    if cpeaks0[k] < (cpeaks0[k-1]+cpeaks0[-(NPEAKS-k-1)])/10.0:
        print ('SIDE-CHCEK: %i, %i, %i, %i' % \
               (cys0[k], cpeaks0[k], cpeaks0[k-1], cpeaks0[-(NPEAKS-k-1)]))
        continue
    #print ('%i, %i' % (cys0[k], cpeaks0[k]))
    cys.append(cys0[k])
    cpeaks.append(cpeaks0[k])
cys, cpeaks = np.array(cys), np.array(cpeaks)
# SORT the peaks with values and SELECT # of the high peaks = NAP
#pp = np.argsort(cpeaks)[-NAP:]
#pp.sort()
#cys, cpeaks = cys[pp], cpeaks[pp]
# FIND PEAKS from the first aperture to the NAP
cys, cpeaks = cys[:NAP], cpeaks[:NAP]
cap = np.arange(NAP)+1

# PLOT the middle plane
if FIT_PLOT: 
    fig, ax = plt.subplots(num=1, figsize=(12,6))
    ax.plot(ys, cols, 'g-', lw=4, alpha=0.4)
    ax.plot(ys, fcols, 'b-', lw=1)
    ax.plot(cys, cpeaks, 'ro', ms=4, alpha=0.7)
    ax.plot(cys-dH/2, cpeaks, 'r|', ms=15)
    ax.plot(cys+dH/2, cpeaks, 'r|', ms=15)
    ax.set_title('APFIND in the middle plane')
    #plt.show()
    fig.savefig('APFIND')
    plt.close('all')

# PLOT the image of spectrum 

fig, ax = plt.subplots(num=2, figsize=(20,10))

ldat = np.arcsinh(dat)
z1, z2 = np.percentile(ldat,60), np.percentile(ldat,95)
ax.imshow(ldat, vmin=z1, vmax=z2, cmap='Blues')
cxs = np.zeros_like(cys)+halfW
for ap, xo, yo in zip(cap, cxs, cys):
    ax.plot(xo, yo, 'r.')
    ax.text(xo, yo, ap, fontsize=10, color='r')
ax.grid()

# SAVE the coeffs. in the text file 
fout = open(APT_FILE, 'w')

# LOOP for aperture
for j in range(NAP)[:]:
    #LEFT SIDE for column
    xc, yc = [], [] 

    for k in [-1, 1]:
        nerr = 0
        # start point in mid-frame
        y, peak = cys[j], cpeaks[j]  
        for i in range(halfW, int(W*(1-k)/2)+k*dW, -k*2*dW):
            # PRE-FITTING with current points
            if len(yc) > 6:
                ctmp = chebfit(xc, yc, APT_ORDER)
                y = chebval(i, ctmp)
            # CROP the nearby box 
            ind1, ind2 = int(y-dH), int(y+dH)
            ycut = np.arange(ind1, ind2)
            vcut = np.median(dat[ind1:ind2,(i-dW):(i+dW)], axis=1)
            # FIND emission features
            y_tmp, peak_tmp = find_emission(ycut, vcut, thres=THRES)
            # if NO emission, SKIP or BREAK
            if len(y_tmp) == 0: 
                nerr += 1
                print ('AP%02d X=%i NO PEAKs' % (cap[j], i))
                if nerr > 5: break
                continue
           
            # SELECT the nearby peak
            mm = np.argmin((y_tmp-y)**2)
            y_tmp, peak_tmp = y_tmp[mm], peak_tmp[mm]
            '''
            # if WEAK emission, SKIP or BREAK
            if peak_tmp < THRES: 
                nerr += 1
                print '%i %i %i WEAK' % (j, i, peak_tmp)
                if nerr > 5: break
                continue
            '''
            xc.append(i)
            yc.append(y_tmp)
            y, peak = y_tmp, peak_tmp
       
    xc = np.array(xc)
    yc = np.array(yc)
    ss = np.argsort(xc)
    xc, yc = xc[ss], yc[ss]
    # FITTING with chebychev polynomials 
    coeffs = chebfit(xc, yc, APT_ORDER)
    fstr = '%5i ' % (cap[j],)
    for c in coeffs:
        fstr += '%15.8e ' % (c,)
    fout.write(fstr+' \n')

    # PLOT in the image 
    xfit = np.arange(W)
    yfit = chebval(xfit, coeffs)
    ax.plot(xfit, yfit, 'r-', lw=5, alpha=0.3)
     
    # PLOT of fitting results for each ap.
    if FIT_PLOT:
        fsub = plt.figure(num=99, figsize=(8,4))
        asub = fsub.add_subplot(111)
        yfit = chebval(xc, coeffs)
        rms = np.std(yc - yfit)
        asub.plot(xc, yc, 'k+', mew=1, ms=10)
        asub.plot(xc, yfit, 'r-', lw=3, alpha=0.5, label='RMS = %.5f' % (rms,))
        asub.set_xlabel('X [pixel]')
        asub.set_ylabel('Y [pixel]')
        asub.set_title('AP%02i Trace FIT' % (cap[j],))
        asub.grid()
        asub.legend()
        fsub.savefig('TRACEFIT-AP%02i' % (cap[j],))
        fsub.clf()
# CLOSE the file 
fout.close()
ax.set_xlim(0,W)
ax.set_ylim(H,0)   
ax.set_title('Diagram of APTRACE')
fig.savefig('diagram-aptrace.pdf')
plt.close('all')






