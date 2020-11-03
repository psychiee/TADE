#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:50:46 2018

@author: wskang
"""
import sys, os, time
import numpy as np 
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt 
from astropy.io import fits
#from astropy.modeling import models, fitting 
from glob import glob 
from specutil import read_params, cr_reject

par = read_params()
os.chdir(par['WORKDIR'])

CR_REJECT = int(par['CRREJECT'])
AP_PLOT = bool(int(par['APEXPLOT']))
#RANGE = np.array(par['APTRANGE'].split(','), int) 
AP_WID1 = int(par['APEXWID1']) # inner-width of aperture 
AP_WID2 = int(par['APEXWID2']) # outer-width of aperture 
APT_FILE = par['APTFILE'] #'aptrace.dat'

#print sys.argv

if len(sys.argv) > 3:
    FLIST = []
    # FLAT / COMP files 
    FLIST += sys.argv[1:3]
    # READ the list of files 
    if sys.argv[3].startswith('@'):
        FLIST += list(np.genfromtxt(sys.argv[3][1:], dtype='U'))
    else:
        FLIST += glob(sys.argv[3])
else:
    if os.path.exists('iflat.fits') & \
       os.path.exists('comp1.fits') & \
       os.path.exists('obj.list'): 
        FLIST = ['iflat.fits','comp1.fits']
        for f in np.genfromtxt('obj.list', dtype='U').flatten():
            FLIST.append('w'+f)
    else:
        print ('''python 03-apextract.py [FLAT filename] [COMP filename] [OBJECT LIST] ...
*** DEFAULT NAMES = iflat.fits comp1.fits w+[@obj.list]
*** [OBJECT LIST] = "@ap.list" or "wobj*.fits" or "test.fits"
*** IF NO FLAT & COMP, python 03-apextract.py none none [OBJECT LIST] ...''')

print ('CR_REJRECT=%i' %(CR_REJECT,))
print ('AP_PLOT=%i' %(AP_PLOT,))
print ('AP_WID= %i, %i' % (AP_WID1, AP_WID2))
print ('APT_FILE=%s' %(APT_FILE,))

# READ aptrace information 
dat = np.genfromtxt(APT_FILE)
aps = dat[:,0] # dat[RANGE[0]:RANGE[1],0]
coeffs = dat[:,1:]
NAP = len(aps)

FLATSPEC = None
# LOOT for EXTRACT apertures for each file 
for k, FNAME in enumerate(FLIST):
    FID = os.path.splitext(FNAME)[0]
    # if ec file EXISTS, SKIP
    if os.path.exists(FID+'.ec.fits'): 
        print ('(%i)%s.ec.fits exists ... SKIP' % (k, FID))
        # if it is FLAT, SAVE flat spectrum 
        if k == 0: 
            FLATSPEC = fits.open(FID+'.ec.fits')[0].data
        continue
    # CHECK the input file     
    if not os.path.exists(FNAME): 
        print ('(%1)%s does not exist ... SKIP' % (k, FNAME))
        continue 
    
    hdu = fits.open(FNAME)[0]
    img, hdr = hdu.data, hdu.header
    #img = img[RANGE[0]:RANGE[1],:]

    # INPUT header 
    hdr.set('DISPAXIS', 1)
    hdr.set('WCSDIM', 2)
    hdr.set('LTM1_1', 1.0)
    hdr.set('LTM2_2', 1.0)
    hdr.set('WAT0_001', 'system=equispec')
    hdr.set('WAT1_001', 'wtype=linear label=Pixel')
    hdr.set('WAT2_001', 'wtype=linear')
    hdr.set('CTYPE1', 'PIXEL')
    hdr.set('CTYPE2', 'LINEAR')
    hdr.set('CRVAL1', 1.)
    hdr.set('CRPIX1', 1.)
    hdr.set('CDELT1', 1.)
    hdr.set('CDELT2', 1.)
    hdr.set('CD1_1', 1.)
    hdr.set('CD2_2', 1.)
    OBJECT = hdr.get('OBJECT')
    H, W = img.shape 
    xp = np.arange(W) 
    print ('(%i)IMAGE: %s, DATA(%i, %i) %s' % (k,FNAME, H, W, OBJECT) )
    ecs = []
    for j in range(NAP):
        apnum = aps[j]
        c = coeffs[j,:]
        aprow = []
        for i in xp:
            yp = chebval(i, c)
            yint = int(yp)
            yflt = yp - yint
            row1 = img[(yint-AP_WID1):(yint+AP_WID2),i]
            row2 = img[(yint-AP_WID1+1):(yint+AP_WID2+1),i]
            total_row = row1*(1-yflt) + row2*yflt
            aprow.append(np.sum(total_row))
            
        # COSMIC RAY Rejection 
        # NSIGMA value = 0; no correction
        if CR_REJECT > 0: 
            caprow = cr_reject(aprow, nsigma=CR_REJECT)
        else:
            caprow = aprow
        
        # FLAT Correction for object files
        # 0:FLAT, 1:COMP, 2~:OBJECT
        if (k > 1) & (np.any(FLATSPEC) != None):
            caprow = caprow / FLATSPEC[j,:]
            aprow = aprow / FLATSPEC[j,:]
        ecs.append(caprow)
        
        yp0 = chebval(H/2, c)
        yp1, yp2 = yp0-AP_WID1, yp0+AP_WID2
        hdr.set('APNUM%i' % (apnum,), \
          '%i %i %.1f %.1f' % (apnum, apnum, yp1, yp2))

        if (k < 2) | AP_PLOT:
            fig, ax = plt.subplots(num=99, figsize=(15,6))
            ax.plot(aprow, 'g-', lw=4, alpha=0.5)
            ax.plot(caprow,'b-', lw=1)
            ax.grid()
            ax.set_xlabel('X [pixel]')
            ax.set_ylabel('Relative intensity')
            if k > 1: 
                ax.set_title('AP%02d EXTRACT %s / Flat Correction' % (apnum, FID))
            else:
                ax.set_title('AP%02d EXTRACT %s ' % (apnum, FID))
            fig.savefig(FID+'-EXAP%02d.png' % (apnum,))
            fig.clf()
        
    ecdat = np.array(ecs)
    hdr.set('AP-WID', '%i, %i' % (AP_WID1,AP_WID2), comment='Aperture width in pixel')
    # (case)FLAT, SAVE the flat spectrum into flatspec
    if k == 0:
        hdr.set('PROC', 'PySpecW.apextract FLAT')
        FLATSPEC = ecdat ### SAVE FLAT spectrum 
    elif k == 1: 
        hdr.set('PROC', 'PySpecW.apextract COMP')
    else:
        hdr.set('PROC', 'PySpecW.apextract + FLAT CORRECTION')
    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    fits.writeto(FID+'.ec.fits', ecdat, hdr, overwrite=True)

plt.close('all')    
        






