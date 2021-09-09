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
from scipy.optimize import curve_fit
from speclib import read_params, find_emission, smooth

# =============================================================
ECID_FILE = 'compFLI.ecid'
ECID_SPEC = 'compFLI.ec.fits'
# OPEN template information of emissions (from ecidentify)
dat = np.genfromtxt(ECID_FILE)
tap, tord = np.array(dat[:, 0], int), np.array(dat[:, 1], int)
tpix, twfit, tw0, tflag = dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 7]
tordset = list(set(tord))
tordset.sort(reverse=True)
# OPEN template spectrum
hdu = fits.open(ECID_SPEC)[0]
tspec = hdu.data
tx = np.arange(tspec.shape[1]) + 1
# ==============================================================

par = read_params()
os.chdir(par['WORKDIR'])

EID_OUTPUT = par['EIDFILE']
ORD_START = int(par['EIDSTART'])
SCALE, SHIFT = float(par['EIDSCALE']), float(par['EIDSHIFT'])
THRES, GPIX = int(par['EIDTHRES']), int(par['EIDGPIX'])
SPEC_PLOT = bool(int(par['EIDPLOT']))
TEST_PLOT = 0

# print sys.argv
if len(sys.argv) > 1:
    COMP_SPEC = sys.argv[1]
else:
    COMP_SPEC = 'comp1.ec.fits'

print('COMP_SPEC=%s' % (COMP_SPEC,))
print('THRES=%.1f, GPIX=%.1f' % (THRES, GPIX))
print('SPEC_PLOT=%i, TEST_PLOT=%i' % (SPEC_PLOT, TEST_PLOT))

# OPEN the FITS file of new spectrum
hdu = fits.open(COMP_SPEC)[0]
spec = hdu.data
# spec = spec[RANGE[0]:RANGE[1],:]
sx = np.arange(spec.shape[1]) + 1
# DEFINE the name of spectrum
FID = os.path.splitext(COMP_SPEC)[0]
# OPEN the line information of new spectrum
fout = open(EID_OUTPUT, 'w')

# LOOP for aperture of new spectrum
for j in range(0, spec.shape[0]):
    sord = ORD_START - j
    print('#AP', j + 1, ' #ORDER', sord)
    # FIND the order & aperture in the template
    if tordset.count(sord) == 0: continue
    tapnum = tordset.index(sord)

    NSTRONG = 40
    # EXTRACT the aperture in the template
    trow = tspec[tapnum, :]
    tcx, tcy = find_emission(tx, trow, thres=3 * THRES, width=GPIX)
    if len(tcx) > NSTRONG:
        ss = np.argsort(tcy)
        tcx, tcy = tcx[ss[-NSTRONG:]], tcy[ss[-NSTRONG:]]
    # EXTRACT the aperture in new spectrum
    srow = spec[j, :]
    ssig = np.std(srow) / 5
    scx, scy = find_emission(sx, srow, thres=3 * THRES, width=GPIX)
    if len(scx) > NSTRONG:
        ss = np.argsort(scy)
        scx, scy = scx[ss[-NSTRONG:]], scy[ss[-NSTRONG:]]
        # FITTING the new spectrum with template spectrum

    # FIND the factors of SCALE, SHIFT
    def conv(x, *p):
        a, b = p
        ox = tx * a + b
        oy = np.interp(x, ox, trow)
        oy[oy < THRES] = 0
        oy = oy / np.max(oy)
        oy = smooth(oy, width=GPIX)
        return oy

    srow[srow < THRES] = 0
    srow2 = srow / np.max(srow)
    srow2 = smooth(srow2, width=GPIX)
    c, cov = curve_fit(conv, sx, srow2, [SCALE, SHIFT], np.zeros_like(sx) + 0.01,
                       bounds=([SCALE * 0.8, SHIFT - 50], [SCALE * 1.2, SHIFT + 50]))
    a, b = c
    print('#factors by curve_fit')
    print('  SCALE   SHIFT')
    print('%7.4f %7.2f' % (a, b))

    # MATCHING the strong emissions
    strong_scx, strong_tcx = [], []
    for ix in tcx:
        dx = scx - (ix * a + b)
        mm = np.argmin(abs(dx))
        strong_scx.append(scx[mm])
        strong_tcx.append(ix)
        # fdx.append(dx)
    strong_scx, strong_tcx = np.array(strong_scx), np.array(strong_tcx)

    # CALIBRATE the template with matching emissions
    strong_num = len(strong_tcx)
    print('#matching strong lines:', strong_num)
    if strong_num == 0:
        print('NO LINES')
        continue
    if len(strong_scx) > 10:
        # FIND the correct matching lines by sigma-clipping
        print(' N pts    dx    sig')
        n = 1
        while n < 10:
            a, b = np.polyfit(strong_tcx, strong_scx, 1)
            dx = strong_scx - (strong_tcx * a + b)
            sig = np.std(dx)
            print(f'{n:2d} {len(strong_tcx):3d} {max(dx)-min(dx):5.1f} {sig:6.2f}')
            cond = abs(dx) < sig
            strong_tcx, strong_scx = strong_tcx[cond], strong_scx[cond]
            if len(strong_tcx) < 20: break
            if sig < 1: break
            if strong_num == len(strong_tcx): break
            strong_num = len(strong_tcx)
            n += 1
    # PRINT the scaling factors between new and template
    SCALE, SHIFT, XSIG = a, b, np.std(strong_scx - (strong_tcx * a + b))
    print('SCALE SHIFT   sig')
    print('%5.3f %5.1f %5.3f' % (SCALE, SHIFT, XSIG))

    # PLOT the matching result of spectrum
    if SPEC_PLOT:
        fig, ax = plt.subplots(num=99, figsize=(15, 6))
        ax.plot(tx * a + b, trow, 'g-', lw=4, alpha=0.5, label='TEMPLATE')
        ax.plot(sx, srow, 'b-', lw=1, label=FID)
        ax.plot(tcx * a + b, np.zeros_like(tcx) - 9000, 'g|', mew=7, ms=20, alpha=0.5)
        ax.plot(scx, np.zeros_like(scx) - 5000, 'b|', mew=3, ms=20, alpha=0.8)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Relative intensity')
        ax.set_ylim(-15000, 200000)
        ax.set_xlim(min(sx), max(sx))
        ax.set_title(f'''AP{j+1:02d} ORD{sord:02d} Transform Result of ThAr
(SCALE={SCALE:.4f}, SHIFT={SHIFT:.2f}, SIG={XSIG:.3f})''')
        ax.grid()
        ax.legend()
        fig.savefig(FID + '-CORR%02d.png' % (j + 1,))
        fig.clf()
    # FIND the positions of emissions in a spectrum
    vv, = np.where(tord == sord)
    # LOOP of lines
    slines, sdpix = [], []
    for vidx in vv:
        # read the ecidentify results
        tpeak0 = tpix[vidx]  # pixel position
        wv0 = tw0[vidx]  # wavelenth of ThAr emission
        flg = tflag[vidx]  # flag for calibration fitting
        # transform pixel position of template
        tpeak1 = tpeak0 * a + b  # for each line
        tx1 = tx * a + b  # for x-axis
        # define crop range
        xmin, xmax = tpeak1 - 2 * GPIX, tpeak1 + 2 * GPIX
        vvt, = np.where((tx1 < xmax) & (tx1 > xmin))
        vvs, = np.where((sx < xmax) & (sx > xmin))
        if len(vvs) == 0: continue
        # correct base level of emission
        cbox0 = trow[vvt] - np.min(trow[vvt])
        cbox1 = srow[vvs] - np.min(srow[vvs])
        # convert into same coordinates
        cx0, cx1 = tx1[vvt], sx[vvs]
        max0, max1 = max(cbox0), max(cbox1)
        ymax = np.max([max1, max0])
        # find peaks in new spec
        peaks, _ = find_emission(cx1, cbox1, thres=THRES, width=GPIX)
        if len(peaks) == 0: continue
        # find the closest line
        mm = np.argmin(abs(peaks - tpeak1))
        speak = peaks[mm]
        dpeak = speak - tpeak1

        # matching with the template emission line
        if abs(dpeak) < GPIX / 3:
            # save the result into the text file
            fstr = '%2d %2d %10.4f %8.3f %8.3f %8.5f %i' % \
                   (j + 1, sord, wv0, tpeak0, speak, dpeak, flg)
            fout.write(fstr + '\n')
            slines.append(int(speak))
            sdpix.append(dpeak)
        else:
            print('SKIP: %i %.2f' % (tpeak0, dpeak))

            if not os.path.exists('test'):
                os.mkdir('test')
            # plot the emissions
            fig, ax = plt.subplots(num=98, figsize=(7, 5))
            ax.plot(cx1, cbox1, 'bo-', label=FID)
            ax.plot(cx0, cbox0, 'go-', lw=2, ms=7, alpha=0.5, label='TEMPLATE')
            ax.plot([peaks, peaks], [0, ymax], 'b--', alpha=0.5)
            # mark the center position of new
            # if find_flag:
            #    ax.plot([mpeak, mpeak], [0, ymax], 'b-', lw=9, alpha=0.3)
            # mark the center of template
            ax.plot([tpeak1, tpeak1], [0, ymax], 'g-', lw=9, alpha=0.3)
            ax.set_xlim(xmin, xmax)
            ax.grid()
            ax.legend(loc='upper right')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Relative intensity')
            ax.set_title('AP%02d - Profile of ThAr %.4f' % (j + 1, wv0))
            fig.savefig('test/AP%02d-PEAK%04d' % (j + 1, tpeak0))
            fig.clf()
fout.close()
plt.close('all')