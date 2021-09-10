#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:02:44 2018

@author: wskang
"""
import numpy as np
from scipy.signal import find_peaks_cwt, convolve
import scipy.optimize as so 
from astropy.io import fits
from numpy.polynomial.chebyshev import chebfit, chebval

def read_params():
    f = open('tade.par', 'r')
    par = {}
    for line in f:
        tmps = line.split('#')[0].split()
        if len(tmps) < 1: continue
        keyword = tmps[0]
        contents = ''.join(tmps[1:])
        par.update({keyword: contents})
    return par

def speccont(y, order=3, sig=[5,1], niter=25):
    '''
    Determine local continuum in an aperture
    Corretion the spectrum fluxes with the local continuum
    '''
    NPIX = len(y)
    x0, y0 = np.arange(NPIX)+1, y
    tx, ty = x0, y0 
    n = 1
    npts = 20
    while n < niter+1:
        weights = np.ones_like(tx)
        weights[0:10], weights[-10:] = 5, 5
        c = chebfit(tx, ty, order, w=weights)
        #c = chebfit(tx, ty, order)
        yfit = chebval(tx, c)
        ydelt = ty - yfit
        ysig = np.std(ydelt)
        ymed = np.median(ydelt)
        vv, = np.where((ydelt < ymed+ysig*sig[0]) & (ydelt > ymed-ysig*sig[1]))
        if len(vv) < NPIX/5: break 
        tx, ty = tx[vv], ty[vv]
#        npts = len(vv)
        n += 1
    ycont = chebval(x0, c)
    return y/ycont  

def speccomb(wv, spec, order=3, sig=[5,2]):
    '''
    Combine multi-spectrum into 1d-spectrum
    '''
    rwv, rspec, rap = speccrop(wv, spec)
    twv, tspec, tap = [], [], [] 
    for wv, spec, ap in zip(rwv, rspec, rap):    
        #ispec = speccont(spec, order=2, sig=[2,2])
        #ispec = speccont(ispec, order=order, sig=sig)
        ispec = speccont(spec, order=order, sig=sig)
        twv = twv + list(wv)
        tspec = tspec + list(ispec)
        tap = tap + list(ap)
        
    return np.array(twv), np.array(tspec), np.array(tap)

def speccrop(wv, spec):
    '''
    Combine multi-spectrum into 1d-spectrum
    '''
    NAP = wv.shape[0]
    #wv1 = np.min(wv, axis=1)
    #wv2 = np.max(wv, axis=1)
    mwv = np.mean(wv, axis=1)
    cwv = (mwv[:-1] + mwv[1:])/2.0
    #cwv = (wv2[:-1] + wv1[1:])/2.0
    
    rwv, rspec, rap = [], [], []
    for j in range(NAP):
        if j == 0:
            xstart = 500
            cc, = np.where(wv[j,:] < cwv[j])
            xend = max(cc)+1
        elif j == (NAP-1):
            cc, = np.where(wv[j,:] > cwv[j-1])
            xstart = min(cc)
            xend = -300
        else:
            cc, = np.where(wv[j,:] > cwv[j-1])
            xstart = min(cc)
            cc, = np.where(wv[j,:] < cwv[j])
            xend = max(cc)+1
            
        rspec.append(spec[j,xstart:xend].copy())
        rwv.append(wv[j,xstart:xend].copy())
        rap.append(np.zeros_like(wv[j,xstart:xend])+j+1)
        
    return rwv, rspec, rap

#############################################################################
# Spectrum utilities ===========================================================
# from https://bitbucket.org/nhmc/pyserpens/src/7826e643ad71/utilities.py
# nhmc / pyserpens by Neil Crighton
#############################################################################

def cr_reject(flux, nsigma=15.0, npix=3, verbose=False):
    """ Given flux and errors, rejects cosmic-ray type or dead
    pixels. These are defined as pixels that are more than
    nsigma*sigma above or below the median of the npixEL pixels on
    either side.

    Returns newflux,newerror where the rejected pixels have been
    replaced by the median value of npix to either side, and the
    error has been set to NaN.

    The default values work ok for S/N~20, Resolution=500 spectra.
    2018-02-06 wskang, modified without errors
    """
    if verbose:  print (nsigma,npix)
    #flux, error = list(flux), list(error)  # make copies
    flux, flag = list(flux), list(np.zeros_like(flux))
    i1 = npix
    i2 = len(flux) - npix
    for i in range(i1, i2):
        # make a list of flux values used to find the median
        fl = flux[i-npix:i] + flux[i+1:i+1+npix]
        er = flag[i-npix:i] + flag[i+1:i+1+npix]
        fl = [f for f,e in zip(fl,er) if e == 0]
        #er = [e for e in er if e > 0]
        medfl = np.median(fl)
        medsg = np.std(fl)
        if np.abs((flux[i] - medfl) / medsg) > nsigma:
            flux[i] = medfl
            flag[i] = flag[i]+1
            if verbose:  print (len(fl), len(er))

    return np.array(flux)
    
def find_emission(fx, fy, thres=200, width=9):
    sawtooth = np.array([0,-1,-2,-1,0,1,2,1,0])
    w0 = len(sawtooth)
    npix = len(fy)

    if npix < width:
        return [], []

    if (width % 2) == 0:
        width += 1
    if w0 >= width:
        width = w0
    else:
        d0 = np.arange(w0)/w0
        dp = np.arange(width)/width
        sawtooth = np.interp(dp, d0, sawtooth)
    # READ the middle column/row in the image
    sfy = np.r_[fy[(width-1):0:-1],fy,fy[-2:(-width-1):-1]]
    pcov = np.convolve(sawtooth, sfy, 'valid')[int(width/2):-int(width/2)]
    cx, cy = [], []
    for i in range(1, npix-1):
        p1, p2 = pcov[i-1], pcov[i]
        py = fy[i-1] + (fy[i]-fy[i-1])*(p1)/(p1-p2)
        px = fx[i-1] + (fx[i]-fx[i-1])*(p1)/(p1-p2)
        if (p1 < 0) & (p2 > 0) & (py > thres):
            cx.append(px)
            cy.append(py)
    return np.array(cx), np.array(cy)


def find_absorption(fx, fy, thres=0.5, width=13):
    sawtooth = np.array([0,-1,-2,-1,0,1,2,1,0])
    w0 = len(sawtooth)
    npix = len(fy)
    if (width % 2) == 0:
        width += 1 
    if w0 >= width: 
        width = w0
    else:
        d0 = np.arange(w0)/w0
        dp = np.arange(width)/width
        sawtooth = np.interp(dp, d0, sawtooth)
    # READ the middle column/row in the image 
    sfy = np.r_[fy[(width-1):0:-1],fy,fy[-2:(-width-1):-1]]
    pcov = np.convolve(sawtooth, sfy, 'valid')[int(width/2):-int(width/2)]
    cx, cy = [], [] 
    for i in range(1, npix-1):
        p1, p2 = pcov[i-1], pcov[i]
        py = fy[i-1] + (fy[i]-fy[i-1])*(p1)/(p1-p2)
        px = fx[i-1] + (fx[i]-fx[i-1])*(p1)/(p1-p2)
        if (p1 > 0) & (p2 < 0) & (py < thres):
            cx.append(px)
            cy.append(py)
    return np.array(cx), np.array(cy)

def smooth(x, width=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')
    if x.size < width:
        raise ValueError('Input vector needs to be bigger than window size.')
    if width < 3:
        return x 
    if not window in ['flat','hanning','hamming','barlett','blackman']:
        raise ValueError("Window is one of 'flat','hanning','hamming','barlett','blackman'")
    
    s = np.r_[x[(width-1):0:-1],x,x[-2:(-width-1):-1]]
    if window == 'flat':
        w = np.ones(width,'d')
    else:
        w = eval('np.'+window+'(width)')
        
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[int(width/2):-int(width/2)] 

def find_absorption0(x, y, width=5, thres=0.99, name='test'):
    # CHECK if width is odd
    if (width % 2) == 0: 
        width += 1 
    # CALC. deriv    
    yp = smooth(np.diff(y)/np.diff(x),width=width)
    xp = (x[1:]+x[:-1])/2.0
    ypp = smooth(np.diff(yp)/np.diff(xp),width=width)
    xpp = (xp[1:]+xp[:-1])/2.0
    yppp = smooth(np.diff(ypp)/np.diff(xpp),width=width)
    xppp = (xpp[1:]+xpp[:-1])/2.0
    
    # FIND 1st position for absorption line
    xcs= []
    for i in range(1,len(yppp)-1):
        p1, p2 = yppp[i-1], yppp[i]
        x1, x2 = xppp[i-1], xppp[i]
        dp, dx = abs(p2-p1), x2-x1
        if (p1 > 0) & (p2 < 0) & (dp > 1):
            xcs.append(x1 + dx*abs(p1/dp))
    xcs = np.array(xcs)
    # FIND derivs for 1st points
    yppcs = np.interp(xcs, xpp, ypp)
    ycs = np.interp(xcs, x, y)
    # CHECK the conditions for absorption line
    vv = np.where((ycs < thres) & (yppcs > max(ypp)/100.0))[0]

    return xcs[vv], ycs[vv]

def find_emission0(fx, fy, thres=200, width=9):
    """
    find the emission features in the spectrum
    ref: https://specutils.readthedocs.io/en/stable/_modules/specutils/fitting/fitmodels.html#find_lines_derivative
    """
    kernal = [1, 0, -1]
    # calc. the derivative using kernal
    dY = convolve(fy, kernal, 'valid')
    # check the sign of derivatives
    S = np.sign(dY)
    # find the sign flipping point
    ddS = convolve(S, kernal, 'valid')

    # the point which changes from positive to negative
    candidates = np.where(dY > 0)[0] + (len(kernal) - 1)
    line_inds = sorted(set(candidates).intersection(np.where(ddS == -2)[0] + 1))

    # find the peaks over threshold
    line_inds = np.array(line_inds)[fy[line_inds] > thres]

    # find the consecutive groups of peak points
    line_inds_grouped = np.split(line_inds, np.where(np.diff(line_inds) != 1)[0] + 1)

    if len(line_inds_grouped[0]) > 0:
        emission_inds = [inds[np.argmax(fy[inds])] for inds in line_inds_grouped]
    else:
        emission_inds = []
    return fx[emission_inds], fy[emission_inds]


#############################################################
#  READ IRAF spectrum FITS file
#############################################################    
"""readmultispec.py
Read IRAF (echelle) spectrum in multispec format from a FITS file.
Can read most multispec formats including linear, log, cubic spline,
Chebyshev or Legendre dispersion spectra.
Usage: retdict = readmultispec(fitsfile, reform=True)
Inputs:
fitfile     Name of the FITS file
reform      If true (the default), a single spectrum dimensioned
            [4,1,NWAVE] is returned as flux[4,NWAVE].  If false,
            it is returned as a 3-D array flux[4,1,NWAVE].
Returns a dictionary with these entries:
flux        Array dimensioned [NCOMPONENTS,NORDERS,NWAVE] with the spectra.
            If NORDERS=1, array is [NCOMPONENTS,NWAVE]; if NCOMPONENTS is also
            unity, array is [NWAVE].  (This can be changed
            using the reform keyword.)  Commonly the first dimension
            is 4 and indexes the spectrum, an alternate version of
            the spectrum, the sky, and the error array.  I have also
            seen examples where NCOMPONENTS=2 (probably spectrum and
            error).  Generally I think you can rely on the first element
            flux[0] to be the extracted spectrum.  I don't know of
            any foolproof way to figure out from the IRAF header what the
            various components are.
wavelen     Array dimensioned [NORDERS,NWAVE] with the wavelengths for
            each order.
header      The full FITS header from pyfits.
wavefields  [NORDERS] List with the analytical wavelength
            description (polynomial coefficients, etc.) extracted from
            the header.  This is probably not very useful but is
            included just in case.
History:
Created by Rick White based on my IDL readechelle.pro, 2012 August 15
Apologies for any IDL-isms that remain!
"""


def nonlinearwave(nwave, specstr, verbose=False):
    """Compute non-linear wavelengths from multispec string
    
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('Not nonlinear dispersion: dtype=' + fields[2])
    if len(fields) < 12:
        raise ValueError('Bad spectrum format (only %d fields)' % len(fields))
    #wt = float(fields[9])
    #w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:

        # cubic spline

        if len(fields) < 15:
            raise ValueError('Bad spline format (only %d fields)' % len(fields))
        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            print ('Dispersion is order-%d cubic spline' % npieces)
        if len(fields) != 15 + npieces + 3:
            raise ValueError('Bad order-%d spline format (%d fields)' % (npieces, len(fields)))
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave, dtype=float) + 1 - pmin) / (pmax - pmin) * npieces
        j = s.astype(int).clip(0, npieces - 1)
        a = (j + 1) - s
        b = s - j
        x0 = a ** 3
        x1 = 1 + 3 * a * (1 + a * b)
        x2 = 1 + 3 * b * (1 + a * b)
        x3 = b ** 3
        wave = coeff[j] * x0 + coeff[j + 1] * x1 + coeff[j + 2] * x2 + coeff[j + 3] * x3

    elif ftype == 1 or ftype == 2:

        # chebyshev or legendre polynomial
        # legendre not tested yet

        if len(fields) < 15:
            raise ValueError('Bad polynomial format (only %d fields)' % len(fields))
        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            if ftype == 1:
                print ('Dispersion is order-%d Chebyshev polynomial' % order)
            else:
                print ('Dispersion is order-%d Legendre polynomial (NEEDS TEST)' % order)
        if len(fields) != 15 + order:
            # raise ValueError('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
            if verbose:
                print ('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
                print ("Changing order from %i to %i" % (order, len(fields) - 15))
            order = len(fields) - 15
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        pmiddle = (pmax + pmin) / 2
        prange = pmax - pmin
        x = (np.arange(nwave, dtype=float) + 1 - pmiddle) / (prange / 2)
        p0 = np.ones(nwave, dtype=float)
        p1 = x
        wave = p0 * coeff[0] + p1 * coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2 * x * p1 - p0
            else:
                # legendre
                p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            wave = wave + p2 * coeff[i]
            p0 = p1
            p1 = p2

    else:
        raise ValueError('Cannot handle dispersion function of type %d' % ftype)

    return wave, fields


def readmultispec(fitsfile, reform=True, quiet=False):
    """Read IRAF echelle spectrum in multispec format from a FITS file
    
    Can read most multispec formats including linear, log, cubic spline,
    Chebyshev or Legendre dispersion spectra
    
    If reform is true, a single spectrum dimensioned 4,1,NWAVE is returned
    as 4,NWAVE (this is the default.)  If reform is false, it is returned as
    a 3-D array.
    """

    fh = fits.open(fitsfile)
    try:
        header = fh[0].header
        flux = fh[0].data
    finally:
        fh.close()
    temp = flux.shape
    nwave = temp[-1]
    if len(temp) == 1:
        nspec = 1
    else:
        nspec = temp[-2]

    # first try linear dispersion
    try:
        crval1 = header['crval1']
        crpix1 = header['crpix1']
        cd1_1 = header['cd1_1']
        ctype1 = header['ctype1']
        if ctype1.strip() == 'LINEAR':
            wavelen = np.zeros((nspec, nwave), dtype=float)
            ww = (np.arange(nwave, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                wavelen[i, :] = ww
            # handle log spacing too
            dcflag = header.get('dc-flag', 0)
            if dcflag == 1:
                wavelen = 10.0 ** wavelen
                if not quiet:
                    print ('Dispersion is linear in log wavelength')
            elif dcflag == 0:
                if not quiet:
                    print ('Dispersion is linear')
            else:
                raise ValueError('Dispersion not linear or log (DC-FLAG=%s)' % dcflag)

            if nspec == 1 and reform:
                # get rid of unity dimensions
                flux = np.squeeze(flux)
                wavelen.shape = (nwave,)
            return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': None}
    except KeyError:
        pass

    # get wavelength parameters from multispec keywords
    try:
        wat2 = header['wat2_*']
        #count = len(wat2)
    except KeyError:
        raise ValueError('Cannot decipher header, need either WAT2_ or CRVAL keywords')

    # concatenate them all together into one big string
    watstr = []
    for i in range(len(wat2)):
        # hack to fix the fact that older pyfits versions (< 3.1)
        # strip trailing blanks from string values in an apparently
        # irrecoverable way
        # v = wat2[i].value
        v = wat2[i]
        v = v + (" " * (68 - len(v)))  # restore trailing blanks
        watstr.append(v)
    watstr = ''.join(watstr)

    # find all the spec#="..." strings
    specstr = [''] * nspec
    for i in range(nspec):
        sname = 'spec' + str(i + 1)
        p1 = watstr.find(sname)
        p2 = watstr.find('"', p1)
        p3 = watstr.find('"', p2 + 1)
        if p1 < 0 or p1 < 0 or p3 < 0:
            raise ValueError('Cannot find ' + sname + ' in WAT2_* keyword')
        specstr[i] = watstr[p2 + 1:p3]

    wparms = np.zeros((nspec, 9), dtype=float)
    w1 = np.zeros(9, dtype=float)
    for i in range(nspec):
        w1 = np.asarray(specstr[i].split(), dtype=float)
        wparms[i, :] = w1[:9]
        if w1[2] == -1:
            raise ValueError('Spectrum %d has no wavelength calibration (type=%d)' %
                             (i + 1, w1[2]))
            # elif w1[6] != 0:
            #    raise ValueError('Spectrum %d has non-zero redshift (z=%f)' % (i+1,w1[6]))

    wavelen = np.zeros((nspec, nwave), dtype=float)
    wavefields = [None] * nspec
    for i in range(nspec):
        # if i in skipped_orders:
        #    continue
        verbose = (not quiet) and (i == 0)
        if wparms[i, 2] == 0 or wparms[i, 2] == 1:
            # simple linear or log spacing
            wavelen[i, :] = np.arange(nwave, dtype=float) * wparms[i, 4] + wparms[i, 3]
            if wparms[i, 2] == 1:
                wavelen[i, :] = 10.0 ** wavelen[i, :]
                if verbose:
                    print ('Dispersion is linear in log wavelength')
            elif verbose:
                print ('Dispersion is linear')
        else:
            # non-linear wavelengths
            wavelen[i, :], wavefields[i] = nonlinearwave(nwave, specstr[i],
                                                         verbose=verbose)
        wavelen *= 1.0 + wparms[i, 6]
        if verbose:
            print ("Correcting for redshift: z=%f" % wparms[i, 6])
    if nspec == 1 and reform:
        # get rid of unity dimensions
        flux = np.squeeze(flux)
        wavelen.shape = (nwave,)
        
    return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': wavefields}
	
#This is a mostly literal port of x_keckhelio.pro from XIDL (http://www.ucolick.org/~xavier/IDL/)

def x_helio(ra, dec, epoch=2000.0, jd=None, tai=None,
            longitude=None, latitude=None, altitude=None, obs='doao'):
    """
    `ra` and `dec` in degrees
    Returns `vcorr`: "Velocity correction term, in km/s, to add to measured
                    radial velocity to convert it to the heliocentric frame."
    but the sign seems to be backwards of what that says:
    helio_shift = -1. * x_keckhelio(RA, DEC, 2000.0)
    uses barvel and ct2lst functions from idlastro, also ported below
    #NOTE: this seems to have some jitter about the IDL version at the .1 km/s level
    """

    if longitude is not None and latitude is not None and altitude is not None:
        print ('using long/lat/alt instead of named observatory')
    elif obs == 'keck':
        longitude = 360. - 155.47220
        latitude = 19.82656886
        altitude = 4000.  #meters
    else:
        print ('Using observatory', obs)
        if obs == 'vlt':
            longitude = 360. - 70.40322
            latitude = -24.6258
            altitude = 2635.      #meters
        elif obs == 'mmt':
            longitude = 360. - 110.88456
            latitude = 31.688778
            altitude = 2600.      #meters
        elif obs == 'lick':
            longitude = 360. - 121.637222
            latitude = 37.343056
            altitude = 1283.      #meters
        elif obs == 'doao':
            longitude = 127.46750
            latitude = 34.53320
            altitude = 60.      #meters			
        else:
            raise ValueError('unrecognized observatory' + obs)

    if jd is None and tai is not None:
        jd = 2400000.5 + tai / (24. * 3600.)
    elif tai is None and jd is not None:
        pass
    else:
        raise ValueError('Must specify either JD or TAI')

    DRADEG = 180.0 / np.pi

    # ----------
    # Compute baryocentric velocity (Accurate only to 1m/s)
    dvelh, dvelb = baryvel(jd, epoch)

    #Project velocity toward star
    vbarycen = dvelb[0]*np.cos(dec/DRADEG)*np.cos(ra/DRADEG) + \
               dvelb[1]*np.cos(dec/DRADEG)*np.sin(ra/DRADEG) + dvelb[2]*np.sin(dec/DRADEG)

    #----------
    #Compute rotational velocity of observer on the Earth

    #LAT is the latitude in radians.
    latrad = latitude / DRADEG

    #Reduction of geodetic latitude to geocentric latitude (radians).
    #DLAT is in arcseconds.

    dlat = -(11. * 60. + 32.743000) * np.sin(2. * latrad) + \
            1.163300 * np.sin(4. * latrad) -0.002600 * np.sin(6. * latrad)
    latrad  = latrad + (dlat / 3600.) / DRADEG

    #R is the radius vector from the Earth's center to the observer (meters).
    #VC is the corresponding circular velocity
    #(meters/sidereal day converted to km / sec).
    #(sidereal day = 23.934469591229 hours (1986))

    r = 6378160.0 * (0.998327073 + 0.00167643800 * np.cos(2. * latrad) - \
       0.00000351 * np.cos(4. * latrad) + 0.000000008 * np.cos(6. * latrad)) \
       + altitude
    vc = 2. * np.pi * (r / 1000.)  / (23.934469591229 * 3600.)

    #Compute the hour angle, HA, in degrees
    LST = 15. * ct2lst(longitude, 'junk', jd) #  convert from hours to degrees
    HA = LST - ra

    #Project the velocity onto the line of sight to the star.
    vrotate = vc * np.cos(latrad) * np.cos(dec/DRADEG) * np.sin(HA/DRADEG)

    return (-vbarycen + vrotate)




def ct2lst(lng, tz, jd, day=None, mon=None, year=None):
    """
    # NAME:
    #     CT2LST
    # PURPOSE:
    #     To convert from Local Civil Time to Local Mean Sidereal Time.
    #
    # CALLING SEQUENCE:
    #     CT2LST, Lst, Lng, Tz, Time, [Day, Mon, Year] #NOT SUPPORTED IN PYTHON PORT!
    #                       or
    #     CT2LST, Lst, Lng, dummy, JD
    #
    # INPUTS:
    #     Lng  - The longitude in degrees (east of Greenwich) of the place for
    #            which the local sidereal time is desired, scalar.   The Greenwich
    #            mean sidereal time (GMST) can be found by setting Lng = 0.
    #     Tz  - The time zone of the site in hours, positive East  of the Greenwich
    #           meridian (ahead of GMT).  Use this parameter to easily account
    #           for Daylight Savings time (e.g. -4=EDT, -5 = EST/CDT), scalar
    #           This parameter is not needed (and ignored) if Julian date is
    #           supplied.    ***Note that the sign of TZ was changed in July 2008
    #           to match the standard definition.***
    #     Time or JD  - If more than four parameters are specified, then this is
    #               the time of day of the specified date in decimal hours.  If
    #               exactly four parameters are specified, then this is the
    #               Julian date of time in question, scalar or vector
    #
    # OPTIONAL INPUTS:
    #      Day -  The day of the month (1-31),integer scalar or vector
    #      Mon -  The month, in numerical format (1-12), integer scalar or vector
    #      Year - The 4 digit year (e.g. 2008), integer scalar or vector
    #
    # OUTPUTS:
    #       Lst   The Local Sidereal Time for the date/time specified in hours.
    #
    # RESTRICTIONS:
    #       If specified, the date should be in numerical form.  The year should
    #       appear as yyyy.
    #
    # PROCEDURE:
    #       The Julian date of the day and time is question is used to determine
    #       the number of days to have passed since 0 Jan 2000.  This is used
    #       in conjunction with the GST of that date to extrapolate to the current
    #       GST# this is then used to get the LST.    See Astronomical Algorithms
    #       by Jean Meeus, p. 84 (Eq. 11-4) for the constants used.
    #
    # EXAMPLE:
    #       Find the Greenwich mean sidereal time (GMST) on 2008 Jul 30 at 15:53 pm
    #       in Baltimore, Maryland (longitude=-76.72 degrees).   The timezone is
    #       EDT or tz=-4
    #
    #       IDL> CT2LST, lst, -76.72, -4,ten(15,53), 30, 07, 2008
    #
    #               ==> lst =  11.356505  hours  (= 11h 21m 23.418s)
    #
    #       The Web site  http://tycho.usno.navy.mil/sidereal.html contains more
    #       info on sidereal time, as well as an interactive calculator.
    # PROCEDURES USED:
    #       jdcnv - Convert from year, month, day, hour to julian date
    #
    # MODIFICATION HISTORY:
    #     Adapted from the FORTRAN program GETSD by Michael R. Greason, STX,
    #               27 October 1988.
    #     Use IAU 1984 constants Wayne Landsman, HSTX, April 1995, results
    #               differ by about 0.1 seconds
    #     Longitudes measured *east* of Greenwich   W. Landsman    December 1998
    #     Time zone now measure positive East of Greenwich W. Landsman July 2008
    #     Remove debugging print statement  W. Landsman April 2009
    """

    # IF N_params() gt 4 THEN BEGIN
    # time = tme - tz
    # jdcnv, year, mon, day, time, jd

    # ENDIF ELSE jd = double(tme)

    #
    #                            Useful constants, see Meeus, p.84
    #
    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0]
    jd2000 = 2451545.0
    t0 = jd - jd2000
    t = t0 / 36525
    #
    #                            Compute GST in seconds.
    #
    theta = c[0] + (c[1] * t0) + t ** 2 * (c[2] - t / c[3])
    #
    #                            Compute LST in hours.
    #
    lst = np.array((theta + lng) / 15.0)
    neg = lst < 0
    if np.sum(neg) > 0:
        if neg.shape == tuple():
            lst = 24. + idl_like_mod(lst, 24.)
        else:
            lst[neg] = 24. + idl_like_mod(lst[neg], 24.)
    return idl_like_mod(lst, 24.)

def baryvel(dje, deq):
#+
# NAME:
#       BARYVEL
# PURPOSE:
#       Calculates heliocentric and barycentric velocity components of Earth.
#
# EXPLANATION:
#       BARYVEL takes into account the Earth-Moon motion, and is useful for
#       radial velocity work to an accuracy of  ~1 m/s.
#
# CALLING SEQUENCE:
#       BARYVEL, dje, deq, dvelh, dvelb, [ JPL =  ]
#
# INPUTS:
#       DJE - (scalar) Julian ephemeris date.
#       DEQ - (scalar) epoch of mean equinox of dvelh and dvelb. If deq=0
#               then deq is assumed to be equal to dje.
# OUTPUTS:
#       DVELH: (vector(3)) heliocentric velocity component. in km/s
#       DVELB: (vector(3)) barycentric velocity component. in km/s
#
#       The 3-vectors DVELH and DVELB are given in a right-handed coordinate
#       system with the +X axis toward the Vernal Equinox, and +Z axis
#       toward the celestial pole.
#
# OPTIONAL KEYWORD SET:
#       JPL - if /JPL set, then BARYVEL will call the procedure JPLEPHINTERP
#             to compute the Earth velocity using the full JPL ephemeris.
#             The JPL ephemeris FITS file JPLEPH.405 must exist in either the
#             current directory, or in the directory specified by the
#             environment variable ASTRO_DATA.   Alternatively, the JPL keyword
#             can be set to the full path and name of the ephemeris file.
#             A copy of the JPL ephemeris FITS file is available in
#                 http://idlastro.gsfc.nasa.gov/ftp/data/
# PROCEDURES CALLED:
#       Function PREMAT() -- computes precession matrix
#       JPLEPHREAD, JPLEPHINTERP, TDB2TDT - if /JPL keyword is set
# NOTES:
#       Algorithm taken from FORTRAN program of Stumpff (1980, A&A Suppl, 41,1)
#       Stumpf claimed an accuracy of 42 cm/s for the velocity.    A
#       comparison with the JPL FORTRAN planetary ephemeris program PLEPH
#       found agreement to within about 65 cm/s between 1986 and 1994
#
#       If /JPL is set (using JPLEPH.405 ephemeris file) then velocities are
#       given in the ICRS system# otherwise in the FK4 system.
# EXAMPLE:
#       Compute the radial velocity of the Earth toward Altair on 15-Feb-1994
#          using both the original Stumpf algorithm and the JPL ephemeris
#
#       IDL> jdcnv, 1994, 2, 15, 0, jd          #==> JD = 2449398.5
#       IDL> baryvel, jd, 2000, vh, vb          #Original algorithm
#               ==> vh = [-17.07243, -22.81121, -9.889315]  #Heliocentric km/s
#               ==> vb = [-17.08083, -22.80471, -9.886582]  #Barycentric km/s
#       IDL> baryvel, jd, 2000, vh, vb, /jpl   #JPL ephemeris
#               ==> vh = [-17.07236, -22.81126, -9.889419]  #Heliocentric km/s
#               ==> vb = [-17.08083, -22.80484, -9.886409]  #Barycentric km/s
#
#       IDL> ra = ten(19,50,46.77)*15/!RADEG    #RA  in radians
#       IDL> dec = ten(08,52,3.5)/!RADEG        #Dec in radians
#       IDL> v = vb[0]*np.cos(dec)*np.cos(ra) + $   #Project velocity toward star
#               vb[1]*np.cos(dec)*sin(ra) + vb[2]*sin(dec)
#
# REVISION HISTORY:
#       Jeff Valenti,  U.C. Berkeley    Translated BARVEL.FOR to IDL.
#       W. Landsman, Cleaned up program sent by Chris McCarthy (SfSU) June 1994
#       Converted to IDL V5.0   W. Landsman   September 1997
#       Added /JPL keyword  W. Landsman   July 2001
#       Documentation update W. Landsman Dec 2005
#-
    #Define constants
    dc2pi = 2*np.pi
    cc2pi = dc2pi
    dc1 = 1.0
    dcto = 2415020.0
    dcjul = 36525.0                     #days in Julian year
    dcbes = 0.313
    dctrop = 365.24219572               #days in tropical year (...572 insig)
    dc1900 = 1900.0
    AU = 1.4959787e8

    #Constants dcfel(i,k) of fast changing elements.
    dcfel = [1.7400353e00, 6.2833195099091e02,  5.2796e-6 \
          ,6.2565836e00, 6.2830194572674e02, -2.6180e-6 \
          ,4.7199666e00, 8.3997091449254e03, -1.9780e-5 \
          ,1.9636505e-1, 8.4334662911720e03, -5.6044e-5 \
          ,4.1547339e00, 5.2993466764997e01,  5.8845e-6 \
          ,4.6524223e00, 2.1354275911213e01,  5.6797e-6 \
          ,4.2620486e00, 7.5025342197656e00,  5.5317e-6 \
          ,1.4740694e00, 3.8377331909193e00,  5.6093e-6 ]
    dcfel = np.array(dcfel).reshape(8,3)

    #constants dceps and ccsel(i,k) of slowly changing elements.
    dceps = [4.093198e-1, -2.271110e-4, -2.860401e-8 ]
    ccsel = [1.675104E-2, -4.179579E-5, -1.260516E-7 \
          ,2.220221E-1,  2.809917E-2,  1.852532E-5 \
          ,1.589963E00,  3.418075E-2,  1.430200E-5 \
          ,2.994089E00,  2.590824E-2,  4.155840E-6 \
          ,8.155457E-1,  2.486352E-2,  6.836840E-6 \
          ,1.735614E00,  1.763719E-2,  6.370440E-6 \
          ,1.968564E00,  1.524020E-2, -2.517152E-6 \
          ,1.282417E00,  8.703393E-3,  2.289292E-5 \
          ,2.280820E00,  1.918010E-2,  4.484520E-6 \
          ,4.833473E-2,  1.641773E-4, -4.654200E-7 \
          ,5.589232E-2, -3.455092E-4, -7.388560E-7 \
          ,4.634443E-2, -2.658234E-5,  7.757000E-8 \
          ,8.997041E-3,  6.329728E-6, -1.939256E-9 \
          ,2.284178E-2, -9.941590E-5,  6.787400E-8 \
          ,4.350267E-2, -6.839749E-5, -2.714956E-7 \
          ,1.348204E-2,  1.091504E-5,  6.903760E-7 \
          ,3.106570E-2, -1.665665E-4, -1.590188E-7 ]
    ccsel = np.array(ccsel).reshape(17,3)

    #Constants of the arguments of the short-period perturbations.
    dcargs = [5.0974222, -7.8604195454652e2 \
           ,3.9584962, -5.7533848094674e2 \
           ,1.6338070, -1.1506769618935e3 \
           ,2.5487111, -3.9302097727326e2 \
           ,4.9255514, -5.8849265665348e2 \
           ,1.3363463, -5.5076098609303e2 \
           ,1.6072053, -5.2237501616674e2 \
           ,1.3629480, -1.1790629318198e3 \
           ,5.5657014, -1.0977134971135e3 \
           ,5.0708205, -1.5774000881978e2 \
           ,3.9318944,  5.2963464780000e1 \
           ,4.8989497,  3.9809289073258e1 \
           ,1.3097446,  7.7540959633708e1 \
           ,3.5147141,  7.9618578146517e1 \
           ,3.5413158, -5.4868336758022e2 ]
    dcargs = np.array(dcargs).reshape(15,2)

    #Amplitudes ccamps(n,k) of the short-period perturbations.
    ccamps = \
    [-2.279594E-5,  1.407414E-5,  8.273188E-6,  1.340565E-5, -2.490817E-7 \
    ,-3.494537E-5,  2.860401E-7,  1.289448E-7,  1.627237E-5, -1.823138E-7 \
    , 6.593466E-7,  1.322572E-5,  9.258695E-6, -4.674248E-7, -3.646275E-7 \
    , 1.140767E-5, -2.049792E-5, -4.747930E-6, -2.638763E-6, -1.245408E-7 \
    , 9.516893E-6, -2.748894E-6, -1.319381E-6, -4.549908E-6, -1.864821E-7 \
    , 7.310990E-6, -1.924710E-6, -8.772849E-7, -3.334143E-6, -1.745256E-7 \
    ,-2.603449E-6,  7.359472E-6,  3.168357E-6,  1.119056E-6, -1.655307E-7 \
    ,-3.228859E-6,  1.308997E-7,  1.013137E-7,  2.403899E-6, -3.736225E-7 \
    , 3.442177E-7,  2.671323E-6,  1.832858E-6, -2.394688E-7, -3.478444E-7 \
    , 8.702406E-6, -8.421214E-6, -1.372341E-6, -1.455234E-6, -4.998479E-8 \
    ,-1.488378E-6, -1.251789E-5,  5.226868E-7, -2.049301E-7,  0.E0 \
    ,-8.043059E-6, -2.991300E-6,  1.473654E-7, -3.154542E-7,  0.E0 \
    , 3.699128E-6, -3.316126E-6,  2.901257E-7,  3.407826E-7,  0.E0 \
    , 2.550120E-6, -1.241123E-6,  9.901116E-8,  2.210482E-7,  0.E0 \
    ,-6.351059E-7,  2.341650E-6,  1.061492E-6,  2.878231E-7,  0.E0 ]
    ccamps = np.array(ccamps).reshape(15,5)

    #Constants csec3 and ccsec(n,k) of the secular perturbations in longitude.
    ccsec3 = -7.757020E-8
    ccsec = [1.289600E-6, 5.550147E-1, 2.076942E00 \
          ,3.102810E-5, 4.035027E00, 3.525565E-1 \
          ,9.124190E-6, 9.990265E-1, 2.622706E00 \
          ,9.793240E-7, 5.508259E00, 1.559103E01 ]
    ccsec = np.array(ccsec).reshape(4,3)

    #Sidereal rates.
    dcsld = 1.990987e-7                   #sidereal rate in longitude
    ccsgd = 1.990969E-7                   #sidereal rate in mean anomaly

    #Constants used in the calculation of the lunar contribution.
    cckm = 3.122140E-5
    ccmld = 2.661699E-6
    ccfdi = 2.399485E-7

    #Constants dcargm(i,k) of the arguments of the perturbations of the motion
    # of the moon.
    dcargm = [5.1679830,  8.3286911095275e3 \
           ,5.4913150, -7.2140632838100e3 \
           ,5.9598530,  1.5542754389685e4 ]
    dcargm = np.array(dcargm).reshape(3,2)

    #Amplitudes ccampm(n,k) of the perturbations of the moon.
    ccampm = [ 1.097594E-1, 2.896773E-7, 5.450474E-2,  1.438491E-7 \
           ,-2.223581E-2, 5.083103E-8, 1.002548E-2, -2.291823E-8 \
           , 1.148966E-2, 5.658888E-8, 8.249439E-3,  4.063015E-8 ]
    ccampm = np.array(ccampm).reshape(3,4)

    #ccpamv(k)=a*m*dl,dt (planets), dc1mme=1-mass(earth+moon)
    ccpamv = [8.326827E-11, 1.843484E-11, 1.988712E-12, 1.881276E-12]
    dc1mme = 0.99999696

    #Time arguments.
    dt = (dje - dcto) / dcjul
    tvec = np.array([1., dt, dt*dt])

    #Values of all elements for the instant(aneous?) dje.
    temp = idl_like_mod(idl_like_pound(tvec,dcfel), dc2pi)
    #PROBLEM: the mod here is where the 100 m/s error slips in
    dml = temp[:,0]
    forbel = temp[:,1:8]
    g = forbel[:,0]                         #old fortran equivalence

    deps = idl_like_mod(np.sum(tvec*dceps), dc2pi)
    sorbel = idl_like_mod(idl_like_pound(tvec, ccsel), dc2pi)
    e = sorbel[:, 0]                         #old fortran equivalence

    #Secular perturbations in longitude.
    #dummy = np.cos(2.0)
    sn = np.sin(idl_like_mod(idl_like_pound(tvec.ravel()[0:2] , ccsec[:, 1:3]),cc2pi))

    #Periodic perturbations of the emb (earth-moon barycenter).
    pertl = np.sum(ccsec[:,0] * sn) + dt*ccsec3*sn.ravel()[2]
    pertld = 0.0
    pertr = 0.0
    pertrd = 0.0

    for k in range(14):
        a = idl_like_mod((dcargs[k,0]+dt*dcargs[k,1]), dc2pi)
        cosa = np.cos(a)
        sina = np.sin(a)
        pertl = pertl + ccamps[k,0]*cosa + ccamps[k,1]*sina
        pertr = pertr + ccamps[k,2]*cosa + ccamps[k,3]*sina
        if k < 11:
            pertld = pertld + (ccamps[k,1]*cosa-ccamps[k,0]*sina)*ccamps[k,4]
            pertrd = pertrd + (ccamps[k,3]*cosa-ccamps[k,2]*sina)*ccamps[k,4]

    #Elliptic part of the motion of the emb.
    phi = (e*e/4)*(((8/e)-e)*np.sin(g) +5*np.sin(2*g) +(13/3)*e*np.sin(3*g))
    f = g + phi
    sinf = np.sin(f)
    cosf = np.cos(f)
    dpsi = (dc1 - e*e) / (dc1 + e*cosf)
    phid = 2*e*ccsgd*((1 + 1.5*e*e)*cosf + e*(1.25 - 0.5*sinf*sinf))
    psid = ccsgd*e*sinf * (dc1 - e*e)**-0.5

    #Perturbed heliocentric motion of the emb.
    d1pdro = dc1+pertr
    drd = d1pdro * (psid + dpsi*pertrd)
    drld = d1pdro*dpsi * (dcsld+phid+pertld)
    dtl = idl_like_mod((dml + phi + pertl), dc2pi)
    dsinls = np.sin(dtl)
    dcosls = np.cos(dtl)
    dxhd = drd*dcosls - drld*dsinls
    dyhd = drd*dsinls + drld*dcosls

    #Influence of eccentricity, evection and variation on the geocentric
    # motion of the moon.
    pertl = 0.0
    pertld = 0.0
    pertp = 0.0
    pertpd = 0.0
    for k in range(2):
        a = idl_like_mod((dcargm[k,0] + dt*dcargm[k,1]), dc2pi)
        sina = np.sin(a)
        cosa = np.cos(a)
        pertl = pertl + ccampm[k,0]*sina
        pertld = pertld + ccampm[k,1]*cosa
        pertp = pertp + ccampm[k,2]*cosa
        pertpd = pertpd - ccampm[k,3]*sina

    #Heliocentric motion of the earth.
    tl = forbel.ravel()[1] + pertl
    sinlm = np.sin(tl)
    coslm = np.cos(tl)
    sigma = cckm / (1.0 + pertp)
    a = sigma*(ccmld + pertld)
    b = sigma*pertpd
    dxhd = dxhd + a*sinlm + b*coslm
    dyhd = dyhd - a*coslm + b*sinlm
    dzhd= -sigma*ccfdi*np.cos(forbel.ravel()[2])

    #Barycentric motion of the earth.
    dxbd = dxhd*dc1mme
    dybd = dyhd*dc1mme
    dzbd = dzhd*dc1mme
    for k in range(3):
        plon = forbel.ravel()[k+3]
        pomg = sorbel.ravel()[k+1]
        pecc = sorbel.ravel()[k+9]
        tl = idl_like_mod((plon + 2.0*pecc*np.sin(plon-pomg)), cc2pi)
        dxbd = dxbd + ccpamv[k]*(np.sin(tl) + pecc*np.sin(pomg))
        dybd = dybd - ccpamv[k]*(np.cos(tl) + pecc*np.cos(pomg))
        dzbd = dzbd - ccpamv[k]*sorbel.ravel()[k+13]*np.cos(plon - sorbel.ravel()[k+5])

    #Transition to mean equator of date.
    dcosep = np.cos(deps)
    dsinep = np.sin(deps)
    dyahd = dcosep*dyhd - dsinep*dzhd
    dzahd = dsinep*dyhd + dcosep*dzhd
    dyabd = dcosep*dybd - dsinep*dzbd
    dzabd = dsinep*dybd + dcosep*dzbd

    #Epoch of mean equinox (deq) of zero implies that we should use
    # Julian ephemeris date (dje) as epoch of mean equinox.
    if deq == 0:
        dvelh = AU * ([dxhd, dyahd, dzahd])
        dvelb = AU * ([dxbd, dyabd, dzabd])
        return dvelh, dvelb

    #General precession from epoch dje to deq.
    deqdat = (dje-dcto-dcbes) / dctrop + dc1900
    prema = premat(deqdat,deq,FK4=True)

    dvelh = AU * idl_like_pound( prema, [dxhd, dyahd, dzahd] )
    dvelb = AU * idl_like_pound( prema, [dxbd, dyabd, dzabd] )

    return dvelh, dvelb

def premat(equinox1, equinox2, FK4=False):
    """
    #+
    # NAME:
    #       PREMAT
    # PURPOSE:
    #       Return the precession matrix needed to go from EQUINOX1 to EQUINOX2.
    # EXPLANTION:
    #       This matrix is used by the procedures PRECESS and BARYVEL to precess
    #       astronomical coordinates
    #
    # CALLING SEQUENCE:
    #       matrix = PREMAT( equinox1, equinox2, [ /FK4 ] )
    #
    # INPUTS:
    #       EQUINOX1 - Original equinox of coordinates, numeric scalar.
    #       EQUINOX2 - Equinox of precessed coordinates.
    #
    # OUTPUT:
    #      matrix - double precision 3 x 3 precession matrix, used to precess
    #               equatorial rectangular coordinates
    #
    # OPTIONAL INPUT KEYWORDS:
    #       /FK4   - If this keyword is set, the FK4 (B1950.0) system precession
    #               angles are used to compute the precession matrix.   The
    #               default is to use FK5 (J2000.0) precession angles
    #
    # EXAMPLES:
    #       Return the precession matrix from 1950.0 to 1975.0 in the FK4 system
    #
    #       IDL> matrix = PREMAT( 1950.0, 1975.0, /FK4)
    #
    # PROCEDURE:
    #       FK4 constants from "Computational Spherical Astronomy" by Taff (1983),
    #       p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
    #       Supplement 1992, page 104 Table 3.211.1.
    #
    # REVISION HISTORY
    #       Written, Wayne Landsman, HSTX Corporation, June 1994
    #       Converted to IDL V5.0   W. Landsman   September 1997
    #-
    """

    deg_to_rad = np.pi/180.0
    sec_to_rad = deg_to_rad/3600.

    T = 0.001 * ( equinox2 - equinox1)

    if not FK4: # FK5
        ST = 0.001*( equinox1 - 2000.)
        #  Compute 3 rotation angles
        A = sec_to_rad * T * (23062.181 + ST*(139.656 +0.0139*ST) \
            + T*(30.188 - 0.344*ST+17.998*T))

        B = sec_to_rad * T * T * (79.280 + 0.410*ST + 0.205*T) + A

        C = sec_to_rad * T * (20043.109 - ST*(85.33 + 0.217*ST) \
              + T*(-42.665 - 0.217*ST -41.833*T))

    else:
        ST = 0.001*( equinox1 - 1900.)
    #  Compute 3 rotation angles

        A = sec_to_rad * T * (23042.53 + ST*(139.75 +0.06*ST) \
            + T*(30.23 - 0.27*ST+18.0*T))

        B = sec_to_rad * T * T * (79.27 + 0.66*ST + 0.32*T) + A

        C = sec_to_rad * T * (20046.85 - ST*(85.33 + 0.37*ST) \
              + T*(-42.67 - 0.37*ST -41.8*T))

    sina = np.sin(A)
    sinb = np.sin(B)
    sinc = np.sin(C)
    cosa = np.cos(A)
    cosb = np.cos(B)
    cosc = np.cos(C)

    r = np.empty([3, 3])
    r[:,0] = [ cosa*cosb*cosc-sina*sinb, sina*cosb+cosa*sinb*cosc,  cosa*sinc]
    r[:,1] = [-cosa*sinb-sina*cosb*cosc, cosa*cosb-sina*sinb*cosc, -sina*sinc]
    r[:,2] = [-cosb*sinc, -sinb*sinc, cosc]

    return r

def idl_like_pound(a, b):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)

    if len(a.shape) == 2 and len(b.shape) == 1:
        return np.dot(a.T, b)
    if len(a.shape) == 1 and len(b.shape) == 2:
        res = np.dot(a, b.T)
        return res.reshape(1, res.size)
    else:
        return np.dot(a, b)

def idl_like_mod(a, b):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)
    res = np.abs(a) % b
    if a.shape == tuple():
        if a<0:
            return -res
        else:
            return res
    else:
        res[a<0] *= -1
    return res


def gaussian(x, background, peak, center, width):
    '''
    Generate the simple Gaussian function 

    def gaussian(x, amplitude, width, background = 0.0, xcenter = 0.0):

    coeff = amplitude
    idx = (x - xcenter)**2 / (2.*width**2)
    body = np.exp(-idx)
    return (coeff*body) + background
    '''
    #background, peak, center, width = p
    return peak*np.exp(-(x-center)**2 / (2.0*width**2)  ) + background
     
def gaussfit(x, y, p0=[0.,1.,0.,1.]):
    '''
    Find the simple Gaussian fitting function  
    '''
    coeff, var_matrix = so.curve_fit(gaussian, x, y, p0=p0)
    return list(coeff)

def JulDate(year,month,day): #,hour,minute,second,timezone):
    ''' 
    JD at 12:00 in a day
    '''
    year, month, day = int(year), int(month), int(day)

    a = int( (14 - month)/12 )
    y = int( year + 4800 - a )
    m = int( month +12*a - 3 )
    jd = day + (153*m + 2)/5 + 365*y + y/4 - y/100 + y/400 - 32045
 
    return jd    

def wavelength_to_rgb(wavelength, gamma=1):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B)	
