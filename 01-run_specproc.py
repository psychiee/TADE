# run FITS image preprocessing 
# BIAS, DARK, FLAT correction
# requirtments: wbias.list, wdark*.list, wflat*.list, wobj*.list 
# wskang
# 20170704

import numpy as np 
import time, os 
#from astropy.stats import sigma_clipped_stats
from astropy.io import fits 
from glob import glob 
from specutil import read_params

par = read_params()
WORKDIR = par['WORKDIR']
RANGE = np.array(par['APTRANGE'].split(','),int)
os.chdir(WORKDIR)

# FILTERING for binning option
BINNING = 1
# MAKE object + bias + dark + flat image file list 
# modify wild cards (*.fits) for your observation file name
flist = glob('*.fit')  

# DEFINE the list for information of FITS files 
TARGET, TYPE, DATEOBS, EXPTIME, FILTER, FNAME = [],[],[],[],[],[]
for fname in flist: 
    hdu = fits.open(fname)
    img, hdr = hdu[0].data, hdu[0].header
    if hdr.get('XBINNING') != BINNING: continue
    if hdr.get('YBINNING') != BINNING: continue
    # READ the FITS file header and INPUT into the list 
    TARGET.append(hdr.get('OBJECT'))
    TYPE.append(str.lower(str.strip(hdr.get('IMAGETYP'))))
    DATEOBS.append(hdr.get('DATE-OBS'))
    EXPTIME.append(hdr.get('EXPTIME'))
    #FILTER.append(str.strip(hdr.get('FILTER')))
    FNAME.append(fname)

# SORT the files for the observation time 
sort_list = np.argsort(DATEOBS) 
# LOOP for the all FITS images and 
# GENERATE the file list for each group 
for s in sort_list:
    fname = flist[s].lower()
    print (fname, DATEOBS[s], TYPE[s], EXPTIME[s], TARGET[s]  )
    # DEFINE the name of list file with FITS header info.  
    if fname.startswith('bias'): 
        lname = 'bias.list'
    elif fname.startswith('dark'): 
        lname = 'dark%is.list' % (EXPTIME[s],)
    elif fname.startswith('flat'):
        lname = 'flat.list'  
    elif fname.startswith('comp'):
        lname = 'comp.list'
    else:
        lname = 'obj.list'

    # WRITE the FITS file name into the LIST file 
    f = open(lname, 'a') 
    f.write(FNAME[s]+'\n')
    f.close()
    print ('add to '+lname+' ...' )
#==========================================================================
print ('Ready to Preprocessing by LISTs')
time.sleep(2)
    
# Make master bias ========================================================
if os.path.exists('bias.fits'):
    print ('bias.fits EXISTS... Read bias.fits...')
    hdu = fits.open('bias.fits')[0]
    master_bias = hdu.data 
else:
    bias_list = np.genfromtxt('bias.list',dtype='U') 
    bias_stack = []
    print ('# of total bias frames = ', len(bias_list))
    for fname in bias_list:
        print ('Read %s file ... ' % (fname,))
        hdu = fits.open(fname)[0]
        dat, hdr = hdu.data, hdu.header
        bias_stack.append(dat)
        print (fname, ' %8.1f %8.1f %8.1f %8.1f ' % \
             (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    print ('(Median) combine of bias frames ...' )
    master_bias = np.median(bias_stack, axis=0) 
    dat = master_bias
    print ('python %8.1f %8.1f %8.1f %8.1f' % \
          (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    
    print ('Save to bias.fits ...')  
    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    hdr.set('OBJECT', 'bias')
    fits.writeto('bias.fits', master_bias, hdr, overwrite=True)

    
# make master dark ========================================================
list_files = glob('dark*.list')
master_darks, exptime_darks = [], [] 
for lname in list_files:
    dark_list = np.genfromtxt(lname, dtype='U') 
    fidx = lname.split('.')[0]
    if os.path.exists(fidx+'.fits'):
        print (fidx+'.fits EXISTS... Read '+fidx+'.fits ...' )
        hdu = fits.open(fidx+'.fits')[0]
        master_dark = hdu.data
        exptime_dark = hdu.header.get('EXPTIME')
    else:
        print ('# of total dark frames(%s) = %i' % (fidx, len(dark_list)))
        dark_stack = [] 
        for fname in dark_list: 
            print ('Read %s file ...' % (fname,))
            hdu = fits.open(fname)[0]
            dat, hdr = hdu.data, hdu.header
            dark_stack.append(dat - master_bias) 
            print (fname, ' %8.1f %8.1f %8.1f %8.1f ' % \
               (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
        master_dark = np.median(dark_stack, axis=0)
        exptime_dark = hdr.get('EXPTIME')
        print ('Save to '+fidx+'.fits ...', hdr.get('IMAGETYP'), hdr.get('EXPTIME'))
        hdr.set('OBJECT',fidx)
        hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
        hdr.set('EXPTIME', exptime_dark)
        fits.writeto(fidx+'.fits', master_dark, hdr, overwrite=True)        

    dat = master_dark 
    print ('Python %8.1f %8.1f %8.1f %8.1f' % \
           (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    master_darks.append(master_dark)
    exptime_darks.append(exptime_dark)


# Do preprocessing of object images 
exptime_darks = np.array(exptime_darks)
flist = np.genfromtxt('obj.list', dtype='U').flatten()
flist = list(flist)
flist = flist + list(np.genfromtxt('comp.list', dtype='U'))
flist = flist + list(np.genfromtxt('flat.list', dtype='U'))
print ('# of total object, comp frames = ', len(flist))
for fname in flist:
    print ('Read %s file ...' % (fname,))
    hdu = fits.open(fname)[0]
    cIMAGE, hdr = hdu.data, hdu.header
    cEXPTIME = hdr.get('EXPTIME')
	
    # Find closest exposure time dark 
    dd = np.argmin(np.abs(exptime_darks - cEXPTIME))
    dEXPTIME = exptime_darks[dd]
    dFRAME = master_darks[dd]
    dFRAME = dFRAME  * (cEXPTIME/dEXPTIME)
	
    cIMAGE = cIMAGE - master_bias - dFRAME 
    # TRIM the images 
    cIMAGE = cIMAGE[:,RANGE[0]:RANGE[1]]

    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    fits.writeto('w'+fname, cIMAGE, hdr, overwrite=True)
	
    print ('Save to ', fname, '(', cEXPTIME,') << [', dEXPTIME,']')

# make master flats =======================================================
flat_list = np.genfromtxt('flat.list', dtype='U')
fidx = 'iflat'
if os.path.exists(fidx+'.fits'):
    print (fidx+'.fits EXISTS... DO NOTHING ...')
else:
    flat_stack = [] 
    print ('# of total flat frames = ', len(flat_list))
    for fname in flat_list:
        fname = 'w'+fname
        print ('Read %s file ...' % (fname,))
        hdu = fits.open(fname)[0]
        dat, hdr = hdu.data, hdu.header
        flat_stack.append(dat)
        print (fname, ' %8.1f %8.1f %8.1f %8.1f ' % \
            (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
    master_flat = np.median(flat_stack, axis=0)
    ## TRIM the images 
    #master_flat = master_flat[:,RANGE[0]:RANGE[1]]
    print ('Save to '+fidx+'.fits ...', hdr.get('IMAGETYP'))
    hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
    hdr.set('OBJECT', fidx)
    fits.writeto(fidx+'.fits', master_flat, hdr, overwrite=True)         

# (Option) Comparison rename ================================
comp_list = list(np.genfromtxt('comp.list', dtype='U'))
comp_stack = [] 
print ('# of total flat frames = ', len(flat_list))
for fname in comp_list:
    fname = 'w'+fname
    print ('Read %s file ...' % (fname,))
    hdu = fits.open(fname)[0]
    dat, hdr = hdu.data, hdu.header
    comp_stack.append(dat)
    print (fname, ' %8.1f %8.1f %8.1f %8.1f ' % \
        (np.mean(dat), np.std(dat), np.max(dat), np.min(dat)))
master_comp = np.mean(comp_stack, axis=0)
## TRIM the images 
#master_comp = master_comp[:,RANGE[0]:RANGE[1]]
print ('Save to comp1.fits ...', hdr.get('IMAGETYP'))
hdr.set('DATE-PRC', time.strftime('%Y-%m-%dT%H:%M:%S'))
hdr.set('OBJECT', 'comp1')
fits.writeto('comp1.fits', master_comp, hdr, overwrite=True)         
    
