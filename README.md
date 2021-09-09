# TARES
## Tool for Automated Routines of Echelle Spectra

Perform the data redcution of echelle spectra. (preferred for the data obtained at Deokheung Optical Astronomy Observatory)

The Goal of Project
 - Be able to perform spectral analysis without IRAF/Linux for the spectra obtained at DOAO (Deokheung Optical Astronomy Observatory).
 - Share the set of verified codes for an astronomical project.
 - Explain the data reduction process of echelle spectra for educational purpose.

## Components
 - specutils.py (functions for data reduction of echelle spectra)
 - tares.par (parameters of echelle spectrum image  
 - 01-run_specproc.py (run the automatic image preprocessing)
 - 02-aptrace.py (run the aperture tracing of echelle spectra)
 - 03-apextract.py (extract the apertures in the echelle spectrum image) 
 - 04-ecidentify.py (run the wavelength calibration of echelle spectra)
 - 05-dispcor.py (generate the data files from the echelle spectra with wavelength and intensity) 
 - compFLI.ecid (wavelength calibration data for echelle spectra obtained at DOAO)
 - compFLI.ec.fits (the spectrum of ThAr lamp, for the comparison of wavelenth calibration)
 - ecid-180314.pdf (the spectrum of ThAr lamp with the emission features and wavelength)

## How to use

### Prepare a set of image files from the echelle spectrograph at DOAO
 - bias-????.fit (bias images)
 - dark-????-????s.fit (dark images)
 - flat-????-?s.fit (flat lamp spectrum images)
 - comp-????-??s.fit (ThAr lamp spectrum images)
 - XXXX-????-????s.fit (object spectrum images: XXXX should be the object name)

### Input the parameters in tape.par 
```    
WORKDIR  ./20201028
NAP      24    # number of aperture for tracing and extracting  
APTSTART 140   # 400/275/170, aperture start point of y-axis for pixel
APTTHRES 100  # 100, feature finding threshold 
APTRANGE 50,1200 # aptracing range in the image 
APTDW    10    # 10, cut-width for aperture detection (dispersion axis)
APTDH    6     # detection for aperture detection (order axis) 
APTPLOT  1     # flag for plotting fitting result
APTORDER 3     # polynomial fitting order for aperture tracing 
APTFILE  aptrace.dat
CRREJECT 10    # sigma for CR rejecting on aperture extracting 
APEXWID1 4     # extraction inner-width for aperture extracting 
APEXWID2 2     # extraction outer-width for aperture extracting 
APEXPLOT 1     # flag for plotting aperture extracting result
EIDFILE  ecreid2.dat
EIDPLOT  1     # flag for plotting spectrum of each order 
EIDTHRES 50   # threshold for Th-Ar emission detecting 
EIDGPIX  9     # width in pixel for emission detecting (>9)
EIDSTART 52    # starting order of comparison spectrum
EIDSCALE 0.6   # 1.3/1.0/0.6 scaling factor w.r.t the template(FLI,26)
EIDSHIFT -20   # -260/-60/-20 shift factor w.r.t the template(FLI,26)
EIDORDER 7,7     # lambda-pixel; chebyshev polynomial fitting order 
CONORDER 3     # polynomial fitting order for continuum fitting 
CONUPREJ 4.0     # upper factor for sigma clipping of continuum determination
CONLOREJ 1.0     # lower factor for sigma clipping of continuum determination
```

### Run TARES codes

 - run 01-run_specproc.py 
   - 

