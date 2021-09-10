# TADE
## Tools for Automatic Data reduction of Echelle spectra

Perform the data redcution of echelle spectra. (preferred for the eShel data obtained at Deokheung Optical Astronomy Observatory)
 - for the data of Shelyak eShel spectrograph, https://www.shelyak.com/description-eshel/?lang=en

The Goal of Project
 - Be able to perform spectral analysis without IRAF/Linux for the spectra obtained at DOAO (Deokheung Optical Astronomy Observatory).
 - Share the set of verified codes for an astronomical project.
 - Explain the data reduction process of echelle spectra for educational purpose.

## Components
 - speclib.py (functions for data reduction of echelle spectra)
 - tade.par (parameters of echelle spectrum image  
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

### Input the parameters in tade.par 
```    
WORKDIR  ./20201028   # path of data files
NAP      24           # number of aperture for tracing and extracting  
APTSTART 140          # aperture start point of y-axis in pixel (400/275/170)
APTTHRES 100          # aperture feature finding threshold  (100)
APTRANGE 50,1200      # aptracing range in the image 
APTDW    10           # 10, cut-width for aperture detection (dispersion axis) [pixel]
APTDH    6            # detection for aperture detection (order axis) [pixel]
APTPLOT  1            # flag for plotting fitting result
APTORDER 3            # polynomial fitting order for aperture tracing 
APTFILE  aptrace.dat  # data file of aperture tracing results
CRREJECT 10           # sigma for CR rejecting on aperture extracting [pixel]
APEXWID  4,2          # extraction inner-width for aperture extracting [pixel]
APEXPLOT 1            # flag for plotting aperture extracting result
EIDFILE  ecreid2.dat  # data file of wavelength calibration results
EIDPLOT  1            # flag for plotting spectrum of each order 
EIDTHRES 50           # threshold for Th-Ar emission detecting 
EIDGPIX  9            # width in pixel for emission detecting (>9)
EIDSTART 52           # starting order of comparison spectrum
EIDSCALE 0.6          # scaling factor w.r.t the template(FLI,26) (1.3/1.0/0.6)
EIDSHIFT -20          # shift factor w.r.t the template(FLI,26) (-260/-60/-20)
EIDORDER 7,7          # lambda-pixel chebyshev polynomial fitting order 
CONORDER 3            # polynomial fitting order for continuum fitting 
CONUPREJ 4.0          # upper factor for sigma clipping of continuum determination
CONLOREJ 1.0          # lower factor for sigma clipping of continuum determination
```

### Run TARES codes

 - run 01-run_specproc.py 
   - do preprocessing of the object images by bias, dark correction
   - generate the master flat image (iflat.fits)
   - generate the master comparison image (comp1.fits)
 - run 02-aptrace.py
   - do aperture tracing using the master flat image (iflat.fits)
   - generate the data file of aperture tracing result
   - generate the pdf and png files from the aperture tracing process
   - !!YOU SHOULD CHECK the result pdf and png files!!
   - ADJUST the parameters for aperture tracing (APTSTART, APTTHRES, APTRANGE, ... APTFILE)
 - run 03-apextract.py
   - do aperture extraction using APTFILE data
   - generate the spectrum file (*.ec.fits) by extraction for each image 
   - generate the png files showing aperture extraction results
   - !!YOU SHOULD CHECK the png files!!
   - ADJUST the parameters for aperture extraction (CRREJECT, APEXWID, ... )
 - run 04-ecidentify.py 
   - do wavelength calibration using the master comparison spectrum (comp1.ec.fits) and template (compFLI.ec.fits)
   - generate the data file of line-matching results b.t.w. master comparison and template (EIDFILE)
   - !!YOU SHOULD CHECK the png files of results!!
   - ADJUST the paramters (EIDTHRES, ... EIDORDER)
 - run 05-dispcor.py
   - apply the heliocentric velocity correction to the spectrum (by object name in FITS header)
   - calculate the wavelength solution for all apertures by polynomial fitting using EIDFILE line data
   - determine the local continuum in each aperture, and normalize it
   - place the spectra end to end for all apertures
   - generate the lpx data files of object spectra (wavelength, aperture numer, intensity)
   - generate the pdf image files of object spectra (graph)
   
