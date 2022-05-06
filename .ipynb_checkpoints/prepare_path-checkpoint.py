import sys
import pandas as pd
import numpy as np
from astropy.io import fits
import subprocess
import utils
import autolens as al
import shlex
import os
import time
from lenstronomy.Data.psf import PSF

# setting our simulations path
simulations_pre_path = './simulations/fits_files/i/'
config_sextractor_path = './config_sextractor/'

# read the input system number
name = int(sys.argv[1])

# search for this system in our dataframe
modelized_systems = pd.read_csv('./test_dataset.csv')
modelized_systems_select = modelized_systems[modelized_systems['OBJID-g'] == name]
# reset index for convenience 
modelized_systems_select.reset_index(inplace=True)

# reading all constants
objid = int(modelized_systems_select['OBJID-g'][0])
pixel_scale = float(modelized_systems_select['pixel_scale-g'][0])
seeing = float(modelized_systems_select['seeing-i'][0])
sky_rms = float(modelized_systems_select['sky_brightness-i'][0])
ccd_gain = float(modelized_systems_select['ccd_gain-i'][0])
mag_zero_point = float(modelized_systems_select['magnitude_zero_point-i'][0])
num_pix = int(modelized_systems_select['numPix-i'][0])
expo_time = float(modelized_systems_select['exposure_time-i'][0])
read_noise = float(modelized_systems_select['read_noise-i'][0])
n_exposures = float(modelized_systems_select['num_exposures-i'][0])
# setting some usefull ones
shape = (num_pix, num_pix)
mag_gamma = 10**(-0.4*(mag_zero_point-sky_rms))
ncombine = expo_time/n_exposures

# read an image
image_i = fits.open(simulations_pre_path+str(objid-1)+'.fits')[0].data

# make a dir 
os.makedirs('./lens_light_subtraction/'+str(objid), exist_ok=True)

# check if it has lens light
teste = utils.lens_light_test(image_i)
has_lens_light, f, x, y = teste.test()

# if it has, set our mask and galfit files
if has_lens_light:
    # finder for our inner mask (only the lens light, we will add this one to the sextractor one)
    finder = utils.find_radius(pre_set_sigma=3, image_array=image_i)
    norm, center, mask_radius= finder.get_radius() # gaussian normalization, center position (1-d) and radius (sigma)
    # check if the algorithm converged in order to achieve a valid value of radius
    radius_value = mask_radius*float(pixel_scale)
    if radius_value > 3.:
        # if not, set a mask with radius equals to 3
        radius_value = 3.
        
    # circular inner mask
    mask = al.Mask2D.circular(shape_native=shape, pixel_scales=pixel_scale, radius=radius_value)
    
    # apply sextractor
    ## original image path
    fits_path = simulations_pre_path+str(objid-1)+'.fits'
    # #moving the original image to sextractor path temporarly 
    subprocess.call(['mv', fits_path, config_sextractor_path])
    ## sextractor cmd command to have a segmentationf ile
    sextractor_cmd = 'sex ' + str(str(objid-1)+ '.fits -c ./default.sex -DETECT_THRESH ' + str(np.var(image_i)) + ' -ANALYSIS_THRESH ' + str(np.mean(image_i)) + ' -MAG_ZEROPOINT ' + str(mag_zero_point) + ' -MAG_GAMMA ' + str(mag_gamma) + ' -GAIN ' + str(ccd_gain) + ' -PIXEL_SCALE ' + str(pixel_scale) + ' -SEEING_FWHM ' + str(seeing))
    
    ## call the command
    subprocess.Popen(shlex.split(sextractor_cmd), cwd="/home/joao/Update_ALF/config_sextractor/")
    ## time to sleep running time for galfit
    time.sleep(2)
    ## segmentation file
    check=fits.open('./config_sextractor/check.fits')[0].data
    ## set a boolean mask (segmentation file sometimes has 1, 2, 3, etc values)
    for i in range(0, len(check)):
        for j in range(0, len(check[i])):
            if check[i][j] > 0:
                # if sextractor detected something set 1 (a mask)
                check[i][j] = 1
    ## resulting mask as circular for lens plus segmentation file            
    mask = mask + check
    ## another boolean check
    for i in range(0, len(mask)):
        for j in range(0, len(mask[i])):
            if mask[i][j] > 1:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    ## moving the original image to its original path            
    subprocess.call(['mv', config_sextractor_path+str(str(objid-1))+'.fits', simulations_pre_path])
    
    # exporting our mask
    hdu_mask = fits.PrimaryHDU(data=mask*1)
    hdu_mask.writeto('./lens_light_subtraction/'+str(objid)+'/mask.fits')
    
    # also exporting our psf
    ## setting lenstronomy kwargs by using seeeing
    kwargs_psf = {'psf_type': 'GAUSSIAN',
                  'fwhm': seeing,
                  'pixel_size': pixel_scale,
                  'truncation': 4/seeing}
    psf_class = PSF(**kwargs_psf)
    ## this line garaties a uniform shape for all images (because they have different seeing values)
    psf = psf_class.kernel_point_source/np.max(psf_class.kernel_point_source)
    ## finally exporting
    hdu_psf = fits.PrimaryHDU(data=psf)
    hdu_psf.writeto('./lens_light_subtraction/'+str(objid)+'/psf.fits')
    
    # exporting a image that has exposure time, gain, read noise and n-combine in its header (this is a must have for galfit)
    hdu_image_i = fits.PrimaryHDU(data=image_i)
    hdu_header = hdu_image_i.header
    hdu_header['EXPTIME'] = str(expo_time)
    hdu_header['GAIN'] = str(ccd_gain)
    hdu_header['RDNOISE'] = str(read_noise)
    hdu_header['NCOMBINE'] = str(ncombine)
    ## finally exporting
    hdu_image_i.writeto('./lens_light_subtraction/'+str(objid)+'/'+str(objid)+'.fits')
    
    # setting galfit feedme configuration file
    f = open('./lens_light_subtraction/'+str(objid)+'/galfit_sph.feedme', 'a') 
    f.write('\n')
    f.write('=============================================================================== \n')
    f.write('A)  /home/joao/Update_ALF/lens_light_subtraction/'+str(objid)+'/'+str(objid)+'.fits # Input data image (FITS file) \n'+
            'B)  /home/joao/Update_ALF/lens_light_subtraction/'+str(objid)+'/imgblock.fits       # Output data image block \n'+
            'C) none                # Sigma image name (made from data if blank or "none") \n'+
            'D)  /home/joao/Update_ALF/lens_light_subtraction/'+str(objid)+'/psf.fits   #        # Input PSF image and (optional) diffusion kernel \n'+
            'E) 1                   # PSF fine sampling factor relative to data  \n'+
            'F)  /home/joao/Update_ALF/lens_light_subtraction/'+str(objid)+'/mask.fits           # Bad pixel mask (FITS image or ASCII coord list) \n'+
            'G) none                # File with parameter constraints (ASCII file)  \n'+
            'H) 1    100   1    100 # Image region to fit (xmin xmax ymin ymax) \n'+
            'I) 100    100          # Size of the convolution box (x y) \n'+
            'J) '+str(mag_zero_point)+' # Magnitude photometric zeropoint  \n'+
            'K) 0.038  0.038        # Plate scale (dx dy)    [arcsec per pixel] \n'+
            'O) regular             # Display type (regular, curses, both) \n'+
            'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n')
    f.write('\n')
    f.write(
            '# INITIAL FITTING PARAMETERS \n'+
            '# \n'+
            '#   For object type, the allowed functions are: \n'+
            '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n'+
            '#       ferrer, powsersic, sky, and isophote. \n'+
            '# \n'+
            '#   Hidden parameters will only appear when theyre specified: \n'+
            '#       C0 (diskyness/boxyness),  \n'+
            '#       Fn (n=integer, Azimuthal Fourier Modes), \n'+
            '#       R0-R10 (PA rotation, for creating spiral structures). \n'+
            '# \n'+ 
            '# -----------------------------------------------------------------------------\n'+
            '#   par)    par value(s)    fit toggle(s)    # parameter description \n'+
            '# -----------------------------------------------------------------------------\n'+
            '# Object number: 1 \n'+
            '0) sersic                 #  object type \n'+
            '1) 50.  50.  1 1          #  position x, y \n'+
            '3) 20.0890     1          #  Integrated magnitude \n'+
            '4) 8.4      1             #  R_e (half-light radius)   [pix] \n'+
            '5) 2.3      1             #  Sersic index n (de Vaucouleurs n=4) \n'+
            '6) 0.0000      0          #     ----- \n'+
            '7) 0.0000      0          #     ----- \n'+
            '8) 0.0000      0          #     ----- \n'+
            '9) 1.      0              #  axis ratio (b/a) \n'+
            '10) 90.0    1             #  position angle (PA) [deg: Up=0, Left=90] \n'+
            'Z) 0                      #  output option (0 = resid., 1 = Dont subtract) \n'
            )
    f.write('\n')
    f.write(
            '# Object number: 2 \n'+
            '0) sky                    #  object type \n'+
            '1) 0.0000      1          #  sky background at center of fitting region [ADUs] \n'+
            '2) 0.0000      0          #  dsky/dx (sky gradient in x) \n'+
            '3) 0.0000      0          #  dsky/dy (sky gradient in y) \n'+
            'Z) 0                      #  output option (0 = resid., 1 = Dont subtract) \n'
            )
    f.close()
