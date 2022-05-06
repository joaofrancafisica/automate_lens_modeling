import sys
import pandas as pd
import numpy as np
from astropy.io import fits
from lenstronomy.Data.psf import PSF
import os
import matplotlib.pyplot as plt
import pyimfit

# fits files pre path
image_pre_path = './lens_light_subtraction/'

# read the input system number
name = int(sys.argv[1])
print('System chosen to fit: ', name-1)

# search for this system in our dataframe
modelized_systems = pd.read_csv('./test_dataset.csv')
modelized_systems_select = modelized_systems[modelized_systems['OBJID-g'] == name]
# reset index for convenience 
modelized_systems_select.reset_index(inplace=True)

# select constants
pixel_scale = float(modelized_systems_select['pixel_scale-g'][0])
print('using a pixel scale of: ', pixel_scale)
sky_rms = float(modelized_systems_select['sky_brightness-i'][0])
print('using a sky rms of: ', sky_rms)
expo_time = float(modelized_systems_select['exposure_time-i'][0])
print('using an exposure time of: ', expo_time)
ccd_gain = float(modelized_systems_select['ccd_gain-i'][0])
print('using a ccd gain of: ', ccd_gain)
read_noise = float(modelized_systems_select['read_noise-i'][0])
print('using a read noise of: ', read_noise)

# reading our image data
## residual image
original_image = fits.open(image_pre_path+str(int(name))+'/'+str(int(name))+'.fits')[0].data
## psf
psf = fits.open(image_pre_path+str(int(name))+'/psf.fits')[0].data

# mask
mask = fits.open(image_pre_path+str(int(name))+'/mask.fits')[0].data

# setting our fit enviroment
## path to configuration file
imfitConfigFile = "./config_imfit/config_galaxy.dat"
## model description
model_desc = pyimfit.ModelDescription.load(imfitConfigFile)

# apply a fit
imfit_fitter = pyimfit.Imfit(model_desc, psf=psf)
imfit_fitter.fit(original_image, gain=ccd_gain, read_noise=read_noise, original_sky=sky_rms, mask=mask)
## only save if it has converged
if imfit_fitter.fitConverged is True:
    print('Fit converged: chi^2 = {0}, reduced chi^2 = {1}'.format(imfit_fitter.fitStatistic,
            imfit_fitter.reducedFitStatistic))
    bestfit_params = imfit_fitter.getRawParameters()
    print('Best-fit parameter values: ', bestfit_params)

    f = open('./lens_light_subtraction/lens_light_report_sph.csv', 'a') 
    f.write(str(name)+','+str(bestfit_params[0])+','+str(bestfit_params[1])+','+str(bestfit_params[2])+','+str(bestfit_params[3])+','+str(bestfit_params[4])+','+str(bestfit_params[5])+','+str(bestfit_params[6])+','+str(imfit_fitter.reducedFitStatistic)+'\n')

    f.close()
        
    # saving our results
    residual_image = original_image - imfit_fitter.getModelImage()
    hdu = fits.PrimaryHDU(data=residual_image)
    hdu.writeto('./lens_light_subtraction/'+str(name)+'/'+str(name)+'_ImFit_SPHSERSIC.fits')