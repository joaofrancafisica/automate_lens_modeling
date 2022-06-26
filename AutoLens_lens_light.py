import sys
import pandas as pd
import numpy as np
from astropy.io import fits
from lenstronomy.Data.psf import PSF
import autofit as af
import autolens as al
import autolens.plot as aplt
import os
import matplotlib.pyplot as plt

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
zl = float(modelized_systems_select['PLANE_1-REDSHIFT-g'][0])
zs = float(modelized_systems_select['PLANE_2-REDSHIFT-g'][0])
print('lens and source redshift: '+str(round(zl, 2))+' & '+str(round(zs, 2)))

# reading our image data
## residual image
original_image = al.Array2D.from_fits(file_path=image_pre_path+str(int(name))+'/'+str(int(name))+'.fits', pixel_scales=pixel_scale, hdu=0)
## noise map
noise_map = np.sqrt(original_image*expo_time+sky_rms)/expo_time
## psf
psf = al.Kernel2D.from_fits(file_path=image_pre_path+str(int(name))+'/psf.fits', hdu=0, pixel_scales=pixel_scale)

# masks
## mask
mask = al.Mask2D.from_fits(image_pre_path+str(int(name))+'/mask.fits', pixel_scales=pixel_scale)
## to avoid border problems on convolving the psf lets apply a big circular mask
circ_mask = al.Mask2D.circular(shape_native=(100, 100), pixel_scales=pixel_scale, radius=10.)

# imaging object
imaging = al.Imaging(image=original_image, noise_map=noise_map, psf=psf)

# apply a mask to imaging object
imaging = imaging.apply_mask(mask=circ_mask+mask)

# setting our model
## source galaxy model
source_galaxy_model = af.Model(al.Galaxy, redshift=zs)
## lens galaxy model

lens_galaxy_model = af.Model(al.Galaxy, redshift=zl, bulge=al.lmp.EllSersic) # EllSersic or SphSersic

# model object
lens_light_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))

# Fit
## search object
search = af.DynestyStatic(path_prefix='./',
                          name = str(name),
                          unique_tag = 'LensLight_ELLSERSIC',
                          nlive = 50,
                          number_of_cores = 4) # be carefull here! verify your core numbers
## analysis object
analysis = al.AnalysisImaging(dataset=imaging)
## results object
result = search.fit(model=lens_light_model, analysis=analysis)

# saving our results
## residual image which is the original subtracted from the lens light
residual_image = original_image - result.unmasked_model_image
residual_image.output_to_fits('./lens_light_subtraction/'+str(name)+'/'+str(name)+'_AutoLens_ELLSERSIC.fits', overwrite=True)
