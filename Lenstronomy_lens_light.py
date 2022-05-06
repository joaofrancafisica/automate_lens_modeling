    # general configurations
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(numPix=100, # horizontal (or vertical number of pixels) 
                                                                                            deltapix=pixel_scale, # pixel scale
                                                                                            center_ra=0, # lens ra position
                                                                                            center_dec=0, # lens dec position
                                                                                            subgrid_res=1, # resoluton factor of our images
                                                                                            inverse=False) # invert east to west?
    lens_light_model = ['SERSIC_ELLIPSE'] # mass distribution to our lens model. 

    # some fit parameters
    #lenstronomy_mask = np.invert(mask)*1
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # setting our image class
    # data input parameters
    kwargs_data = {'background_rms': sky_rms,  # rms of background in ADUs
                   'exposure_time': expo_time,  # exposure time
                   'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
                   'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel 
                   'transform_pix2angle': Mpix2coord,  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
                   'image_data': np.zeros((100, 100))}  # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.

    data_class = ImageData(**kwargs_data) 
    data_class.update_data(cutout)
    kwargs_data['image_data'] = cutout

    # setting our priors
    ######## Lens ########
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []

    # initial guess, sigma, upper and lower parameters
    fixed_lens_light.append({})
    kwargs_lens_light_init.append({'R_sersic': 4., 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': 0, 'center_y': 0})
    kwargs_lens_light_sigma.append({'n_sersic': 2.5, 'R_sersic': 1., 'e1': 0.5, 'e2': 0.5, 'center_x': 1., 'center_y': 1.})
    kwargs_lower_lens_light.append({'e1': -1., 'e2': -1., 'R_sersic': 1., 'n_sersic': 0.1, 'center_x': -10, 'center_y': -10})
    kwargs_upper_lens_light.append({'e1': 1., 'e2': 1., 'R_sersic': 8., 'n_sersic': 8., 'center_x': 10, 'center_y': 10})

    # creating an object to have all this attributes
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
    kwargs_params = {'lens_light_model': lens_light_params}

    # Likelihood kwargs
    kwargs_likelihood = {'source_marg': False}
    kwargs_model = {'lens_light_model_list': lens_light_model} # Sersic, SIE, etc
    # here, we have 1 single band to fit
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]] # in this example, just a single band fit
    # if you have multiple  bands to be modeled simultaneously, you can append them to the mutli_band_list
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'multi-linear': every imaging band has independent solutions of the surface brightness, 'joint-linear': there is one joint solution of the linear coefficients demanded across the bands.
    # we dont have a constraint
    kwargs_constraints = {}

    # running an mcmc algorithm
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 200, 'n_iterations': 200}]]

    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()

    lens_light_model_class = LightModel(light_model_list=lens_light_model)

    imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image = imageModel.image(kwargs_lens_light=kwargs_result['kwargs_lens_light'])

    hdu = fits.PrimaryHDU(data=image)
    hdu.writeto('./fits_results/lens_light/'+str(name)+'_Lenstronomy[ELLSERSIC].fits')