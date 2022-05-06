import numpy as np
from scipy.optimize import minimize

class lens_light_test:
    def __init__(self, image_array):
        self.image_array=np.array(image_array)
    def test(self):
        image_bellow_zero=-self.image_array
        min_value=np.min(image_bellow_zero)
        value_50 = image_bellow_zero[50][50]
        #print(value_50/min_value)
        if value_50/min_value<0.3:
            [x_index], [y_index] = np.where(image_bellow_zero==min_value)
            return False, value_50/min_value, x_index, y_index
        else:
            [x_index], [y_index] = np.where(image_bellow_zero==min_value)
            return True, value_50/min_value, x_index, y_index
        
class find_radius:
    def __init__(self, pre_set_sigma, image_array):
        #axis_integrated_image=np.sum(image_array, axis=0)
        axis_integrated_image=image_array[50]
        over_mean_int_image=axis_integrated_image[50-pre_set_sigma:50+pre_set_sigma+1]
        self.y=over_mean_int_image/np.max(over_mean_int_image)
        self.x=np.linspace(50-pre_set_sigma, 50+pre_set_sigma, 2*pre_set_sigma+1, dtype=int)
    
    def gauss_func(self, x_val, norm_val, x0_val, sigma_val):
        return norm_val*np.exp(-0.5*((x_val-x0_val)/sigma_val)**2)

    def chi_squared(self, par):
        norm, x0, sigma = par
        gauss_array = self.gauss_func(self.x, norm, x0, sigma)
        return np.sum((self.y-gauss_array)**2)
    
    def get_radius(self, init_guess=[1., 50, 2], method='Nelder-Mead'):
        result=minimize(self.chi_squared, init_guess, method=method)
        return result.x
    
class rotate_matrix:
    def __init__(self, matrix):
        self.matrix = matrix
    def rotate(self):
        return [[self.matrix[jj][ii] for jj in range(len(self.matrix))] for ii in range(len(self.matrix[0])-1,-1,-1)]
    
