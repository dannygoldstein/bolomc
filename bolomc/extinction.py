__whatami__ = 'Dust extinction in tensorflow.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'

import tensorflow as tf

def fitzpatrick99(wave, ebv, r_v=3.1):
    """Fitzpatrick (1999) dust extinction function for arbitrary R_V.
    
    Parameters
    ----------
    
    wave: tf.Tensor, shape (nwave,) 
        Wavelengths at which to evaluate the reddening law. 
    
    ebv: tf.Tensor, shape (), or float
        The extinction E(B-V) parameter of the reddening law. 
    
    r_v: tf.Tensor, shape (), or float
        The R_V parameter of the reddening law. 
    
    Returns
    -------
    
    trans: tf.Tensor, shape (nwave,)
        The transmissivity of the reddening law, i.e.,
    
            F_trans = trans * F_incident
    
        Where F_trans and F_incident are both monochromatic flux
        densities.
    """
    pass
 
    
def od94(wave, ebv, r_v):
    """O'Donnell (1994) dust extinction function. 
    
    Parameters
    ----------
    
    wave: tf.Tensor, shape (nwave,) 
        Wavelengths at which to evaluate the reddening law. 
    
    ebv: tf.Tensor, shape (), or float
        The extinction E(B-V) parameter of the reddening law. 
    
    r_v: tf.Tensor, shape (), or float
        The R_V parameter of the reddening law. 
    
    Returns
    -------
    
    trans: tf.Tensor, shape (nwave,)
        The transmissivity of the reddening law, i.e.,
    
            F_trans = trans * F_incident
    
        Where F_trans and F_incident are both monochromatic flux
        densities.
    """
    pass
