__whatami__ = 'Dust extinction in tensorflow.'
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__all__ = ['fitzpatrick99', 'od94']

import tensorflow as tf

# Optical coefficients for CCM89-like laws:
ccm89_coeffs_n = 8
ccm89_coeffs_a = tf.constant([1., 0.17699, -0.50447, -0.02427, 0.72085,
                              0.01979, -0.77530, 0.32999])
ccm89_coeffs_b = tf.constant([0., 1.41338, 2.28305, 1.07233, -5.38434,
                              -0.62251, 5.30260, -2.09002])
od94_coeffs_n = 9
od94_coeffs_a = tf.constant([1., 0.17699, -0.50447, -0.02427, 0.72085,
                             0.01979, -0.77530, 0.32999])
od94_coeffs_b = tf.constant([0., 1.952, 2.908, -3.989, -7.985, 11.102,
                             5.491, -10.805, 3.347])

def _ccm89like(wave, 

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
    
    a_v = ebv * r_v
    
    

    pass
