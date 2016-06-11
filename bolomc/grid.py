
__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Warping surface grid."

import numpy as np


class Mesh(np.ndarray):

    def log(self, axes):
        cp = self.copy()
        cp[..., axes] = np.log10(cp[..., axes])
        return cp 
        
        
    def exp(self, axes):
        cp = self.copy()
        cp[..., axes] = 10**self[..., axes]
        return cp 
        

class Grid(object):
    
    def __init__(self, dim_body, min_body, max_body):
        
        """A uniform n-dimensional grid object with body-centered
        and face-centered veiws. 
        
        Parameters
        ----------
    
        dim_body: iterable of ints, the shape of the grid (number of cells
            along each dimension).

        min_body: iterable of floats, the body-centered minimum
            coordinate along each dimension.
        
        max_body: iterable of floats, the body-centered maximum
            coordinate along each dimension.

        """

        self.dim_body = np.atleast_1d(dim)
        self.min_body = np.atleast_1d(min_body)
        self.max_body = np.atleast_1d(max_body)

        # Simple checks on input.
        if (self.dim < 1).any():
            raise ValueError('no element of grid dim vector can be ' \
                             'less than 1.' % self.dim)
        if (min_body >= max_body).any():
            raise ValueError('no element of min_body can be greater than or '\
                             'equal to the corresponding element of max_body.')

        # Compute cell size / body-centered spacing.
        self.dx = (self.max_body - self.min_body) / (self.dim - 1)

        # Create meshes.
        # Body-centered. 
        self.extents_body = [np.linspace(*args) for args in 
                             zip(self.min_body, 
                                 self.max_body, 
                                 self.dim_body)]
        

        self.body = np.stack(np.meshgrid(*extents_body, indexing='ij', 
                                         sparse=True), axis=-1).view(Mesh)
        
        # Face-centered. 
        self.min_face = self.min_body - self.dx / 2
        self.max_face = self.max_body + self.dx / 2
        self.dim_face = self.dim_body + 1
        
        self.extents_face = [np.linspace(*args) for args in 
                             zip(self.min_face, 
                                 self.max_face, 
                                 self.dim_face)]
        
        self.face = np.stack(np.meshgrid(*extents_face, indexing='ij', 
                                         sparse=True), axis=-1).view(Mesh)
        

    def locate(self, data):
        """Return a dictionary 
        
        
