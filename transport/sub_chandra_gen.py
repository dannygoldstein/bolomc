from scipy.spatial import distance
from escape import dgen, points
import scipy
import pickle

lower = scipy.array([0.2, 0.1, 0.0, 0.0])
upper = scipy.array([1.0, 0.9, 0.3, 1.38])
point_type = points.IronSedonaPoint
score_func = lambda x: distance.pdist(x.scaled).min()

ga = dgen.dgen_ga(point_type, lower, upper, score_func, 
                  200, generations=None, verbose=True,
                  save_each_iter_best=True, 
                  timelimit=4*3600)

pickle.dump(ga, open('sub_chandra.grid.pkl','wb'))
