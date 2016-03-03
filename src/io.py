
import os

join = os.path.join
burns14data_filename = join(__file__, '../data/burns2014.tab')

def _get_burns(name):
    if name.startswith('SN'):
        name = name[2:]
    with open(burns14data_filename, 'r') as f:
        for line in f:
            if line.startswith(name):
                return line.split()
    raise KeyError(name)
