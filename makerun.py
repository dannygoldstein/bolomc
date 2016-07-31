
# Create the run files for a grid of LC fits

import os
import glob

try:
    os.mkdir('run')
except:
    pass

files = map(os.path.abspath, glob.glob('data/CSP_Photometry_DR2/*'))
yaml_base = open('default.yml').read()
sub = open('edison.sh').read()

for file in files:
    name = file.split('/')[-1].split('opt')[0]
    yaml = yaml_base.replace('_LCF_', file)
    yaml = yaml.replace('_OFN_', os.path.abspath('run/' + name + '.out'))
    yname = os.path.abspath('run/%s.yml' % name)
    
    with open(yname, 'w') as f:
        f.write(yaml)
    with open('run/%s.sh' % name, 'w') as f:
        f.write(sub.replace('xx.yml', yname).replace('fit.py', os.path.abspath('fit.py')))
        
        
        
    
