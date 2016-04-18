
import os
import glob

exe = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), 
                   'bolomc/model.py')

runfile = """
#!/bin/bash
%s %s 10 20 %s %s 

"""

try:
    os.mkdir('run')
except:
    pass

lcfiles = glob.glob('data/*/*')
for fn in lcfiles:
    runf_basename = fn.split('/')[-1].split('opt')[0]
    fname = 'run/%s.run' % runf_basename
    outname = '%s.h5' % runf_basename
    logname = '%s.log' % runf_basename
    ffn = os.path.abspath(fn)
    with open(fname, 'w') as f:
        f.write(runfile % (exe, ffn, outname, logname))
    
        
    
