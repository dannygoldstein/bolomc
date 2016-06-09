#!/usr/bin/env python 

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Scrape B-band t0 fit values from CSP website.'

import bs4
import glob
import time
import pandas
import requests

# Make a list of supernovae we have data for. 
sn_names = map(lambda x: x.split('opt')[0].split('/')[-1],
               glob.glob('data/CSP*/*.dat'))

# Send requests to this base URL. 
baseurl = 'http://csp.obs.carnegiescience.edu/data/lowzSNe/'

# Store results here. 
result = []; nodata = []

# For each supernova, 
for i, sn in enumerate(sn_names):
    
    # Get the HTML from the CSP website. 
    
    if i == 0:
        tstrt = time.asctime(time.gmtime())
    resp = requests.get(baseurl + sn)
    if not resp.ok:
        raise Exception(sn)

    # Create the html parser. 
    soup = bs4.BeautifulSoup(resp.content, "lxml")
    
    # Each "LC Fit" table has a caption tag. 
    listings = soup.findAll('table', class_="listing")

    # If there are no listings, then the assumption above is wrong --
    # either something about the site's HTML has changed, or the data
    # we are seeking is not present (unreleased).
    if len(listings) == 0:
        
        # If there is no data, make a note in the file.
        nodata.append(sn)
        
        # And print just for quick reference.
        print sn

    # The following loop finds the galaxy type and writes it to a
    # file.
    for listing in listings:
        for row in listing.findAll("tr"):
            data = row.findAll("td")
            if len(data) > 0 and 'Host type' in data[0].text:
                result.append((sn, data[1].text)) # tmax_B

df = pandas.DataFrame(result, columns=['name', 'host_type'])

with open('data/gal.dat', 'w') as f:
    f.write('# this file was generated between %s and %s UTC \n' % (
        tstrt, time.asctime(time.gmtime())))
    f.write('# based on the CSP website %s during that time.\n' % baseurl)
    f.write('# the t0 values are mjd dates of B-band maximum \n')
    f.write('# taken from SNooPy dm15 fits.\n')
    if len(nodata) > 0:
        f.write('# data was unsuccessfully requested for the following sne:\n')
        for sn in nodata:
            f.write('# %s\n' % sn)
    df.sort('name', inplace=True)
    df.to_csv(f, index=False, sep=' ')
