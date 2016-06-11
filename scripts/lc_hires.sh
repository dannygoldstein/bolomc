#!/bin/bash

ipython representative.py ../run/SN2005el
ipython representative.py ../run/SN2006os
ipython representative.py ../run/SN2005eq

ipython psurf.py ../run/SN2005el
ipython psurf.py ../run/SN2005eq
ipython psurf.py ../run/SN2006os