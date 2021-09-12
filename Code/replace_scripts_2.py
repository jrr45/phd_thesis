# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:01:20 2021

@author: ffdra
"""

import glob
import re

filenames = glob.glob("*.svg")
#filenames = ["JR200115_11_300K_IDvVDS-loop_VG-decreasing_log_txt.svg"]

re_origional = re.compile(r"""V\^\{1\/2\}""" 
                          )

newtext = """V<tspan
   style="font-size:65%;baseline-shift:super"
   id="tspan15378">1/2</tspan>"""
                       
for filename in filenames:
    print(filename)
    with open(filename, 'r') as f:
      text = f.read()

    text = re_origional.sub(newtext, text)

    
    with open(filename, 'w') as f:
        f.write(text)