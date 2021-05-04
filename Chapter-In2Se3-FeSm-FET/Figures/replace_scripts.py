# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:01:20 2021

@author: ffdra
"""

import glob
import re

filenames = glob.glob("*.svg")
#filenames = ["JR200115_11_300K_IDvVDS-loop_VG-decreasing_log_txt.svg"]

re_origional = re.compile(r"""10</tspan><tspan\s*""" +
                          r""".*font-size:7px.*"\s*""" +
                          r"""x=".*"\s*""" 
                          r"""y=".*\"""" 
                          , re.MULTILINE)

newtext = """10<tspan
                     style="font-size:65%;baseline-shift:super\""""
                       
for filename in filenames:
    print(filename)
    with open(filename, 'r') as f:
      text = f.read()

    text = re_origional.sub(newtext, text)
    
                       
    for i in range(0,13):   
        text = text.replace(str(i) + '</tspan></text>',
                            str(i) +'</tspan></tspan></text>')

    
    with open(filename, 'w') as f:
        f.write(text)