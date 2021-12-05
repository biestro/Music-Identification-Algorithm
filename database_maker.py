'''
| F1009: Situación Problema: ¿cuál es el nombre de esa canción?
| Alberto Ruiz Biestro - A017075504 - Dec 2021
| hacedor machín de databases
| no tan machín porque es python 

'''
import numpy as np
from identification import *
import os
from os.path import isfile, join

'''
| variables
'''
div = 5 # divisor del samplerate

# se puede aprovechar el downsampling
# no he comprobado resultados cuantitativamente
samplerate = 8820


# lectura de todos los song names
songnames = [f for f in os.listdir("./database_audio/") if \
    isfile(join('./database_audio/', f))]

songid = np.arange(0,len(songnames), 1)

# Creamos una matriz de forma [song ID, song Name]
pairs = np.transpose(np.vstack((songid, songnames)))

# imprimir las canciones2
print(pairs)

# proceso lento
songdata = list(np.zeros(len(songid), dtype=object))

for ii in range(len(songid)): #len(pairs[:,0])
    string = ('./database_audio/' + pairs[ii,1])
    songdata[ii], _ = readthis(string, samplerate)

mag = t = list(np.zeros(len(songid), dtype=object))

for ii in range(len(songid)):
    _,_ , mag[ii] , _, interval = fourier(songdata[ii], samplerate, divisor = div)
    
np.save('./database_peaks/mag.npy', mag)
np.save('./database_peaks/pairs.npy', pairs)
np.save('./database_peaks/div.npy', div)
np.save('./database_peaks/samplerate.npy', samplerate)