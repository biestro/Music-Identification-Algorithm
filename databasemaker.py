import numpy as np
import os
from os.path import isfile, join
from identification import *
from chooser import chooseFolder 

# leemos los nombres de las canciones
try: 
    songnames = [f for f in os.listdir('./music_data/database_audio/') if \
           isfile(join("./music_data/database_audio/", f))]
except:
    print()
    print("Could not locate './music_data/database_audio/'")
    print()
    print("Creating directory './music_data/database_audio/")
    os.mkdir("./music_data/")
    os.mkdir("./music_data/database_audio/")
    chooseFolder('./music_data/database_audio/')
    songnames = [f for f in os.listdir('./music_data/database_audio/') if \
           isfile(join("./music_data/database_audio/", f))]
# creamos los IDs
songid = np.arange(0,len(songnames), 1)

# Creamos una matriz de forma [song ID, song Name]
pairs = np.transpose(np.vstack((songid, songnames)))

# check if list is empty (aka if there are no songs)
print(pairs)
# leer canciones
songdata = list(np.zeros(len(songid)))

# el samplerate es el mismo para todos
# nota: revisar si sirve menor samplerate
samplerate = 44100

print()
print("Reading songs...")
for ii in range(len(songid)): #len(pairs[:,0])
    string = ('./music_data/database_audio/' + pairs[ii,1])
    songdata[ii], _ = read_this(string, samplerate)

# transformada y peaks, checar si mejor guardamos los peaks
div = 5 # divisor del samplerate
mag = list(np.zeros(len(songid)))
for ii in range(len(songid)):
    _,_ , mag[ii] , _, interval = fourier(songdata[ii], samplerate, divisor = div)
    
try:
    np.save('./music_data/database_peaks/mag.npy', mag)
except:
    print()
    print("Could not find directory './music_data/database_peaks'")
    print()
    print("Creating directory './music_data/database_peaks'")
    os.mkdir("./music_data/database_peaks/")
finally:
    np.save('./music_data/database_peaks/mag.npy', mag)
    np.save('./music_data/database_peaks/pairs.npy', pairs)
    np.save('./music_data/database_peaks/div.npy', div)
    np.save('./music_data/database_peaks/SAMPLERATE.npy', samplerate)

print()
print('\nDone!\n')
