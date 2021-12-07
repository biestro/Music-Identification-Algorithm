'''
| F1009: Situación Problema: ¿cuál es el nombre de esa canción?
| Alberto Ruiz Biestro - A017075504 - Dec 2021
| archivo principal que reconoce las canciones
| no copiar eh ಠ_ಠ
'''

import subprocess
import time
import matplotlib.pyplot as plt
from identification import *
import os
from numpy import load

while True:

    div = load('./music_data/database_peaks/div.npy')
    sr = load('./music_data/database_peaks/samplerate.npy')
    
    print(
"""
,___________________________________________
|\`-._(   /                                 |
| \  .'-._\      Alberto R B           ,   ,|
|-.\`    .-;                         .'\`-' |
|   \  .' (       F1009 SP        _.`   \   |
|.--.\`   _)                    ;-;      \._|
|    ` _\(_)/_                  \ `'-,_,-'\ |
jgs____ /(O)\  _________________/____)_`-._\|
"""
        )

    print('\nEnter recording number:\n') 
    recording = 'recording_' + input('In[]: ') + '.wav'
    
    print('==============================================')
    print('\nPlaying Recording (Press "q" to quit)\n')
    print('==============================================')

    subprocess.run(["mpv", "./music_data/recordings_audio/" + recording])
    win, myscores, myid = findmatch(recording, srate=sr) 

    if myid == False:
        print('\nReturning to main')
        time.sleep(2)
        os.system('cls||clear')
    else:

        print('==============================================')
        print('\nPlaying Matched Song (Press "q" to quit)\n')
        print('==============================================')

        subprocess.run(["mpv", "./music_data/database_audio/" + win, '--start=30%'])
        time.sleep(1)
        os.system('cls||clear')

    
