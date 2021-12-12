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
    sr = load('./music_data/database_peaks/SAMPLERATE.npy')
    dbase = load('./music_data/database_peaks/pairs.npy')
    
    print(
"""
,___________________________________________
|\`-._(   /                                 |
| \  .'-._\      Alberto R B           ,   ,|
|-.\`    .-;                         .'\`-' |
|   \  .' (       F1009 SP        _.`   \   |
|.--.\`   _)                    ;-;      \._|
|    ` _\(_)/_                  \ `'-,_,-'\ |
|______ /(O)\  _________________/____)_`-._\|
"""
        )

    print('\nEnter "R" to record:\n') 
    print('\nEnter "Q" to quit:\n') 
    print('\nEnter "L" to list available songs:\n') 
    recording = input('Input: ')
    if recording.lower() == 'q':
    	break
    elif recording.lower() == 'r':
    	myrecording()
    elif recording.lower() == 'l':
        print(dbase)
        time.sleep(5)
    else:
    	print('\nfine, have it your way... rebel\n')
    	myrecording()
    
    '''
    print('==============================================')
    print('\nPlaying Recording (Press "q" to quit)\n')
    print('==============================================')

    subprocess.run(['mpv', './recording.wav', '--volume=60'])
    '''
    win, myscores, myid = findmatch('recording.wav', srate=sr) 
    if myid[0] == False: 
        print('\nReturning to main')
        time.sleep(2)
        os.system('cls||clear')
    else:

        print('==============================================')
        print(f'\nPlaying Matched Song: \n\t{win}\n (Press "q" to quit)\n')
        print('==============================================')

        subprocess.run(["mpv", "./music_data/database_audio/" + win, '--start=60%', '--geometry=20%'])
        time.sleep(1)
        os.system('cls||clear')

    
