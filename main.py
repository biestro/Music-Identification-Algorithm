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

    div = load('./database_peaks/div.npy')
    sr = load('./database_peaks/samplerate.npy')
    print('============================================')
    print('=== A01707550 --  SP -- Alberto Ruiz B. ===')
    
       
    recording = 'recording_' + input('In[]: ') + '.wav'
    print('============================================')
    
    print('============================================')
    print('\nPlaying Recording (Press "q" to quit)\n')
    print('============================================')
    subprocess.run(["mpv", "./recordings_audio/" + recording])
    
    win, myscores, myid = findmatch(recording, div, srate=sr) 

    # plot
    plt.bar(myid, myscores)
    plt.xlabel('Song ID')
    plt.title('Puntajes para "' + recording + '"')
    plt.ylabel('Score')
    plt.show()

    print('============================================')
    print('\nPlaying Matched Song (Press "q" to quit)\n')
    print('============================================')
    subprocess.run(["mpv", "./database_audio/" + win, '--start=30%'])
    time.sleep(1)
    os.system('cls||clear')
    
