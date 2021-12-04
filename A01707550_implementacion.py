#!/usr/bin/env python
# coding: utf-8

# # F1009: Situación Problema: ¿cuál es el nombre de esa canción?
# Alberto Ruiz Biestro - A017075504 - Dec 2021

# In[1]:


import numpy as np                                            # 
import matplotlib.pyplot as plt                               # plots
import librosa                                                # lectura
import warnings                                               # para evitar un warning tedioso
import IPython.display as ipy
warnings.filterwarnings('ignore')                             # siempre me da warning

from scipy.fft import fft

import os
from os.path import isfile, join

plt.style.use('./tgtermes.mplstyle')

plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 140


# In[2]:


# función para leer cosas
def read_this(string, srate):
    signal, samplerate = librosa.load(string, sr = srate, mono = True)
    return(signal, samplerate)

# función que me regresa la transformada de Fourier iterada
def fourier(signal, samplerate, divisor):

    T = 1 / samplerate                                         # periodo de muestreo
    interval = int(samplerate / divisor)                       # intervalo en frecuencias
    interval_sec = interval / samplerate                       # intervalo en segundos
    
    magnitude = np.zeros((0,interval//2))                      # inicializamos el arreglo de la transformada
    position = 0                                               # inicializamos la posición
    size = int(len(signal) / interval)                         # el tamaño a recorrer
    
    for ii in range(size):                                     # recorremos toda la señal por intervalos
        mini_sig = signal[position:position + interval] 
        N = len(mini_sig)
        yf = abs(fft(mini_sig))[:interval//2]                  # calculamos FFT (scipy) para el fragmento de la señal
        yf = yf.reshape(1, interval//2)                        # también reducimos a la mitad todo dado que la transformada es simétrica
        magnitude = np.concatenate((magnitude , yf), axis = 0) # guardamos sólo la magnitud
        position += interval                                   # magnitude.shape debe dar len(signal_2) / 2

    mag_db = np.log(magnitude)                                 # magnitud en dB, al parecer no hace falta multiplicar por 20
                                                               # si hace falta añadir 'where=(magnitude != 0)' por definición de ln  
    f = np.linspace(0, samplerate / 2, int(samplerate / 2))    # ajuste horrible para poder graficar frecuency axis
    
    return(mag_db, None, magnitude, None, interval)

def mypeaks(matrix):

    m = matrix# np.array([mag_MAIN[90], mag_MAIN[91], mag_MAIN[92], mag_MAIN[93]])
    x = m[:,:int(2000 / div)] # hasta 2000 Hz
    fvec = np.arange(0, len(x[0]) * div, div)
    a_n = 4 # número de pedazos
    b_n = int(2000 / a_n / div) # 
    num = len(matrix)
    # 4 rangos de frecuencias
    peak_freq = np.zeros((num,4))
    
    k = 0
    for ii in range(num):    
        for jj in range(a_n):
            vector = x[ii, k : b_n + k]
            peak = max(vector)
            index = np.argmax(vector)
            peak_freq[ii][jj] = (index + k) * div
            k += b_n
        k = 0
    return(peak_freq)

# obtener un score para qué tanto coinciden
def getscore(song_magnitude, sample_magnitude):
    
    assert len(sample_magnitude)<len(song_magnitude), 'Array too big!' 
    score = []
    song_peaks = mypeaks(song_magnitude)
    sample_peaks = mypeaks(sample_magnitude)
    for ii in range(len(song_peaks) - len(sample_peaks)):
        score.append(sum(sum(sample_peaks == song_peaks[ii: len(sample_peaks) + ii])))     
    score_max = max(score)
    return(score_max)

def findmatch(goal):
    samplerate = 44100
    div = np.load('./music_data/database_peaks/div.npy')
    mag = np.load('./music_data/database_peaks/mag.npy', allow_pickle=True)
    pairs = np.load('./music_data/database_peaks/pairs.npy')
    
    songid = pairs[:,0]          
    sample, _ = read_this('./music_data/recordings_audio/' + goal, 44100)
    _, _, samplemag, _, _ = fourier(sample, samplerate, divisor = div)
    
    size = len(mag)

    samplescores = list(np.zeros(size))

    # se le asigna puntaje a cada canción
    for ii in range(size):
        samplescores[ii] = getscore(mag[ii], samplemag)

    sampleid = int(np.argwhere(samplescores == np.amax(samplescores)))

    runnerup = np.sort(samplescores)[::-1][1]

    winner = pairs[sampleid][1]

    print(f'Song is: {winner} with a {samplescores[sampleid] - runnerup} point margin over the runner up')
    
    return(winner, samplescores, songid)


# # Creación de Base de Datos
# 
# Se necesitan tres fólders o directorios: uno para las bases de datos del audio de las canciones (database_audio/), uno para lasbases de datos de los puntos altos o peaks (database_peaks/), y uno para donde se encuentran las grabaciones (recordings_audio/). 
# 
# Por orden alfabético se seleccionan las canciones dentro de la base de datos, y se les asigna un índice único:
# 
# Lo invito a cambiar los directorios directorio por unos suyos con bases de datos de sus canciones (tenga en cuenta que, como es un lenguaje interpretado, al final sí es pesado en la memoria). También lo invito a ver el archivo .html, para que vea qué tan bien puede identificar un fragmento pequeño de una canción con mucho ruido.

# In[3]:


# leemos los nombres de las canciones
songnames = [f for f in os.listdir('./music_data/database_audio/') if              isfile(join("./music_data/database_audio/", f))]
# creamos los IDs
songid = np.arange(0,len(songnames), 1)

# Creamos una matriz de forma [song ID, song Name]
pairs = np.transpose(np.vstack((songid, songnames)))

print(pairs)


# Ahora se leen todas las canciones de la base de datos, y se guardan dentro de un vector `songdata`. Este proceso es un poco tardado y pesado, dependiendo de la longitud de las canciones. En un futuro se podría prealocar espacio para cada canción al multiplicar su tiempo total por la frecuencia de muestreo, ya que esto contribuye a que sea un proceso lento.

# In[4]:


# Esto tarda un tiempo

songdata = list(np.zeros(len(songid)))

# el samplerate es el mismo para todos
samplerate = 44100

for ii in range(len(songid)): #len(pairs[:,0])
    string = ('./music_data/database_audio/' + pairs[ii,1])
    songdata[ii], _ = read_this(string, samplerate)


# Se calculan las magnitudes de los espectros con la Transformada de Fourier, y se guardan en otro vector `mag`. Esto también se tarda un par de segundos. Subsecuentemente se guardan estos datos en el folder correspondiente.

# In[5]:


div = 5 # divisor del samplerate

mag = t = list(np.zeros(len(songid)))

for ii in range(len(songid)):
    _,_ , mag[ii] , _, interval = fourier(songdata[ii], samplerate, divisor = div)
    


# In[6]:


np.save('./music_data/database_peaks/mag.npy', mag)
np.save('./music_data/database_peaks/pairs.npy', pairs)
np.save('./music_data/database_peaks/div.npy', div)


# Es importante guardar todo lo anterior (de nuevo lo invito a utlizar sus propios directorios) en un mismo directorio. El proceso que sigue es más eficiente y limpio si es hecho dentro de un archivo distinto, pero por cuestiones de presentación, se adjunta en este mismo código, al igual que con las funciones definidas al inicio.

# # Proceso de clasificar la grabación
# ## Canción 1
# 
# Obtenemos nuestra grabación y queremos compararla contra la base de datos. Como lo explico en el documento, esto se puede lograr a pesar del sonido y la calidad del micrófono, siempre y cuando la frecuencia de muestreo sea la misma que con la que se grabó. La mayoría de los teléfonos graban con una frecuencia de meustreo de 44.1 kHz.

# In[7]:


# canción de muestra
recording = 'steve_reich_electric_counterpart_3_rec.wav'

# escuchar la grabación
ipy.Audio('./music_data/recordings_audio/' + recording, rate=44100)


# Se llama a la función `findmatch` y se logra encontrar una canción dentro de la base de datos que, de acuerdo al programa, es la "ganadora" en cuanto a similitud con la grabación.

# In[8]:


win, myscores, myid = findmatch(recording)

# plot
plt.bar(myid, myscores)
plt.xlabel('Song ID')
plt.title('Puntajes para "' + recording + '"')
plt.ylabel('Score')
plt.show()


# In[9]:


# escuchar canción que encontró ser similar

ipy.Audio('./music_data/database_audio/'  + win, rate=44100)


# ## Canción 2
# 
# Ahora se intenta con otra canción

# In[13]:


# canción de muestra
recording = 'boards_of_canada_roygbiv_rec.wav'

ipy.Audio('./music_data/recordings_audio/' + recording, rate=44100)


# In[14]:


win, myscores, myid = findmatch(recording)

# plot
plt.bar(myid, myscores)
plt.xlabel('Song ID')
plt.title('Puntajes para "' + recording + '"')
plt.ylabel('Score')
plt.show()


# In[15]:


ipy.Audio('./music_data/database_audio/'  + win, rate=44100)


# ## Canción 3

# In[16]:


# canción de muestra
recording = 'mac_de_marco_another_one_rec.wav'

ipy.Audio('./music_data/recordings_audio/' + recording, rate=44100)


# In[17]:


win, myscores, myid = findmatch(recording)

# plot
plt.bar(myid, myscores)
plt.xlabel('Song ID')
plt.title('Puntajes para "' + recording + '"')
plt.ylabel('Score')
plt.show()


# Vemos que, aunque nos haya clasificado la canción adecuadamente, el margen de victoria no es tan alto esta vez como con grabaciones más largas.

# In[18]:


ipy.Audio('./music_data/database_audio/' + win, rate=44100)

