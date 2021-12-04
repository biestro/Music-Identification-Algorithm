#!/usr/bin/env python
# coding: utf-8

# # F1009: Reto Final Final ahora sí el bueno
# ### A01707550
# ### Alberto Ruiz Biestro
# 
# Este documento se utilizó para demostrar mi acercamiento y proceso. El algoritmo como tal ya implementado está en el programa "shazam". De todos modos sugiero revisar este, pues es una buena explicación de cómo se llegó al resultado.

# # Importar paquetes

# In[1]:


import numpy as np                                            # 
import matplotlib.pyplot as plt                               # plots
import librosa                                                # lectura
import warnings                                               # para evitar un warning tedioso
import IPython.display as ipy
warnings.filterwarnings('ignore')                             # siempre me da warning

from scipy.signal import find_peaks, decimate, resample, correlate
from scipy.fft import fft
from scipy.stats import mode

#plt.style.use('default')
# para exportarlo con formato chido:
plt.style.use('./tgtermes.mplstyle')

plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 140


# # Funciones

# In[2]:


# función para leer cosas
def read_this(string, srate):
    signal, samplerate = librosa.load(string, sr = srate, mono = True)
    return(signal, samplerate)

# función que me regresa la transformada de Fourier iterada
def fourier(signal, samplerate, divisor):

    T = 1 / samplerate                                         # periodo de muestreo
    t = np.arange(0, len(signal) / (samplerate), T)            # vector tiempo para el plot
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
    
    return(mag_db, t, magnitude, f, interval)

# función que regresa una canción con sonido y menor calidad
def noisify(string, srate):
    noisefact = 0.01 # factor de la amplitud del sonido (0.01 es mucho)
    song, samplerate = read_this(string, srate)
    noise = np.random.normal(0, noisefact, song.shape)
    return(song + noise, samplerate)
    
# función de espectrograma
def spectrogram(matrix, time, freq, title, bartitle):
    colormap = plt.cm.magma
    fig, ax = plt.subplots()
    fig.set_size_inches(5,4)
    cax = ax.imshow(np.transpose(matrix),
               origin = 'lower', 
               aspect = 'auto',
               cmap = colormap,
               # el extent sólo modifica los ticks, pero sí afecta si se usa log scale
               extent = [0, time[len(time) - 1], 0, freq[len(freq) - 1]]) 
    
    #ax.set_yscale('symlog', linthresh = 200) # para comportamiento similar a la mel scale
    ax.set_xlabel('Tiempo /s')
    ax.set_ylabel('Frecuencia /Hz')
    
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel(bartitle, rotation=270, labelpad = 10)
    ax.set_title(title)
    ax.set_ylim([0,1000]) # frecuencias hasta el 1000 Hz
    ax.grid(0)

# función que grafica los puntos claves del espectro
def constellation(matrix, time, freq, interval, pmn, title):           
    cnstel = []                                                 # inicializamos el arreglo de las constelaciones
    fig, ax = plt.subplots(1)
    
    for ii in range(len(matrix)):                               # a lo mejor con simplemente extraer una feature única de los puntos?
        vec = matrix[ii]                                        # ver el código de abajo de los peaks para visualizar mejor
        peaks, _ = find_peaks(vec,  prominence = pmn)           # seguramente hay una forma más rápida y menos confusa de hacerlo
        cnstel.append(np.transpose(peaks * 44100/ interval))                      # pero mi yo del pasado es igual de complicado que yo
        xlow = int(interval * time[ii]) * np.ones(len(cnstel[ii]))                   # a veces no entiendo lo que hago
        ax.plot(xlow, cnstel[ii], 'xk', markersize = 5, alpha = 0.5)
    
    ax.set_xlabel('Tiempo /s')
    ax.set_ylabel('Frecuencia /Hz')
    ax.set_ylim([0,1000])
    fig.set_size_inches(4,4)
    plt.title(title)
    ax.grid()

# función para el hash que no sirve
def hashfun(x, plot):
    htable = []
    for ii in range(len(x)):
        hash_ = 11
        peaks, _ = find_peaks(x[ii],  prominence = 4)
        for jj in range(len(peaks)):
            hash_ = (hash_  + 13 * peaks[jj])  # versión vieja de mi hash
        htable.append(hash_)
    if plot == True:
        plt.plot(htable)
        plt.xlabel('Índice')
        plt.ylabel('Hash')
        plt.title('Hash Table')
        plt.show()
    
    return(htable)

# obtener sólo 4 peaks
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
    
    print('Match :)') if (score_max > 70) else print('No match :(')

    return(score_max)


# # Canciones y Muestras
# ### Lectura
# Se emplearon 4 señales:
# 1. Una canción base que nos servirá como ejemplo, `song_MAIN`
# 2. Un fragmento grabado desde mi celular, `song_CUT`
# 3. El mismo fragmento pero con sonido blanco y menor calidad, `song_NOISE`
# 4. Una canción completamente diferente `song_WRONG`
# 
# Cabe mencionar que para el análisis de puntajes es de suma importancia que las grabaciones a comparar tengan una duración similar.
# 
# La frecuencia de muestreo del micrófono de mi teléfono es de 44.1kHz, mientras que su bit-rate es de 128kbps

# In[6]:


song, sr = read_this('./music/boards_of_canada_roygbiv.wav', 44100) 
song_CUT, sr_CUT = read_this('./music/boards_of_canada_roygbiv_rec_2.wav', 44100)
song_NOISE, sr_NOISE = noisify('./music/boards_of_canada_roygbiv_cut.wav', srate=8000)

song_WRONG, sr_WRONG = read_this('./music/mac_demarco_another_one_cut.wav', srate=44100)


# In[7]:


# es de suma importancia que el samplerate sea el correspondiente
ipy.Audio(song, rate=sr)


# In[8]:


ipy.Audio(song_CUT, rate=sr_CUT)


# In[9]:


ipy.Audio(song_NOISE, rate=sr_NOISE)


# In[10]:


ipy.Audio(song_WRONG, rate=sr_WRONG)


# ## Aplicación de la Transformada

# In[11]:


div = 5 # divisor del samplerate

# obtenemos magnitud en dB, vector de tiempo (solo para plot),
# magnitud pero no en dB, vector de frecuencias (solo para plot),
# intervalo de tiempo en el que se aplicó la transformada

mag_MAIN, t_MAIN, mag_MAIN_no_dB, f_MAIN, interval_MAIN = fourier(song, sr, divisor = div)


# In[12]:


mag_CUT, t_CUT, mag_CUT_no_DB, f_CUT, interval_CUT = fourier(song_CUT, sr_CUT, divisor = div)


# In[13]:


mag_NOISE, t_NOISE, mag_NOISE_no_dB, f_NOISE, interval = fourier(song_NOISE, sr_NOISE, divisor = div)


# In[14]:


mag_WRONG, t_W, mag_WRONG_no_dB, f_W, interval_W = fourier(song_WRONG, sr_WRONG, divisor = div)


# ## Espectrogramas

# In[15]:


spectrogram(mag_MAIN, t_MAIN, f_MAIN, title = 'Plot de magnitud (en dB)', bartitle = 'Intensidad /dB')


# In[16]:


spectrogram(mag_CUT_no_DB, t_CUT, f_CUT, title = 'Plot de magnitud (fragmento)', bartitle = 'Intensidad /a.u.')


# In[17]:


spectrogram(mag_NOISE_no_dB, t_NOISE, f_NOISE, title = 'Plot de magnitud (fragmento con sonido)', bartitle = 'Intensidad /a.u.')


# In[18]:


spectrogram(mag_WRONG_no_dB, t_W, f_W, title = 'Plot de magnitud (canción diferente)', bartitle = 'Intensidad /a.u.')


# Objetivo: Obtener coordenadas y comparar

# ## Primer iteración de puntos máximos
# El primer método que se usó fue con la función de '`scipy.signal.find_peaks()`. Los resultados para una sección del espectro está abajo:

# In[19]:


# Visualización de los puntos altos
x = mag_MAIN[90]
peaks_p, _ = find_peaks(x,  prominence = 4)
peaks_w, _ = find_peaks(x, height = 4.5)

fvec = np.arange(0, len(x) * div, div)


# In[20]:


fig, (ax1, ax2) = plt.subplots(2, sharex = True)
ax1.plot(fvec, x, linewidth = 0.3); ax1.plot(div * peaks_p, x[peaks_p], 'or', markersize = 3); 
ax1.axvline(x= 5000, color = 'b', linestyle = '--', linewidth = 1); ax1.legend(['_nolegend_','prominence'])
ax2.plot(fvec, x, linewidth = 0.3); ax2.plot(div * peaks_w, x[peaks_w], 'ob', markersize = 3); 
ax2.axvline(x= 5000, color = 'b', linestyle = '--', linewidth = 1); ax2.legend(['_nolegend_','width'])
plt.xlabel('Frecuencia /Hz')
plt.ylabel('Magnitud /dB')
fig.set_size_inches(5,2.5)
plt.subplots_adjust(hspace=0.05)
plt.show()


# En el artículo de Avery Wang de Shazam se incluye un *mapa de constelaciones*, el cual representa lo mismo que las gráficas anteriores pero a lo largo de todo el espectro

# In[21]:


constellation(mag_MAIN, t_MAIN, f_MAIN, interval_MAIN, pmn=5, title = 'Mapa de constelaciones para toda la canción')


# ### Método de Hashing (inútil)
# **Canción normal y con sonido**
# 
# Primero se intentó aplicar una función de hashing a la canción original y a la misma canción con sonido, para ver qué tan diferentes eran los puntos altos. Cabe mencionar que se utilizó la canción entera para ambos casos, y no los fragmentos (eso viene después).

# In[22]:


song_NOISE_FULL, sr_NOISE_FULL = noisify('./music/boards_of_canada_roygbiv.wav', srate=8000)
_, _, mag_NOISE_FULL, _, interval = fourier(song_NOISE_FULL, sr_NOISE_FULL, divisor = div)
# Normalización para poder restar los hashes
hashtable = hashfun(mag_MAIN_no_dB[:,: int(5000 / div)], plot=False)
np.save('hashtable.npy', hashtable)
hashtable_NOISE = hashfun(mag_NOISE_FULL[:,: int(5000 / div)], plot=False)
norm_ = hashtable / max(hashtable)
norm_NOISE = hashtable_NOISE / max(hashtable_NOISE)
difference = np.subtract(norm_, norm_NOISE)
# obtenemos el mean square error
mse = (difference ** 2).mean()


# In[23]:


fig, ax = plt.subplots(1)
ax.plot(norm_)
ax.plot(norm_NOISE)
ax.plot(difference)
ax.legend(['Hash', 'Hash$_{NOISE}$', '$H - H_{noise}$'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Song vs. Song_Noise')
plt.show()

print('MSE = %.5f' %mse)
print('Signals may match') if mse < 0.1 else print('Signals may not match')


# **Canciones diferentes**

# In[24]:


A, sr = read_this('./music/tool_eon_blue_apocalypse.wav', 44100)
B, _ = read_this('./music/boards_of_canada_tswn.wav', 44100)

_, t, Afft, f, interval = fourier(A, sr, divisor = 5)
_, t, Bfft, f, interval = fourier(B, sr, divisor = 5)

hashtable_A = hashfun(Afft[:,: int(5000 / div)], plot=False)
norm_A = hashtable_A / max(hashtable_A)
hashtable_B = hashfun(Bfft[:,: int(5000 / div)], plot=False)
norm_B = hashtable_B / max(hashtable_B)
difference = np.subtract(norm_A, norm_B[:len(hashtable_A)])
# hacemos el :len para que puedan ser restados

mse = (difference ** 2).mean()


# In[25]:


fig, ax = plt.subplots(1)
ax.plot(norm_A)
ax.plot(norm_B)
ax.plot(difference)
plt.legend(['Hash$_A$', 'Hash$_B$', '$H - H_{noise}$'], loc='center left', bbox_to_anchor=(1, 0.5))
ax.text(415, -0.35, '$MSE$ = %.4f' %mse)
plt.title('Canción vs canción diferente')
plt.show()

print('Signals may match') if mse < 0.1 else print('Signals may not match')


# Este es un método muy limitado (sólo se puede comprobar entre canciones enteras, es un método inútil más bien), pero pudimos comprobar y visualizar que el sonido no afecta tanto al valor de nuestros hashes, mientras que se mantiene la sensibilidad a la diferencia entre canciones que no coinciden. Esto es importante dado que el siguiente método no tiene nada visual.

# ## Método de Puntaje
# 
# Había intentado obtener los peaks con la función de `scipy`, pero como esta me regresaba un distinto número de puntos máximos, decidí mejor obtener sólo el punto máximo en `a_n` intervalos (en este caso 4, 'inclusive'). Abajo se representa una visualización de las áreas 

# In[26]:


m = np.array([mag_MAIN[90], mag_MAIN[91], mag_MAIN[92]])
x = m[:,:int(2000 / div)] # hasta 2000 Hz
fvec = np.arange(0, len(x[0]) * div, div)

a_n = 4 # número de pedazos
b_n = int(2000 / a_n / div)

peak_freq = np.zeros((3,a_n))


# In[27]:


fig, ax = plt.subplots(3, sharex = True)
k = 0
for ii in range(3):    
    ax[ii].plot(fvec, x[ii], linewidth = 0.3)
    for jj in range(a_n):
        vector = x[ii, k:b_n + k]
        peak = max(vector)
        index = np.argmax(vector)
        ax[ii].plot((index + k) * div, x[ii][index + k], 'or', markersize = 3)
        for kk in range(a_n + 1): #
            ax[ii].axvline(x= b_n * div * (kk), color = 'b', linestyle = ':', linewidth = 0.8)
        ax[ii].legend(['Index: %i'%(90 + ii)], loc='center left', bbox_to_anchor=(1, 0.5))
        peak_freq[ii][jj] = (index + k) * div
        k += b_n
    k = 0
ax[0].set_title('Visualización de puntos máximos en rangos de frecuencias')
plt.xlabel('Frecuencia /Hz')
ax[1].set_ylabel('Magnitud /dB')
plt.show()


# ![Cat](plan.jpg)

# In[28]:


mag_M_2 = mag_MAIN[:int(len(mag_MAIN)) -1]
print(f'SCORE canción principal vs canción principal (max score): {getscore(mag_MAIN,mag_M_2)}\n')
print(f'SCORE canción principal vs fragmento con sonido: {getscore(mag_MAIN, mag_NOISE)}\n')
print(f'SCORE canción principal vs fragmento: {getscore(mag_MAIN, mag_CUT)}\n')
print(f'SCORE canción principal vs canción diferente: {getscore(mag_MAIN, mag_WRONG)}\n')


# In[29]:


# Funciones, otra vez...
# obtener sólo 4 peaks
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
    print('Match :)') if (score_max > 70) else print('No match :(')
    return(score_max)


# Da un AssertionError aquí abajo (es intencional). Correr desde una cell abajo.

# In[30]:


# 'Match :)' if (getscore(mag_MAIN, mag_MAIN) > 70) else 'No match :('


# # Segunda Lectura

# In[31]:


song, sr = read_this('./music/boards_of_canada_in_the_annexe.wav', 44100) 
song_CUT, sr_CUT = read_this('./music/boards_of_canada_in_the_annexe_cut.wav', 44100)
song_NOISE, sr_NOISE = noisify('./music/boards_of_canada_in_the_annexe_cut.wav', srate=8000)
song_WRONG, sr_WRONG = read_this('./music/boards_of_canada_roygbiv_cut.wav', srate=44100)

mag_MAIN, t_MAIN, mag_MAIN_no_dB, f_MAIN, interval_MAIN = fourier(song, sr, divisor = div)
mag_CUT, t_CUT, mag_CUT_no_DB, f_CUT, interval_CUT = fourier(song_CUT, sr_CUT, divisor = div)
mag_NOISE, t_NOISE, mag_NOISE_no_dB, f_NOISE, interval = fourier(song_NOISE, sr_NOISE, divisor = div)
mag_WRONG, t_W, mag_WRONG_no_dB, f_W, interval_W = fourier(song_WRONG, sr_WRONG, divisor = div)


# In[32]:


ipy.Audio(song, rate=sr)


# In[33]:


ipy.Audio(song_CUT, rate=sr_CUT)


# In[34]:


ipy.Audio(song_NOISE, rate=sr_NOISE)


# In[35]:


ipy.Audio(song_WRONG, rate=sr_WRONG)


# **Quise incluír los espectros porque se ven como unos gusanos.**

# In[36]:


spectrogram(mag_MAIN, t_MAIN, f_MAIN, title = 'Plot de magnitud (en dB)', bartitle = 'Intensidad /dB')


# In[37]:


spectrogram(mag_CUT_no_DB, t_CUT, f_CUT, title = 'Plot de magnitud (fragmento)', bartitle = 'Intensidad /a.u.')


# In[38]:


spectrogram(mag_NOISE_no_dB, t_NOISE, f_NOISE, title = 'Plot de magnitud (fragmento con sonido)', bartitle = 'Intensidad /a.u.')


# In[39]:


spectrogram(mag_WRONG_no_dB, t_W, f_W, title = 'Plot de magnitud (canción diferente)', bartitle = 'Intensidad /a.u.')


# In[40]:


# resultados

mag_M_2 = mag_MAIN[:int(len(mag_MAIN)) -1]
print(f'SCORE (2) canción principal vs canción principal (max score): {getscore(mag_MAIN,mag_M_2)}\n')
print(f'SCORE (2) canción principal vs fragmento con sonido: {getscore(mag_MAIN, mag_NOISE)}\n')
print(f'SCORE (2) canción principal vs fragmento: {getscore(mag_MAIN, mag_CUT)}\n')
print(f'SCORE (2) canción principal vs canción diferente: {getscore(mag_MAIN, mag_WRONG)}\n')


# # Lectura con micrófono
# 
# Se incluyó una grabación con el micrófono de mi celular. La canción es un poco más larga, entonces el análisis para la canción base tarda más que con canciones anteriores. Para que no dure tanto se redujo la frecuencia de muestreo (downsampling). Mientras siempre se trate con las frecuencias de muestreo respectivas, no debería de haber problema alguno en el análisis.

# In[41]:


song, sr = read_this('./music/steve_reich_electric_counterpart_3.wav', 22050) 
song_NOISE, sr_NOISE = noisify('./music/steve_reich_electric_counterpart_3_rec.wav', 22050)
song_WRONG, sr_WRONG = read_this('./music/uziq_meinheld_cut.wav', srate=22050)


# In[42]:


mag_MAIN, t_MAIN, mag_MAIN_no_dB, f_MAIN, interval_MAIN = fourier(song, sr, divisor = div)
mag_NOISE, t_NOISE, mag_NOISE_no_dB, f_NOISE, interval = fourier(song_NOISE, sr_NOISE, divisor = div)
mag_WRONG, t_W, mag_WRONG_no_dB, f_W, interval_W = fourier(song_WRONG, sr_WRONG, divisor = div)


# In[43]:


ipy.Audio(song, rate=sr)


# In[44]:


ipy.Audio(song_NOISE, rate=sr_NOISE)


# In[45]:


ipy.Audio(song_WRONG, rate=sr_WRONG)


# In[46]:


mag_M_2 = mag_MAIN[:int(len(mag_MAIN)) -1]
print(f'SCORE (3) canción principal vs canción principal (max score): {getscore(mag_MAIN,mag_M_2)}\n')
print(f'SCORE (3) canción principal vs fragmento grabado: {getscore(mag_MAIN, mag_NOISE)}\n')
print(f'SCORE (3) canción principal vs canción diferente: {getscore(mag_MAIN, mag_WRONG)}\n')


# # Limitaciones
# 
# * Sólo pueden ser archivos .wav o .mp3 (o cualquier que use PCM).
# * Sólo se pueden reconocer canciones que estén dentro de la base de datos
# * La longitud de la muestra debe estar en un cierto rango de : `len(sample) < len(song)`.
#     * Este método funciona mejor con muestras de similar longitud para hacer una comparación digna (al final de cuentas el `score` otorgado no tiene unidades, ver el programa "shazam").
#     * Con técnicas como Combinatorial Hashing se pueden llegar a mejores resultados.
# * No tarda casi nada el programa, pero para un sistema de cientas o miles de canciones, se torna un proceso lento generar una base de datos.
#     * Sustituír ciclos `for` por funciones `map()`.
#     * Utilizar un lenguaje compilado para la creación de base de datos (Ver el programa "shazam").
#     * Utilizar la técnica de Combinatorial Hashing.
