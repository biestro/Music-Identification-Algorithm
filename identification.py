import numpy as np
import matplotlib.pyplot as plt
import time
import librosa                
from scipy.fft import fft


def readthis(string, srate):
    signal, samplerate = librosa.load(string, sr = srate, mono = True)
    return(signal, samplerate)

def fourier(signal, samplerate, divisor):
    T = 1 / samplerate
    interval = int(samplerate / divisor)
    interval_sec = interval / samplerate
    magnitude = np.zeros((0,interval//2))
    position = 0
    size = int(len(signal) / interval)
    for ii in range(size):
        mini_sig = signal[position:position + interval] 
        N = len(mini_sig)
        yf = abs(fft(mini_sig))[:interval//2]
        yf = yf.reshape(1, interval//2)
        magnitude = np.concatenate((magnitude , yf), axis = 0)
        position += interval
    mag_db = np.log(magnitude)
    f = np.linspace(0, samplerate / 2, int(samplerate / 2))
    
    return(mag_db, None, magnitude, None, interval)

def mypeaks(matrix, div):

    m = matrix# np.array([mag_MAIN[90], mag_MAIN[91], mag_MAIN[92], mag_MAIN[93]])
    x = m[:,:int(1000 / div)] # hasta 2000 Hz
    fvec = np.arange(0, len(x[0]) * div, div)
    a_n = 4 # número de pedazos
    b_n = int(1000 / a_n / div) # 
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
def getscore(song_magnitude, sample_magnitude, divisor):
    
    assert len(sample_magnitude)<len(song_magnitude), 'Array too big!' 
    score = []
    song_peaks = mypeaks(song_magnitude, divisor)
    sample_peaks = mypeaks(sample_magnitude, divisor)
    for ii in range(len(song_peaks) - len(sample_peaks)):
        score.append(sum(sum(sample_peaks == song_peaks[ii: len(sample_peaks) + ii])))     
    score_max = max(score)
    return(score_max)

def findmatch(goal, div, srate):
    samplerate = srate
    div = np.load('./database_peaks/div.npy')
    mag = np.load('./database_peaks/mag.npy', allow_pickle=True)
    pairs = np.load('./database_peaks/pairs.npy')
    
    songid = pairs[:,0]          
    sample, _ = readthis('./recordings_audio/' + goal, 44100)
    _, _, samplemag, _, _ = fourier(sample, samplerate, divisor = div)
    
    size = len(mag)

    samplescores = list(np.zeros(size))

    # se le asigna puntaje a cada canción
    for ii in range(size):
        samplescores[ii] = getscore(mag[ii], samplemag, div)

    sampleid = int(np.argwhere(samplescores == np.amax(samplescores)))

    runnerup = np.sort(samplescores)[::-1][1]

    winner = pairs[sampleid][1]

    print('============================================')
    print('\n    I have a match! \ (•◡•) /             ')
    print(f'\n\nSong is: {winner} \nwith a {samplescores[sampleid] - runnerup} point margin over the runner up\n\n')
    print('============================================')
    
    return(winner, samplescores, songid)

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