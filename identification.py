import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import librosa


# función para leer cosas
def read_this(string, srate):
    signal, samplerate = librosa.load(string, sr = srate, mono = True)
    return(signal, samplerate)

def spectrogram(sample, samplerate, divs, title, bartitle):
    
    
    _, _, matrix, _, _ = fourier(sample, samplerate, divisor = divs)

    
    T = 1 / samplerate
    time = np.arange(0, len(sample) / (samplerate), T) 
    
    freq = np.linspace(0, samplerate / 2, int(samplerate / 2))
    
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

# función que me regresa la transformada de Fourier iterada
def fourier(signal, samplerate, divisor):

    interval = int(samplerate / divisor)                      
    
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

    mag_db = np.log(magnitude, where=(magnitude>0))                               
                                                            
    
    return(mag_db, None, magnitude, None, interval)

def mypeaks(matrix, division):
    
    div = division
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
def getscore(song_magnitude, sample_magnitude, division):
    
    assert len(sample_magnitude)<len(song_magnitude), 'Array too big!' 
    score = []
    song_peaks = mypeaks(song_magnitude, division)
    sample_peaks = mypeaks(sample_magnitude, division)
    for ii in range(len(song_peaks) - len(sample_peaks)):
        score.append(sum(sum(sample_peaks == song_peaks[ii: len(sample_peaks) + ii])))     
    score_max = max(score)
    return(score_max)

def findmatch(goal, srate):
    divs = np.load('./music_data/database_peaks/div.npy')
    mag = np.load('./music_data/database_peaks/mag.npy', allow_pickle=True)
    pairs = np.load('./music_data/database_peaks/pairs.npy')
    samplerate = np.load('./music_data/database_peaks/samplerate.npy')
    
    songid = pairs[:,0]          
    sample, _ = read_this('./music_data/recordings_audio/' + goal, srate)
    _, _, samplemag, _, _ = fourier(sample, samplerate, divisor = divs)
    
    size = len(mag)

    samplescores = list(np.zeros(size))

    # se le asigna puntaje a cada canción
    for ii in range(size):
        samplescores[ii] = getscore(mag[ii], samplemag, divs)
    
    location = np.argwhere(samplescores == np.amax(samplescores))
    
    '''if isinstance(location, np.ndarray):
        print(f'Couldn\'t find a consistent match (two possible candidates)')
        return(None, None, None)'''
        
    sampleid = int(location[0])
    
    runnerup = np.sort(samplescores)[::-1][1]

    difference = samplescores[sampleid] - runnerup
    
    
    if difference < 10:
        print('\n==============================================\n')
        print('         (╯°□°）╯︵ ┻━┻    ')
        print(f'\nCouldn\'t find a consistent match\n\n (score difference = {difference}) < 10')
        print('\n==============================================\n')
        return(None, None, False)
    
    print('\n==============================================\n')
    print('        I found a match!!  \ (•◡•) /  ')
    print('\n==============================================\n')
    winner = pairs[sampleid][1]

    print(f'Song is: {winner} with a {difference} point margin over the runner up')

    plt.bar(songid, samplescores)
    plt.xlabel('Song ID')
    plt.title('Puntajes para "' + goal + '"')
    plt.ylabel('Score')
    plt.show()

    return(winner, samplescores, songid)


