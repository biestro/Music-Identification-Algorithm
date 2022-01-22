"""
This file includes most of the functions used by the main.py file

Author: Alberto Ruiz Biestro

Please read the license before deciding to copy any of this.

Last revision: 21/01/2022
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import librosa
import pyaudio
import wave


def read_this(string, srate):
    """  
    Reads a specified audio file (specified in the 'string' parameter). 
    The 'srate' parameter specifies the samplerate. This is usually
    44100 Hz, but it is used for testing purposes.
    
    It is converted and returned as a mono signal for it to be compatible 
    with the FFT function,
    
    i.e., the 'string' could be '$HOME/Music/sample1.wav'.
    
    """

    SIGNAL, SAMPLERATE = librosa.load(string, sr = srate, mono = True)
    return(SIGNAL, SAMPLERATE)



def spectrogram(sample, SAMPLERATE, divs, title, bartitle):
    """
	This function is solely for the research paper. It ouputs a spectrogram
	of the selected song. It has little to none use in the identification
	of samples and recordings.

	"""
    
    _, _, matrix, _, _ = fourier(sample, SAMPLERATE, divisor = divs)

    
    T = 1 / SAMPLERATE
    time = np.arange(0, len(sample) / (SAMPLERATE), T) 
    
    freq = np.linspace(0, SAMPLERATE / 2, int(SAMPLERATE / 2))
    
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




def fourier(SIGNAL, SAMPLERATE, divisor):
	"""
	The signal is first cut to be properly divisible by the 'divisior' factor. 
	It then performs the fast FFT from scipy.fft on each interval or block,
	whose length is specified by the 'divisor'. What follows is derived from
	simple array operations dealing with the lengths, ranges, and the 'universal' 
	44.1kHz constant.

	It returns the final magnitude in decibels, the final magnitude, and the interval
	calculated with help of the divisor.

	"""

    interval = int(SAMPLERATE / divisor)                      
    
    magnitude = np.zeros((0,interval//2))                     
    position = 0                                              
    size = int(len(SIGNAL) / interval)                        
    
    for ii in range(size):                                    
        mini_sig = SIGNAL[position:position + interval] 
        N = len(mini_sig)
        yf = abs(fft(mini_sig))[:interval//2]                 
        yf = yf.reshape(1, interval//2)                       
        magnitude = np.concatenate((magnitude , yf), axis = 0)
        position += interval                                  

    mag_db = np.log(magnitude, where=(magnitude>0))                               

    return(mag_db, None, magnitude, None, interval)



def mypeaks(matrix, division):
	"""
	Calculates the maximum in every division within the block passed (the 'matrix'
	parameter). 
	
	The variable 'a_n' specifies the divisions or windows in which it will extract
	the maximum's index position.
	"""
    
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



def getscore(song_magnitude, sample_magnitude, division):
	"""
	This function calculates the final score of the similarty between
	the recording and the specified database or pool song. A higher score corresponds 
	to a higher likelihood that the peaks from the recording match the peaks 
	from said database song.

	It does this through comparing the array peaks from the recording
	to those from the specified database song in each interval, and then 
	advancing unit step through the peak array (from the database song) and 
	comparing again. During the comparison of each block, there is a binary
	matrix with a '1' corresponding to a peak-match and a '0' corresponding 
	to position without a match. It performs a (double) sum of the 1's and 
	then saves this score in the score variable. This process is repeated 
	through the next block.

	It then gets the maximum score from the array of scores. It does this to 
	prevent an incredibly large song to match a few peaks, but due to its length, 
	getting a higher score than the more appropiate match.

	This guarantees a sensible score for comparison between a song pool.
	"""
    
    assert len(sample_magnitude)<len(song_magnitude), 'Array too big!' 
    score = []
    song_peaks = mypeaks(song_magnitude, division)
    sample_peaks = mypeaks(sample_magnitude, division)
    for ii in range(len(song_peaks) - len(sample_peaks)):
        score.append(sum(sum(sample_peaks == song_peaks[ii: len(sample_peaks) + ii])))     
    score_max = max(score)
    return(score_max)


def findmatch(goal, srate):
	"""
	This function attempts to find a match between the 'goal' and the whole
	song pool. 

	It compares the score differences between all the songs in the database.

	Should the score be too low, it returns a message specifying the situation.
	Otherwise, it returns the allegedly matched song in the database and the 
	process is deemed complete.

	"""
    divs = np.load('./music_data/database_peaks/div.npy')
    mag = np.load('./music_data/database_peaks/mag.npy', allow_pickle=True)
    pairs = np.load('./music_data/database_peaks/pairs.npy')
    SAMPLERATE = np.load('./music_data/database_peaks/SAMPLERATE.npy')
    
    songid = pairs[:,0]          
    sample, _ = read_this('./' + goal, srate)
    _, _, samplemag, _, _ = fourier(sample, SAMPLERATE, divisor = divs)
    
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
        return(None, None, [False])
    
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

def myrecording():
	"""
	With the aid of 'Cryo' on StackOverflow, this records the sound later used
	for the identification.
	"""
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	RECORD_SECONDS = 10
	WAVE_OUTPUT_FILENAME = 'recording.wav'

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
		        channels=CHANNELS,
		        rate=RATE,
		        input=True,
		        frames_per_buffer=CHUNK)

	print('* recording')

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	print('* done recording')

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

