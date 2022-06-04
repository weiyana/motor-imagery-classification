from scipy.signal import butter, lfilter
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, kaiserord, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def firwin_bandpass_filter(signal,lowcut,highcut,Fs):
    stopbbanAtt = 60  #stopband attenuation of 60 dB.
    width = .5 #This sets the cutoff width in Hertz
    nyq = 0.5*Fs
    ntaps, gb= kaiserord(stopbbanAtt, width/nyq)
    atten = kaiser_atten(ntaps, width/nyq)
    beta = kaiser_beta(atten)
    a = 1.0
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,window=('kaiser', beta), scale=False)
    filtered_signal = filtfilt(taps, a, signal)

    return filtered_signal