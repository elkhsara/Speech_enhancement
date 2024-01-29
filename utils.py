import matplotlib.pyplot as plt
import IPython.display as ipd

import librosa
from librosa.util import find_files
from librosa import load

import os
import numpy as np
import random

from pesq import pesq
import soundfile as sf
from pystoi import stoi




def calculate_snr(clean, noisy):
    min_length = min(len(clean), len(noisy))
    clean = clean[:min_length]
    noisy = noisy[:min_length]
    power_clean = np.sum(clean**2) / len(clean)
    power_noise = np.sum((clean - noisy)**2) / len(clean)
    snr = 10 * np.log10(power_clean / power_noise)
    return snr

def get_duration():
    file_list = os.listdir("clean/train_small")
    dur = []
    for audio in file_list:
        try:
            mix, sr = load(os.path.join("clean/train_small", audio))
            duration = librosa.get_duration(y=mix, sr=sr)
            dur.append(duration)
        except IndexError as e:
            print("Error processing audio %s: %s" % (audio, str(e)))
    return dur


def print_plot_play(x, Fs, text=''):
    duration = len(x) / Fs  
    time_axis = np.linspace(0, duration, len(x)) 
    print('%s Fs = %d, Duration = %.2f seconds, x.shape = %s, x.dtype = %s' % (text, Fs, duration, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(time_axis, x, color='gray')
    plt.xlim([0, duration])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))

def SaveSpectrogram(noisy,clean, filename, path) :
    S_noisy = np.abs(librosa.stft(noisy,n_fft=window_size,hop_length=hop_length,window=window)).astype(np.float32)
    S_clean = np.abs(librosa.stft(clean,n_fft=window_size,hop_length=hop_length,window=window)).astype(np.float32)
    
    path =  './Spectrogram' + path
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path,filename+'.npz'),noisy=S_noisy,clean=S_clean)
    
def LoadSpectrogram(path) :
    path = './Spectrogram' + path
    filelist = find_files(path, ext="npz")
    x_list = []
    y_list = []
    for file in filelist :
        data = np.load(file)
        x_list.append(data['noisy'])
        y_list.append(data['clean'])
    return x_list, y_list


def Magnitude_phase(spectrogram) :
    Magnitude_list = []
    Phase_list = []
    for X in spectrogram :
        mag, phase = librosa.magphase(X)
        Magnitude_list.append(mag)
        Phase_list.append(phase)
    return Magnitude_list, Phase_list

def create_spectrograms(noisy_path , clean_path , terminal_path) :
    
    file_list = os.listdir(noisy_path)
    
    for audio in file_list:
        try:
            if os.path.exists(os.path.join('./Spectrogram' + terminal_path, audio + '.npz')):
                print("Already exists")
                continue
            
            # Load the noisy and clean audio
            noisy, _ = load(os.path.join(noisy_path, audio), sr=None)
            clean, _ = load(os.path.join(clean_path, audio), sr=None)
            
            # Save the spectrogram
            SaveSpectrogram(noisy, clean, audio,terminal_path)
        except IndexError as e:
            print("Wrong Directory")
            pass
    print("Success")

    def spectrogramme_display(audio1, audio2):
        S1 = np.abs(librosa.stft(audio1,n_fft=window_size,hop_length=hop_length,window=window)).astype(np.float32)
        S2 = np.abs(librosa.stft(audio2,n_fft=window_size,hop_length=hop_length,window=window)).astype(np.float32)
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    # Plot the first spectrogram
        img1 = librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max),
                                    y_axis='log', x_axis='time', ax=axs[0])
        axs[0].set_title('Power spectrogram - S1')
        fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")

    # Plot the second spectrogram
        img2 = librosa.display.specshow(librosa.amplitude_to_db(S2, ref=np.max),
                                    y_axis='log', x_axis='time', ax=axs[1])
        axs[1].set_title('Power spectrogram - S2')
        fig.colorbar(img2, ax=axs[1], format="%+2.0f dB")

        plt.tight_layout()
        plt.show()


def predict_and_reconstruct_1(model,audio_path, window_size=1024, hop_length=208):
    mix, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(mix,n_fft=window_size,hop_length=hop_length)
    S_mag , phase = librosa.magphase(S)
    S_mag = S_mag[np.newaxis, 1:, 1:, np.newaxis]  # Adjust shape for the model input
    clean_mag_pred = model.predict(S_mag)
    #clean_mag_pred = S_mag 
    S_mag_pred = np.pad(clean_mag_pred, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant', constant_values=1)
    S_mag_pred = S_mag_pred.reshape(S_mag_pred.shape[1],S_mag_pred.shape[2])
    clean_spec_pred = S_mag_pred * phase
    clean_audio_pred = librosa.istft(clean_spec_pred,n_fft=1024,hop_length=hop_length)
    return S_mag_pred , S_mag ,clean_audio_pred, sr

def predict_and_reconstruct_1_lstm(model,audio_path, window_size=1024, hop_length=208, time_step= 30):
    mix, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(mix,n_fft=window_size,hop_length=hop_length)
    S_mag , phase = librosa.magphase(S)
    S_mag = S_mag.T
    shape_train =(np.shape(S_mag)[0])//time_step
    max_train = shape_train * time_step
    phase = phase.T[:max_train,:]
    X3d_train = S_mag[:max_train,:]
    X3d_train = np.reshape(X3d_train, (shape_train,time_step,513))
    clean_mag_pred = model.predict(X3d_train)
    S_mag_pred = clean_mag_pred.reshape(max_train,513).T
    clean_spec_pred = S_mag_pred.T * phase
    clean_spec_pred = clean_spec_pred.T
    print(np.shape(clean_spec_pred))
    clean_audio_pred = librosa.istft(clean_spec_pred,n_fft=1024,hop_length=hop_length)
    return S_mag_pred , S_mag ,clean_audio_pred, sr



def write_files(model,noisy_path, predict_folder,lstm=False):
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
        
    file_list = os.listdir(noisy_path)
    for audio in file_list:
        if lstm:
            _ ,_ , clean_audio_pred, sr = predict_and_reconstruct_1_lstm(model,
                                         os.path.join(noisy_path, audio), window_size, hop_length)
        else:
            _ ,_ , clean_audio_pred, sr = predict_and_reconstruct_1(model,
                                         os.path.join(noisy_path, audio), window_size, hop_length)
        predict_audio_path = os.path.join(predict_folder, audio)
        sf.write(predict_audio_path, clean_audio_pred, sr)
    print(f"Done writing to {predict_folder}")


def evaluate_model(clean_path, noisy_path,sr=8000):

    # Get the list of file names in the noisy path
    file_list = os.listdir(noisy_path)

    pesq_scores = []
    stoi_scores = []
    snr_scores = []

    for audio in file_list:
        try:
            # Get the corresponding clean audio file path
            clean_audio_path = os.path.join(clean_path, audio)
            # Load the true clean audio for comparison
            clean_audio, _ = librosa.load(clean_audio_path, sr=None)
            clean_audio_pred, _ = librosa.load(os.path.join(noisy_path, audio), sr=None)
            min_length = min(len(clean_audio), len(clean_audio_pred))
            clean_audio = clean_audio[:min_length]
            clean_audio_pred = clean_audio_pred[:min_length]

            # Compute PESQ
            pesq_score = pesq(sr, clean_audio, clean_audio_pred, 'nb')
            pesq_scores.append(pesq_score)

            # Compute STOI
            stoi_score = stoi(clean_audio, clean_audio_pred,fs_sig = 8000)
            stoi_scores.append(stoi_score)

            # Compute SNR
            snr_score = calculate_snr(clean_audio, clean_audio_pred)
            snr_scores.append(snr_score)
        except Exception as e:
            print(f"Error processing audio {audio}: {str(e)}")

    return  pesq_scores, stoi_scores, snr_scores


def noising_1audio(clean_audio, noise):
 #random number to start noise
 num=random.randint(0, len(noise)-len(clean_audio)) # pour prendre une partie aléatoire du signal de bruit
 RSB=random.random()*6 #noise coeff
 noise=noise[num:num+len(clean_audio)]
 noisy_son=clean_audio + noise* np.sqrt(clean_audio.dot(clean_audio)/noise.dot(noise))*math.pow(10,-RSB/20)
 return noisy_son

def noising_all(clean_path, noisy_path,noise,sr=8000):
    if not os.path.exists(noisy_path):
        os.makedirs(noisy_path)
    file_list = os.listdir(noisy_path)
    for audio in file_list:
        noisy_audio = noising_1audio(audio,noise)
    noisy_audio_path = os.path.join(noisy_path, audio)
    sf.write(noisy_audio_path, noisy_audio, sr)
    print(f'Done noising and writing noisy audios to{noisy_path}')