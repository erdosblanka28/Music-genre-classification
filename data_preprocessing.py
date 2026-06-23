import librosa #for MFCC,FFT,DCT
import os
import numpy as np


##Reading in data##

path_search = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path_search,"genres", "input.mf")

song_raw = [] 

with open(path, "r") as file: #with automatically closes file

    for row in file:

        row = row.strip() #remove \n
        splitted_data = row.split('\t')
        old_path = splitted_data[0]
        genres = splitted_data[1]

        song_file_name = os.path.basename(old_path) #gets name of .wav file
        song_folder = os.path.dirname(old_path)
        song_folder_name = os.path.basename(song_folder) #gets the name of genre folder
        new_path = os.path.join(path_search,"genres",song_folder_name,song_file_name)

        song_raw.append((new_path,genres))

##print({len(song_raw)}) testing
##print(song_raw[0]) testing

mfcc = []
label = []

for new_path, genres in song_raw:
    
    amplitude, sampling_rate = librosa.load(new_path)
    mfcc_1 = librosa.feature.mfcc(y=amplitude, sr=sampling_rate, n_mfcc=20) #cuts the songs into ~25ms pieces, saving the 20 first coefficient
    mfcc_mean = np.mean(mfcc_1, axis=1) #taking the mean for each row (20 seperate mfcc value averaged over the whole song)
   #saving to list
    mfcc.append(mfcc_mean)
    label.append((genres))

#converting,compressing & saving the data
x = np.array(mfcc)
y = np.array(label)
np.savez_compressed("music_features.npz", features=x,labels=y)

print("music_features.npz is successfully saved to current folder.")
