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

        song_raw.append([new_path,genres])

##print({len(song_raw)}) testing
##print(song_raw[0]) testing
