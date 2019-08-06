import data_utils_train
import numpy as np
import time
import csv
import glob

np.set_printoptions(threshold=np.nan) #Comment that line out, to print reduced matrices

chord_train_dir = './data1/'
chord_train_files = glob.glob("%s*.mid" %(chord_train_dir))
resolution_factor = 12
time=float(0)
prev = float(0)
#Preprocessing: Get highest and lowest notes + maximum midi_ticks overall midi files
chord_lowest_note, chord_highest_note = data_utils_train.getNoteRangeAndTicks(chord_train_files, res_factor=resolution_factor)

#Create Piano Roll Representation of the MIDI files. Return: 3-dimensional array or shape (num_midi_files, maximum num of ticks, note range)
chord_roll = data_utils_train.fromMidiCreatePianoRoll(chord_train_files, chord_ticks, chord_lowest_note, chord_highest_note,
                                                res_factor=resolution_factor)

#Double each chord_roll and mel_roll. Preprocessing to create Input and Target Vector for Network
double_chord_roll = data_utils_train.doubleRoll(chord_roll)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)

