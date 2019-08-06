# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:06:49 2015

@author: Konstantin
"""

from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage
from keras.callbacks import Callback
import numpy as np
#np.set_printoptions(threshold=np.nan)


def getNoteRangeAndTicks(files_dir, res_factor=1):
    ticks = []
    notes = []
    for file_dir in files_dir:
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)                   
        
        for track in mid.tracks: #preprocessing: Checking range of notes and total number of ticks
            num_ticks = 0           
            for message in track:
                if not isinstance(message, MetaMessage):
                    if message.type == 'note_on':
                        notes.append(message.note)
                        num_ticks += int(message.time/res_factor)
                ticks.append(num_ticks)
                    
    return min(notes), max(notes), max(ticks)


def fromMidiCreatePianoRoll(files_dir, ticks, lowest_note, highest_note, res_factor=1):
   # num_files = len(files_dir)        
        
    #piano_roll = np.zeros((num_files, ticks, highest_note-lowest_note+1), dtype=np.float32)
    piano_roll = np.zeros(( ticks, highest_note-lowest_note+1), dtype=np.float32)
    for i, file_dir in enumerate(files_dir):
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)
        note_time_onoff = getNoteTimeOnOffArray(mid, res_factor)
        note_on_length = getNoteOnLengthArray(note_time_onoff)
        for message in note_on_length:
            piano_roll[ message[1]:(message[1]+int(message[2]/2)), message[0]-lowest_note] = 1
    

    return piano_roll


def fromMidiCreatePianoRoll_ARIMA(files_dir, ticks, lowest_note, highest_note, res_factor=1):
   # num_files = len(files_dir)        
        
    #piano_roll = np.zeros((num_files, ticks, highest_note-lowest_note+1), dtype=np.float32)
    piano_roll = np.zeros(( ticks, 1), dtype=np.float32)
    PianoRoll  =np.empty((0,1), int)
    for i, file_dir in enumerate(files_dir):
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)
        note_time_onoff = getNoteTimeOnOffArray(mid, res_factor)
        note_on_length = getNoteOnLengthArray(note_time_onoff)
        for message in note_on_length:
            #piano_roll[ message[1]:(message[1]+int(message[2])), message[0]-lowest_note] = 1
            piano_roll[ message[1]:(message[1]+int(message[2]/2))]=message[0]
            PianoRoll=np.vstack((PianoRoll,piano_roll))  
    
    np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/piano_seq_ARIMA.csv", PianoRoll, delimiter=",")

    return PianoRoll



def getNoteTimeOnOffArray(mid, res_factor):
    
    note_time_onoff_array = []  
    
    for track in mid.tracks:
        current_time = 0
        for message in track:
            if not isinstance(message, MetaMessage):
                if (message.type == 'note_on' or message.type == 'note_off'):
                    current_time += int(message.time/res_factor)
                    if message.type == 'note_on':
                        note_onoff = 1
                    elif message.type == 'note_off':
                        note_onoff = 0
                    else:
                        print("Error: Note Type not recognized!")
                    
                    note_time_onoff_array.append([message.note, current_time, note_onoff])
    np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/note_time_onoff_array.csv", note_time_onoff_array, delimiter=",")
           
    return note_time_onoff_array
    
    
def getNoteOnLengthArray(note_time_onoff_array):
    note_on_length_array = []
    for i, message in enumerate(note_time_onoff_array):
        if message[2] == 1: #if note type is 'note_on'
            start_time = message[1]
            for event in note_time_onoff_array[i:]: #go through array and look for, when the current note is getting turned off
                if event[0] == message[0] and event[2] == 0:
                    length = event[1] - start_time
                    break
                
            note_on_length_array.append([message[0], start_time, length])
    np.savetxt("C:/Users/Apoorva.Radhakrishna/Documents/Rocmatt/EC2BACKUP/EC2/melody/note_on_length_array.csv", note_on_length_array, delimiter=",")        
    return note_on_length_array
    

def doubleRoll(roll):
    double_roll = []
    for song in roll:
        double_song = np.zeros((roll.shape[1]*2, roll.shape[2]))
        double_song[0:roll.shape[1], :] = song
        double_song[roll.shape[1]:, :] = song
        double_roll.append(double_song)
        
    return np.array(double_roll)
    

def createNetInputs(roll, target, seq_length=3072):
    #roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
    #seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
    #seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars
    
    X = []
    y = []
    
    for i, song in enumerate(roll):
        pos = 0
        while pos+seq_length < song.shape[0]:
            sequence = np.array(song[pos:pos+seq_length])
            X.append(sequence)
            y.append(target[i, pos+seq_length])
            pos += 1

    
    return np.array(X), np.array(y)
    

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
