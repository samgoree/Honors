#midi_parser.py
# Uses mido from https://github.com/olemb/mido to parse midi files into usable format (pickle)

# midi is kinda weird, a track contains events, events are either note on or note off, with a note number,
# a channel, a velocity (how hard the key was pressed) and a time since  the last note

import mido
import sys
import copy
import os
import pickle


from Utilities.note import *


# parse a midi file and return the piece as a list of num_voices lists of notes, some notes may not be assigned to voices, they will be in piece[num_voices]
def parse_midi(filepath, num_voices, debug=False):
    mid = mido.MidiFile(filepath)
    piece = [[] for i in range(num_voices + 1)]
    time = 0
    worklist = []
    for track in mid.tracks:
        for message in track:
            if debug: 
                print(message)
                print_piece([worklist])
            # If we have all voices present, label them
            if len(worklist) == num_voices:
                worklist = sorted(worklist, key=(lambda n: n.num))
                for i,n in enumerate(worklist): n.voice=i
            time += message.time
            # add new notes to the worklist
            if message.type == 'note_on':
                i = 0
                # refresh the worklist - we need this because midi notates voice exchanges as a single note
                tempworklist = []
                while i < len(worklist):
                    if worklist[i].start_time != time:
                        worklist[i].stop_time = time
                        piece[worklist[i].voice] += [worklist[i]]
                        newNote = copy.deepcopy(worklist[i])
                        del(worklist[i])
                        newNote.start_time = time
                        newNote.stop_time = -1
                        tempworklist += [newNote]
                    else: i+=1

                worklist += [Note(message.note, time, num_voices=num_voices)]
                worklist += tempworklist
            # remove notes from the worklist when they finish
            elif message.type == 'note_off':
                for n in worklist:
                    if n.num == message.note:
                        n.stop_time = time
                        piece[n.voice] += [n]
                        worklist.remove(n)
                        break
    if debug: print_piece(piece)
    return piece

# take a piece where piece[-1] contains notes that do not have voices assigned and assign them most parsimoniously to avoid voice crossing and overlap and pick the voice best matching the note's pitch
def assign_voices(piece, debug=False):
    while len(piece[-1]) > 0:
        i = 0
        # loop through unassigned notes
        while i < len(piece[-1]):
            # handle voice crossing: for each note that overlaps in time with the designated note, change the upper/lower bounds on our note's voice
            for j in range(len(piece)-1):
                for k in range(len(piece[j])):
                    if piece[j][k].start_time < piece[-1][i].stop_time and piece[j][k].stop_time > piece[-1][i].start_time:
                        if piece[j][k].num < piece[-1][i].num:
                            if debug: print(piece[-1][i], "lower bound set to ", max(piece[-1][i].lower_bound, j + 1), "because of crossing with ", piece[j][k])
                            piece[-1][i].lower_bound = max(piece[-1][i].lower_bound, j + 1)
                        else:
                            if debug: print(piece[-1][i], "upper bound set to ", min(piece[-1][i].upper_bound, j - 1), "because of crossing with ", piece[j][k])
                            piece[-1][i].upper_bound = min(piece[-1][i].upper_bound, j - 1)
            # if we've pinned down which voice it is, don't test for overlap
            if piece[-1][i].lower_bound == piece[-1][i].upper_bound:
                v = piece[-1][i].lower_bound
                if debug: print(piece[-1][i], " assigned to voice ", v)
                piece[-1][i].voice = v
                piece[v] += [piece[-1][i]]
                piece[-1].remove(piece[-1][i])
                piece[v] = sorted(piece[v], key=(lambda n: n.start_time))
                continue
            elif piece[-1][i].lower_bound > piece[-1][i].upper_bound:
                print("Unable to label ", piece[-1][i], "lower bound: ", piece[-1][i].lower_bound, " upper bound: ", piece[-1][i].upper_bound)
                return piece
            # handle voice overlap: for each note that occurs most recently before this note in a voice, adjust the lower/upper bound
            for j in range(len(piece)-1):
                last_note = None
                for k in range(len(piece[j])):
                    if piece[j][k].stop_time < piece[-1][i].start_time: last_note = piece[j][k]
                    else: break
                if last_note is None: continue
                elif last_note.num < piece[-1][i].num:
                    if debug: print(piece[-1][i], "lower bound set to ",  max(piece[-1][i].lower_bound, j), "because of overlap with ", piece[j][k])
                    piece[-1][i].lower_bound = max(piece[-1][i].lower_bound, j)
                elif last_note.num > piece[-1][i].num:
                    if debug: print(piece[-1][i], "upper bound set to ", min(piece[-1][i].upper_bound, j), "because of overlap with ", piece[j][k])
                    piece[-1][i].upper_bound = min(piece[-1][i].upper_bound, j)
            # try to assign a voice
            if piece[-1][i].lower_bound == piece[-1][i].upper_bound:
                v = piece[-1][i].lower_bound
                if debug: print(piece[-1][i], " assigned to voice ", v)
                piece[-1][i].voice = v
                piece[v] += [piece[-1][i]]
                piece[-1].remove(piece[-1][i])
                piece[v] = sorted(piece[v], key=(lambda n: n.start_time))
                continue
            elif piece[-1][i].lower_bound > piece[-1][i].upper_bound:
                print("Unable to label ", piece[-1][i], "lower bound: ", piece[-1][i].lower_bound, " upper bound: ", piece[-1][i].upper_bound)
                return piece
            else:
                closest = None
                for j in range(piece[-1][i].lower_bound, piece[-1][i].upper_bound+1):
                    last_note = None
                    for k in range(len(piece[j])):
                        if piece[j][k].stop_time < piece[-1][i].start_time: last_note = piece[j][k]
                        else: break
                    if last_note is None: continue
                    if closest is None and last_note is not None: closest = last_note
                    elif abs(last_note.num - piece[-1][i].num) < abs(closest.num - piece[-1][i].num):
                        closest = last_note
                if closest is not None:
                    v = closest.voice
                    if debug: print(piece[-1][i], "couldn't be pinned down, assigned to closest earlier note voice", v)
                else:
                    # this is an annoying case - unlabelable thing at the start of the piece
                    for j in range(piece[-1][i].lower_bound, piece[-1][i].upper_bound+1):
                        next_note = None
                        for k in range(len(piece[j])):
                            if piece[j][k].stop_time < piece[-1][i].start_time: last_note = piece[j][k]
                            else: 
                                next_note = piece[j][k]
                                break
                        if next_note is None: continue
                        if closest is None and next_note is not None: closest = next_note
                        elif abs(next_note.num - piece[-1][i].num) < abs(closest.num - piece[-1][i].num):
                            closest = next_note
                    if closest is None:
                        continue
                    else: 
                        v = closest.voice
                        if debug: print(piece[-1][i], "couldn't be pinned down, assigned to closest later note voice", v)

                
                piece[-1][i].voice = v
                piece[v] += [piece[-1][i]]
                piece[-1].remove(piece[-1][i])
                piece[v] = sorted(piece[v], key=(lambda n: n.start_time))
                continue
            i+=1
    return piece

def print_piece(piece):
    for i in range(len(piece)):
        for j in range(len(piece[i])):
            sys.stdout.write(str(piece[i][j]) + ', ')
        sys.stdout.write('\n')


def parse_dataset(directory, num_voices):
    files = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    pieces = []
    for f in files:
        print("Parsing", f)
        pieces += [assign_voices(parse_midi(f, num_voices))]
    return pieces


if __name__ == '__main__':
    piece = parse_midi('/usr/users/quota/students/18/sgoree/Downloads/JSB Chorales/train/164.mid', 4, debug=True)
    piece = assign_voices(piece, debug=True)
    #pickle.dump(parse_dataset('/usr/users/quota/students/18/sgoree/Downloads/JSB Chorales/train/', 4), 'Data/train.p')
    #pickle.dump(parse_dataset('/usr/users/quota/students/18/sgoree/Downloads/JSB Chorales/validate/', 4), 'Data/validate.p')
    #pickle.dump(parse_dataset('/usr/users/quota/students/18/sgoree/Downloads/JSB Chorales/test/', 4), 'Data/test.p')
    













