# train.py
# contains the main training loop for all of the models I am working with

import sys
import os

from Utilities.note import Note
from Utilities.midi_parser_random import output_midi
from Models.generative import *
from Models.product_of_experts import VoiceSpacingExpert

import pickle
import fractions
from collections import OrderedDict
from time import strftime,localtime

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano_lstm

import numpy as np
np.set_printoptions(threshold=np.inf)

epsilon = 10e-9

# takes the path to a pickle file
# assumes the pickled object is python list of pieces, where each piece is a list of voices and each voice is a list of notes.
def load_dataset(*filepath):
	raw_dataset = []
	for path in filepath:
		print("Loading", path)
		# unpickle the file
		raw_dataset += pickle.load(open(path, 'rb'))
	# find the GCD of note lengths
	gcd = None
	min_num = 2^31
	max_num = -2^31
	piece_lengths = []
	for piece in raw_dataset:
		piece_length = 0
		for voice in piece:
			for note in voice:
				if note.stop_time > piece_length: piece_length = note.stop_time
				length = note.stop_time - note.start_time
				if note.num < min_num: min_num = note.num
				if note.num+1 > max_num: max_num = note.num+1
				if gcd is None: gcd = length
				else: gcd = fractions.gcd(gcd, length)
		piece_lengths.append(piece_length)
	# iterate through the voices and notes, convert each voice to a list of timesteps
	num_voices = [len(raw_dataset[i])for i in range(len(raw_dataset))]
	timestep_data = [np.zeros([len(raw_dataset[i]), piece_lengths[i]//gcd, max_num-min_num], dtype='int64')  for i in range(len(raw_dataset))]
	for i,piece in enumerate(raw_dataset):
		for j,voice in enumerate(piece):
			for note in voice:
				timestep_data[i][j,note.start_time//gcd:note.stop_time//gcd, note.num-min_num] = 1
	# return the list of lists of lists of timesteps, the min num and max num
	return timestep_data, min_num, max_num, gcd

# takes a list of one-hot encoded timesteps
# the midi number of the 0-position encoding
# the length of a timestep
# outputs a list of notes
def timesteps_to_notes(one_hot_voice, min_num, timestep_length):
	voice = onehot_matrix_to_int_vector(one_hot_voice)
	i = 0
	curr_note = None
	curr_time = 0
	notes = []
	while i < len(voice):
		curr_time += timestep_length
		if curr_note is not None and voice[i] == curr_note.num:
			i+=1
		elif curr_note is not None:
			curr_note.stop_time = curr_time
			notes.append(curr_note)
			if voice[i] == -1:
				curr_note = None
			else:
				curr_note = Note(voice[i], curr_time)
			i+=1
		else:
			if voice[i] != -1:
				curr_note = Note(voice[i], curr_time)
			i+=1
	return notes


def train(model_name):
	dataset, min_num, max_num, timestep_length = load_dataset("/usr/users/quota/students/18/sgoree/Honors/Data/train.p", "/usr/users/quota/students/18/sgoree/Honors/Data/validate.p")
	
	if model_name == 'SimpleGenerative':
		model = SimpleGenerative(max_num - min_num, [100,200,100], len(dataset[0]), 3)
	elif model_name == 'VoiceSpacingExpert':
		model = VoiceSpacingExpert(max_num-min_num, [100,200,100])
	output_dir = '/usr/users/quota/students/18/sgoree/Honors/Data/Output/' + model_name +'/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	#os.mkdir(output_dir)

	# make validation set
	validation_pieces = np.random.choice(len(dataset), size=len(dataset)//3, replace=False)
	validation_set = []
	training_set = []
	for i in range(len(dataset)):
		if i in validation_pieces:
			validation_set.append(dataset[i])
		else: training_set.append(dataset[i])

	# magic number - a minibatch element is two bars
	minibatch_size = 240*8/timestep_length
	# magic number - a minibatch is 10 such segments
	minibatch_number = 10

	print("Training...")
	# main training loop
	minibatch_count = 1
	best_loss = np.inf
	terminate = False
	while not terminate:
		# choose our minibatch
		pieces = np.random.choice(len(training_set), size=minibatch_number, replace=False)
		minibatch = None
		prior_timesteps = None
		for i in pieces:
			start = np.random.randint(0, len(training_set[i][0])-(minibatch_size+1))
			prior_timestep = training_set[i][None,:,None,start-1] if start > 0 else np.zeros_like(training_set[i][None,:,None,start])
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
		# train
		print('Minibatch', minibatch_count, ": ", model.train(minibatch, prior_timesteps))
		# every 20 minibatches, validate
		if minibatch_count % 20 == 0:
			print("Minibatch ", minibatch_count)
			pieces = np.random.choice(len(validation_set), size=minibatch_number, replace=False)
			minibatch = None
			prior_timesteps = None
			for i in pieces:
				start = np.random.randint(0, len(training_set[i][0])-(minibatch_size+1))
				prior_timestep = training_set[i][None,:,None,start-1]
				if minibatch is None:
					minibatch = training_set[i][None,:,start:start+minibatch_size]
					prior_timesteps = prior_timestep
				else:
					minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
					prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
			# validate
			loss = model.validate(minibatch, prior_timesteps)
			print("Loss: ", loss)
			if loss < best_loss + epsilon: best_loss = loss
			else:
				print("Loss increasing, finishing training...")
				terminate = True
		# every 100 minibatches, sample a piece
		if minibatch_count % 100 == 0 or terminate:
			print("Minibatch", str(minibatch_count), " sampling...")
			sample_piece = training_set[np.random.randint(len(training_set))]
			new_voice = model.generate(sample_piece)
			output_midi([timesteps_to_notes(new_voice, min_num, timestep_length)], output_dir + str(minibatch_count) + '.mid')
		minibatch_count += 1

if __name__=='__main__':
	train('VoiceSpacingExpert')