# train.py
# contains the main training loop for all of the models I am working with

import sys
import os

from Utilities.note import Note
from Utilities.midi_parser_random import output_midi
from Utilities.visualizer import visualize_multiexpert
from Models.generative import *
from Models.identity import Identity
from Models.product_of_experts import MultiExpert
from Models.expert_models import VoiceSpacingExpert, VoiceContourExpert, RhythmExpert

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
	voice = onehot_matrix_to_int_vector(one_hot_voice) if one_hot_voice.ndim==2 else one_hot_voice
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

# main training loop
# model is a subclass of GenerativeLSTM or a MultiExpert
# model name is the string to use when figuring out the output directory
# dataset is a four dimensional python array - first dimension is piece, second is voice, third is time, fourth is pitch
# min_num, max_num and timestep_length are constants describing the dataset, load_dataset returns them
# visualize tells us whether or not to call the visualization function, for now this is only supported with MultiExpert
def train(model, model_name, dataset, min_num, max_num, timestep_length, visualize=False):
	
	output_dir = '../Data/Output/' + model_name +'/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/' + model_name): os.mkdir('../Data/Output/' + model_name)
	os.mkdir(output_dir)

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
	minibatch_count = 0 if visualize else 1
	best_loss = np.inf
	terminate = False
	while not terminate:
		# choose our minibatch
		pieces = np.random.choice(len(training_set), size=minibatch_number, replace=False)
		# train
		print('Minibatch', minibatch_count, ": ", model.train(pieces, training_set, minibatch_size))
		# every 20 minibatches, validate
		if minibatch_count % 20 == 0:
			print("Minibatch ", minibatch_count)
			pieces = np.arange(len(validation_set))
			validation_minibatch_size = min([len(piece[0]) for piece in validation_set])
			print(validation_minibatch_size)
			# validate
			if visualize:
				loss, minibatch, prior_timesteps, timestep_info = model.validate(pieces, validation_set, validation_minibatch_size)
				if not os.path.exists(output_dir + 'visualize/'): os.mkdir(output_dir + 'visualize/')
				if type(model) is MultiExpert:
					visualize_multiexpert(model, minibatch[:10], prior_timesteps[:10], timestep_info[:10], directory=output_dir + 'visualize/minibatch' + str(minibatch_count) +'/')
				else:
					visualize_expert(model, minibatch[:10], prior_timesteps[:10], timestep_info[:10], directory=output_dir + 'visualize/minibatch' + str(minibatch_count) + '/')
			else:
				loss = model.validate(pieces, validation_set, validation_minibatch_size)
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

# shortcut for instantiating some common model structures
# model name indicates which model, see source for exactly what it does
# min_num, max_num, timestep_length are constants relevant to the dataset, load_dataset returns them
# visualize tells the model whether to compile functions for getting probabilities in order to visualize output
def instantiate_model(model_name, min_num, max_num, timestep_length, visualize):
	if model_name == 'SimpleGenerative':
		model = SimpleGenerative(max_num - min_num, [100,200,100], 4, 3)
	elif model_name == 'VoiceSpacingExpert':
		model = VoiceSpacingExpert(max_num-min_num, [100,200,100], 0, 3)
	elif model_name == 'Identity':
		model = Identity(max_num - min_num, [100,100], 3)
	elif model_name == 'VoiceContourExpert':
		model = VoiceContourExpert(min_num, max_num, [100,200,100], 3)
	elif model_name == 'RhythmExpert':
		model = RhythmExpert(240*4//timestep_length, max_num-min_num, [100,200,100], 3)
	elif model_name == 'MultiExpert':
		model = MultiExpert(['SimpleGenerative', 'VoiceSpacingExpert', 'VoiceContourExpert', 'RhythmExpert'], 4, 3,  min_num, max_num, timestep_length, transparent=visualize)
	elif model_name == 'justSpacingContour':
		model = MultiExpert(['VoiceSpacingExpert', 'VoiceContourExpert'], 4, 3,  min_num, max_num, timestep_length, transparent=visualize)
	elif model_name == 'justSimpleRhythm':
		model = MultiExpert(['SimpleGenerative', 'RhythmExpert'], 4, 3,  min_num, max_num, timestep_length, transparent=visualize)
	else:
		print("Unknown Model")
		sys.exit(1)
	return model

if __name__=='__main__':
	dataset, min_num, max_num, timestep_length = load_dataset("../Data/train.p", "../Data/validate.p")
	model = instantiate_model('MultiExpert', min_num, max_num, timestep_length, visualize=True)
	train(model, 'MultiExpert', dataset, min_num, max_num, timestep_length, visualize=True)
