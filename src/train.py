# train.py
# contains the main training loop for all of the models I am working with

import sys
import os

from Utilities.note import Note
from Utilities.midi_parser_random import output_midi
from Utilities.visualizer import *
from Models.generative import *
from Models.identity import Identity
from Models.product_of_experts import MultiExpert
from Models.expert_models import VoiceSpacingExpert, VoiceContourExpert, RhythmExpert

import dill as pickle # allows us to pickle lambda expressions
#import pickle
import fractions
from collections import OrderedDict
from time import strftime,localtime

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano_lstm

import music21

import numpy as np
np.set_printoptions(threshold=np.inf)

VALIDATION_EVERY = 100
GENERATION_EVERY = 200

epsilon = 10e-9
PPQ = 480 # pulses per quarter note -- a midi thing that specifies the length of a timestep


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
	timestep_data = [np.zeros([len(raw_dataset[i]), piece_lengths[i]//gcd, max_num-min_num], dtype='int64')  for i in range(len(raw_dataset))]
	for i,piece in enumerate(raw_dataset):
		for j,voice in enumerate(piece):
			for note in voice:
				timestep_data[i][j,note.start_time//gcd:note.stop_time//gcd, note.num-min_num] = 1
	# return the list of lists of lists of timesteps, the min num and max num
	return timestep_data, min_num, max_num, gcd

# uses the music21 dataset system instead of the mido powered midi parsing system to load a dataset
# articulation data should be interpreted much as pitch timesteps, except instead of having max_num-min_num pitches, it has 3 possible one-hot values, articulate, sustain and rest, in that order
def load_dataset_music21(file_list):
	print("Loading music21 dataset")
	raw_dataset = []
	gcd = None
	min_num = 2^31
	max_num = -2^31
	piece_lengths = []
	for path in file_list:
		score = music21.converter.parse(path)
		if len(score.parts) != 4:
			#print("Warning: score ", path, " was omitted because it has ", len(score.parts), " voices")
			continue
		# this should look much like load_dataset, just using music21's note class instead of my own (has a length, not start time and stop time, etc.)
		raw_dataset.append(score)
		piece_length = 0
		for voice in score.parts:
			for note in voice.flat:
				if isinstance(note, music21.note.Note):
					stop_time = note.duration.quarterLength + note.offset
					if stop_time > piece_length: piece_length = stop_time
					if note.pitch.midi < min_num: min_num = note.pitch.midi
					if note.pitch.midi+1 > max_num: max_num = note.pitch.midi + 1
					if gcd is None: gcd = note.duration.quarterLength
					else: gcd = fractions.gcd(gcd, note.duration.quarterLength)
				elif isinstance(note, music21.note.Rest):
					stop_time = note.duration.quarterLength + note.offset
					if stop_time > piece_length: piece_length = stop_time
					if gcd is None: gcd = note.duration.quarterLength
					else: gcd = fractions.gcd(gcd, note.duration.quarterLength)


		piece_lengths.append(piece_length)
	# convert each note to a list of timesteps
	timestep_data = [np.zeros([len(raw_dataset[i].parts), int(piece_lengths[i]//gcd), max_num-min_num], dtype='int64')  for i in range(len(raw_dataset))]
	articulation_data = [np.zeros([len(raw_dataset[i].parts), int(piece_lengths[i]//gcd), 3]) for i in range(len(raw_dataset))]
	for i,score in enumerate(raw_dataset):
		for j,voice in enumerate(score.parts):
			for note in voice.flat:
				if isinstance(note, music21.note.Note):
					timestep_data[i][j,int(note.offset//gcd):int((note.offset + note.duration.quarterLength)//gcd), note.pitch.midi - min_num] = 1
					# set onehot of the first timestep of the note to 0
					articulation_data[i][j,int(note.offset//gcd),0] = 1
					# set onehot of the rest within the note to 1
					articulation_data[i][j,int(note.offset//gcd)+1:int((note.offset + note.duration.quarterLength)//gcd), 1] = 1
				elif isinstance(note, music21.note.Rest):
					# set onehot of all the timesteps to 2
					articulation_data[i][j,int(note.offset//gcd):int((note.offset + note.duration.quarterLength)//gcd), 2] = 1
	return timestep_data, articulation_data, min_num,max_num,gcd

# takes a list of one-hot encoded timesteps
# the midi number of the 0-position encoding
# the length of a timestep in midi timesteps
# outputs a list of notes
def timesteps_to_notes(one_hot_voice, one_hot_articulation, min_num, timestep_length_midi):
	voice = onehot_matrix_to_int_vector(one_hot_voice) if one_hot_voice.ndim==2 else one_hot_voice
	articulation = onehot_matrix_to_int_vector(one_hot_articulation) if one_hot_articulation.ndim==2 else one_hot_articulation
	if len(voice) != len(articulation):
		print("Error, paired voice (len",len(voice), ") and articulation (len ", len(articulation), ") are not the same length.")
		sys.exit(1)
	i = 0
	curr_note = None
	curr_time = 0
	notes = []
	stats = [0,0,0]
	while i < len(voice):
		stats[articulation[i]] += 1
		# articulate
		if articulation[i] == 0:
			if curr_note is not None:
				curr_note.stop_time = curr_time
				notes.append(curr_note)
			curr_note = Note(voice[i] + min_num, curr_time)
		# sustain
		elif articulation[i] == 1:
			pass
		# rest
		elif articulation[i] == 2:
			if curr_note is not None:
				curr_note.stop_time = curr_time
				notes.append(curr_note)
				curr_note = None
		curr_time += timestep_length_midi
		i+=1
	print(stats)
	return notes

# main training loop
# model is a subclass of GenerativeLSTM or a MultiExpert
# model name is the string to use when figuring out the output directory
# dataset is a four dimensional python array - first dimension is piece, second is voice, third is time, fourth is pitch
# min_num, max_num and timestep_length are constants describing the dataset, load_dataset returns them
# if output dir is not specified, it will create a new one
# visualize tells us whether or not to call the visualization function, for now this is only supported with MultiExpert
def train(model, model_name, dataset, articulation_data, min_num, max_num, timestep_length, output_dir=None, visualize=False):
	
	if output_dir is None:
		output_dir = '../Data/Output/' + model_name +'/' + strftime("%a,%d,%H:%M", localtime())+ '/'
		if not os.path.exists('../Data/Output/' + model_name): os.mkdir('../Data/Output/' + model_name)
		os.mkdir(output_dir)

	# make validation set
	validation_pieces = np.random.choice(len(dataset), size=len(dataset)//4, replace=False)
	validation_set = []
	validation_set_articulation = []
	training_set = []
	training_set_articulation = []
	for i in range(len(dataset)):
		if i in validation_pieces:
			validation_set.append(dataset[i])
			validation_set_articulation.append(articulation_data[i])
		else: 
			training_set.append(dataset[i])
			training_set_articulation.append(articulation_data[i])


	# magic number - a minibatch element is four bars
	minibatch_size = int(32//timestep_length)
	# magic number - a minibatch is 20 such segments
	minibatch_number = 20

	print("Training...")
	# main training loop
	minibatch_count = 0 if visualize else 1
	best_loss = np.inf
	best_articulation_loss = np.inf
	stop_training_pitch = False
	stop_training_articulation = False
	while not (stop_training_pitch and stop_training_articulation):
		# choose our minibatch
		pieces = np.random.choice(len(training_set), size=minibatch_number, replace=False)
		# train
		if not stop_training_pitch:
			print('Minibatch', minibatch_count, " pitch model: ", model.train(pieces, training_set, training_set_articulation, minibatch_size))
		if not stop_training_articulation:
			print('Minibatch', minibatch_count, " articulation model: ", model.articulation_model.train(pieces, training_set_articulation, minibatch_size))
		# every 20 minibatches, validate
		if minibatch_count % VALIDATION_EVERY == 0:
			print("Minibatch ", minibatch_count)
			pieces = np.arange(len(validation_set))
			validation_minibatch_size = min([len(piece[0]) for piece in validation_set])
			# validate
			if visualize and not stop_training_pitch:
				loss, minibatch, prior_timesteps, timestep_info = model.validate(pieces, validation_set, validation_set_articulation, validation_minibatch_size)
				articulation_loss = model.articulation_model.validate(pieces, validation_set_articulation, validation_minibatch_size)
				if not os.path.exists(output_dir + 'visualize/'): os.mkdir(output_dir + 'visualize/')
				if type(model) is MultiExpert:
					visualize_multiexpert(model, minibatch[:10], prior_timesteps[:10], timestep_info[:10], directory=output_dir + 'visualize/minibatch' + str(minibatch_count) +'/')
				else:
					visualize_expert(model, minibatch[:10], prior_timesteps[:10], timestep_info[:10], directory=output_dir + 'visualize/minibatch' + str(minibatch_count) + '/')
			elif not visualize and not stop_training_pitch:
				loss = model.validate(pieces, validation_set, validation_set_articulation, validation_minibatch_size)
				articulation_loss = model.articulation_model.validate(pieces, validation_set_articulation, validation_minibatch_size)
			else:
				articulation_loss = model.articulation_model.validate(pieces, validation_set_articulation, validation_minibatch_size)
			print("Loss: ", loss)
			print("Articulation Loss: ", articulation_loss)
			if loss < best_loss + epsilon: best_loss = loss
			else:
				print("Loss increasing, finishing training...")
				stop_training_pitch = True
			if articulation_loss < best_articulation_loss + epsilon: best_articulation_loss = articulation_loss
			else:
				print("Articulation loss increasing, finishing training...")
				stop_training_articulation = True
		# every 100 minibatches, sample a piece
		if minibatch_count % GENERATION_EVERY == 0 or (stop_training_pitch and stop_training_articulation):
			print("Minibatch", str(minibatch_count), " sampling...")
			n = np.random.randint(len(validation_set))
			sample_piece = validation_set[n]
			sample_articulation = validation_set_articulation[n]
			if visualize:
				new_voice, new_articulation, probs = model.generate(sample_piece, sample_articulation)
				i = 0
				for expert_model,p in zip(model.expert_models, probs):
					visualize_probs(p, title=expert_model.__class__.__name__ + 'GeneratedOutput' + str(minibatch_count), 
						path=output_dir + expert_model.__class__.__name__ + str(i)+ 'GeneratedOutput' + str(minibatch_count))
					i+=1
				visualize_probs(probs[-2], title= 'ArticulationModelGeneratedOutput' + str(minibatch_count), 
						path=output_dir + 'ArticulationModelGeneratedOutput' + str(minibatch_count))
				visualize_probs(probs[-1], title= 'FinalGeneratedOutput' + str(minibatch_count), 
						path = output_dir + 'FinalGeneratedOutput' + str(minibatch_count))



			else:
				new_voice, new_articulation = model.generate(sample_piece, sample_articulation)
			store_weights(model, output_dir + str(minibatch_count) +'.p')
			output_midi([timesteps_to_notes(new_voice, new_articulation, min_num, timestep_length * PPQ)], output_dir + str(minibatch_count) + '.mid')

		minibatch_count += 1

# store weights from model to a file at path
# just uses pickle dump, so not at all efficient
def store_weights(model, path):
	sys.setrecursionlimit(100000)
	pickle.dump(model, open(path, 'wb'))

# load weights from a path
def load_weights(path):
	print("Loading model from "  + path)
	return pickle.load(open(path, 'rb'))

# shortcut for instantiating some common model structures
# model name indicates which model, see source for exactly what it does
# min_num, max_num, timestep_length are constants relevant to the dataset, load_dataset returns them
# visualize tells the model whether to compile functions for getting probabilities in order to visualize output
# TODO: rewrite this so it works with the new model instantiation process
"""def instantiate_model(model_name, min_num, max_num, timestep_length, visualize):
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
	return model"""

if __name__=='__main__':
	VOICE_TO_PREDICT = 0
	#dataset, min_num, max_num, timestep_length = load_dataset("../Data/train.p", "../Data/validate.p")
	dataset, articulation_data, min_num, max_num, timestep_length = pickle.load(open('../Data/music21_articulation_dataset.p', 'rb'))

	rhythm_encoding_size = int(4//timestep_length) # modified for music21: units are no longer midi timesteps (240 to a quarter note) but quarterLengths (1 to a quarter note)
	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,VOICE_TO_PREDICT]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[0,VOICE_TO_PREDICT])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]

	spacing_models = []
	for i in range(4):
		if i == VOICE_TO_PREDICT: continue
		spacing_models.append(VoiceSpacingExpert(min_num,max_num, [100,200,100], i, VOICE_TO_PREDICT,pieces=pieces, piece=piece, rng=rng))
	spacing_multiexpert = MultiExpert(spacing_models, 4, VOICE_TO_PREDICT, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=False)
	contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], VOICE_TO_PREDICT,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], VOICE_TO_PREDICT, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	#simple_generative = SimpleGenerative(max_num-min_num, [100,200,100], 4,VOICE_TO_PREDICT,
	#	pieces=pieces, prior_timesteps=prior_timesteps, piece=piece, rng=rng)
	model = MultiExpert([spacing_multiexpert, contour_expert, rhythm_expert], 4, VOICE_TO_PREDICT,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=True)
	train(model, 'no_spacing_multiexpert', dataset, articulation_data, min_num, max_num, timestep_length, visualize=True)

	dataset, articulation_data, min_num, max_num, timestep_length = pickle.load(open('../Data/music21_articulation_dataset.p', 'rb'))

	rhythm_encoding_size = int(4//timestep_length) # modified for music21: units are no longer midi timesteps (240 to a quarter note) but quarterLengths (1 to a quarter note)
	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,VOICE_TO_PREDICT]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[0,VOICE_TO_PREDICT])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]

	simple_generative = SimpleGenerative(max_num-min_num, [100,200,100], 4,VOICE_TO_PREDICT,
		pieces=pieces, prior_timesteps=prior_timesteps, piece=piece, rng=rng)
	model2 = MultiExpert([simple_generative], 4, VOICE_TO_PREDICT, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=True)
	train(model2, 'simple_generative', dataset, articulation_data, min_num, max_num, timestep_length, visualize=True)