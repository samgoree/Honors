# generative.py
# simple generative model using theano_lstm by Jonathan Raiman

import sys
sys.path.append('/usr/users/quota/students/18/sgoree/Honors/src/')

from Utilities.note import Note
from Utilities.midi_parser_random import output_midi

import pickle
import fractions
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano_lstm

import numpy as np
np.set_printoptions(threshold=np.inf)

# takes the path to a pickle file
# assumes the pickled object is python list of pieces, where each piece is a list of voices and each voice is a list of notes.
def load_dataset(filepath):
	print("Loading", filepath)
	# unpickle the file
	raw_dataset = pickle.load(open(filepath, 'rb'))
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


# initial try at this, I'll probably refactor later
# input is one-hot encodings for voices from this timestep and the previous
# the neural net should see all voices from the previous timestep and some number from this one and have to predict the next note
def build_model(min_num, max_num, num_voices, voice_to_predict):
	print("Building generative model")
	model = theano_lstm.StackedCells((max_num-min_num) * 2, layers=[20, 20], activation=T.tanh, celltype=theano_lstm.LSTM)
	model.layers[0].in_gate2_activation = lambda x: x
	model.layers.append(theano_lstm.Layer(20, max_num-min_num, lambda x: T.nnet.softmax(x)[0]))

	rng = theano.tensor.shared_randomstreams.RandomStreams()

	piece = T.itensor3() # list of voices, each voice is a list of timesteps, each timestep is a 1-hot encoding
	full_piece = T.sum(piece, axis=0) # one-hot encoding of pitches for each timestep
	all_but_one_voice = (T.sum(piece[0:voice_to_predict], axis=0) 
		+ T.sum(piece[voice_to_predict+1:], axis=0)) if voice_to_predict + 1 < num_voices else T.sum(piece[0:voice_to_predict], axis=0)

	# step function for theano scan - body of the symbolic for loop
	# prev_notes should be a one-hot encoding of the previous timestep
	def step(prev_notes, curr_notes, *prev_hiddens):
		# fire nn on them
		new_states = model.forward(T.concatenate([prev_notes, curr_notes]), prev_hiddens)
		# return new hiddens
		return new_states

	prev_notes = T.concatenate([T.zeros_like(full_piece[0]).dimshuffle('x', 0), full_piece[:-1]])

	# we want the prev_notes to be all 0 at the first timestep, then full_piece[:-1]
	# we want curr_notes to be the full piece with a voice removed
	results, updates1 = theano.scan(step, n_steps=piece.shape[1], 
		sequences=[prev_notes, all_but_one_voice],
		outputs_info=[dict(initial=layer.initial_hidden_state, taps=[-1]) 
		for layer in model.layers if hasattr(layer, 'initial_hidden_state')] + [None])

	# results is three dimensions, 0 is layers, 1 is time, 2 is pitch
	generated_probs = results[-1]
	
	# calculate cost - negative log liklihood of generating the correct piece
	cost = -T.sum(T.log(generated_probs[piece[voice_to_predict]==1]))

	# create optimization updates using adadelta
	updates2, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, model.params, method='adadelta')


	# training function
	training = theano.function([piece], [cost], updates=updates2, allow_input_downcast=True)

	# generative pass scan

	def gen_step(curr_notes, prev_other_notes, prev_note, *prev_hiddens):
		prev_notes = prev_other_notes + prev_note
		new_states = model.forward(T.concatenate([prev_notes, curr_notes]), prev_hiddens)
		# complicated part: sample from the distribution in new_states[-1] and return
		chosen_pitch = rng.choice(size=[1], a=max_num-min_num, p=new_states[-1])
		current_timestep_onehot = T.cast(int_to_onehot(chosen_pitch, max_num-min_num), 'int64')
		return [current_timestep_onehot] + new_states[:-1]

	gen_results, updates3 = theano.scan(gen_step, n_steps=piece.shape[1],
		sequences=[dict(input=T.concatenate([T.zeros_like(full_piece[0]).dimshuffle('x', 0), all_but_one_voice]), taps=[0,-1])],
		outputs_info=[dict(initial=T.cast(T.zeros_like(full_piece[0]), 'int64'), taps=[-1])] + 
		[dict(initial=layer.initial_hidden_state, taps=[-1])
		for layer in model.layers if hasattr(layer, 'initial_hidden_state')])

	# generative function
	generating = theano.function([piece], [gen_results[0]], updates=updates3, allow_input_downcast=True)

	return training, generating

# theano symbolic way to convert an int to a one-hot encoding
def int_to_onehot(n, len):
	a = T.zeros([len])
	if n == -1: return a
	a = T.set_subtensor(a[n], 1)
	return a

def onehot_matrix_to_int_vector(onehot):
	output = []
	for val in onehot:
		for i in range(len(val)):
			if val[i] == 1: 
				output.append(i)
				break
		output.append(-1)
	return np.array(output)

def train():
	dataset, min_num, max_num, timestep_length = load_dataset("/usr/users/quota/students/18/sgoree/Honors/Data/train.p")
	train, generate = build_model(min_num, max_num, len(dataset[0]), 3)
	print("Training...")
	# main training loop
	for epoch in range(1, 250):
		cost = 0.
		for piece in dataset:
			cost += train(piece)[0]
		if epoch% 20 == 0:
			print("epoch: ", epoch, "cost: ", str(cost/len(dataset)))
			sample_piece = dataset[np.random.randint(0,len(dataset))]
			for voice in sample_piece:
				print(onehot_matrix_to_int_vector(voice))
			new_voice = generate(sample_piece)[0]
			output_midi([timesteps_to_notes(new_voice, min_num, timestep_length)], '/usr/users/quota/students/18/sgoree/Honors/Data/Output/generative/epoch' + str(epoch) + '.mid')

train()