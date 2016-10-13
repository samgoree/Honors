# generative.py
# simple generative model using theano_lstm by Jonathan Raiman

import sys
import os
sys.path.append('/usr/users/quota/students/18/sgoree/Honors/src/')

from Utilities.note import Note
from Utilities.midi_parser_random import output_midi

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

class GenerativeLSTM:

	# initial try at this, I'll probably refactor later
	# input is one-hot encodings for voices from this timestep and the previous
	# the neural net should see all voices from the previous timestep and some number from this one and have to predict the next note

	# NEW: encoding_size is the input dimension of the lstm network
	# prev_notes is a minibatched list of piece segments, where each piece segment is a list of timesteps and each timestep is an input to the NN
	# curr_notes is similar to prev_notes except it does not include the voice information we're looking to generate, should be the same shape as prev_notes, can be none
	# full_piece is for generation: a tensor the shape of a single full piece
	# all_but_one_voice is full_piece with one voice removed
	# returns a tensor of the same shape of prev_notes that is the outputted probabilties, a tensor of the same shape as full_piece and the random variable updates for generation
	def __init__(self, encoding_size, prev_notes, curr_notes, full_piece, all_but_one_voice):
		print("Building a generative model")
		self.model = theano_lstm.StackedCells(encoding_size * 2 if curr_notes is not None else encoding_size, layers=[100, 200, 100], activation=T.tanh, celltype=theano_lstm.LSTM)
		self.model.layers[0].in_gate2_activation = lambda x: x
		self.model.layers.append(theano_lstm.Layer(100, encoding_size, lambda x: T.nnet.softmax(x.T).T if x.ndim > 1 else T.nnet.softmax(x)[0]))

		rng = theano.tensor.shared_randomstreams.RandomStreams()


		# step function for theano scan - body of the symbolic for loop
		# prev_notes should be a one-hot encoding of the previous timestep
		def step(prev_notes, curr_notes, *prev_hiddens):
			# fire nn on them
			new_states = self.model.forward(T.concatenate([prev_notes, curr_notes], axis=1), prev_hiddens)
			# return new hiddens
			return new_states

		# we want the prev_notes to be all 0 at the first timestep, then full_piece[:-1]
		# we want curr_notes to be the full piece with a voice removed
		results, updates1 = theano.scan(step, n_steps=prev_notes.shape[1], 
			sequences=[prev_notes.dimshuffle(1,0,2), curr_notes.dimshuffle(1,0,2)],
			outputs_info=[T.extra_ops.repeat(layer.initial_hidden_state.dimshuffle('x',0), prev_notes.shape[0], axis=0)
			for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')] + [None])

		# results is four dimensions, 0 is layers, 1 is piece, 2 is time, 3 is pitch
		generated_probs = results[-1]
		
		# calculate cost - negative log liklihood of generating the correct piece


		# training function
		#self.train = theano.function([prev_notes, curr_notes], loss, updates=updates2, allow_input_downcast=True)
		#self.validate = theano.function([prev_notes, curr_notes], loss, allow_input_downcast=True)

		# generative pass scan

		def gen_step(curr_notes, prev_other_notes, prev_note, *prev_hiddens):
			prev_notes = prev_other_notes + prev_note
			new_states = self.model.forward(T.concatenate([prev_notes, curr_notes]), prev_hiddens)
			# complicated part: sample from the distribution in new_states[-1] and return
			chosen_pitch = rng.choice(size=[1], a=encoding_size, p=new_states[-1])
			current_timestep_onehot = T.cast(int_to_onehot(chosen_pitch, encoding_size), 'int64')
			return [current_timestep_onehot] + new_states[:-1]

		gen_results, updates3 = theano.scan(gen_step, n_steps=full_piece.shape[0],
			sequences=[dict(input=T.concatenate([T.zeros_like(full_piece[0]).dimshuffle('x', 0), all_but_one_voice]), taps=[0,-1])],
			outputs_info=[dict(initial=T.cast(T.zeros_like(full_piece[0]), 'int64'), taps=[-1])] + 
			[dict(initial=layer.initial_hidden_state, taps=[-1])
			for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')])

		# generative function
		#self.generate = theano.function([full_piece, all_but_one_voice], gen_results[0], updates=updates3, allow_input_downcast=True)
		return generated_probs, gen_results[0], updates3

	def train(self):
		raise NotImplementedError("Please use a subclass for your specific model!")
	def validate(self):
		raise NotImplementedError("Please use a subclass for your specific model!")
	def generate(self):
		raise NotImplementedError("Please use a subclass for your specific model!")

class SimpleGenerative(GenerativeLSTM):
	def __init__(self, encoding_size, num_voices, voice_to_predict):
		print("Building Simple Generative Model")
		# variables for training
		pieces = T.itensor4() # minibatch of instances, each instance is a list of voices, each voice is a list of timesteps, each timestep is a 1-hot encoding
		prior_timesteps = T.itensor4() # the timestep before the start of each piece in pieces, prior_timestep.shape[2] should be 1

		full_pieces = T.sum(pieces, axis=1) # one-hot encoding of pitches for each timestep for each piece
		all_but_one_voice = (T.sum(pieces[:,0:voice_to_predict], axis=1)
			+ T.sum(pieces[:,voice_to_predict+1:], axis=1)) if voice_to_predict + 1 < num_voices else T.sum(pieces[:,0:voice_to_predict], axis=1)

		# should be three dimensions, pieces, time, pitch
		prev_notes = T.concatenate([T.sum(prior_timesteps, axis=1), full_pieces[:, :-1]], axis=1)
		curr_notes = all_but_one_voice
	

		# stuff for generation
		piece = T.itensor3() # a full piece
		full_piece = T.sum(piece, axis=0)
		all_but_one_voice = (T.sum(piece[0:voice_to_predict], axis=0) 
			+ T.sum(piece[voice_to_predict+1:], axis=0)) if voice_to_predict + 1 < num_voices else T.sum(piece[0:voice_to_predict], axis=0)

		generated_probs, generated_piece, rng_updates = super(SimpleGenerative, self).__init__(encoding_size, prev_notes, curr_notes, full_piece,all_but_one_voice)

		cost = -T.sum(T.log(generated_probs[pieces[:,voice_to_predict]==1]))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')
		self.train = theano.function([pieces, prior_timesteps], cost, updates=updates, allow_input_downcast=True)
		self.validate = theano.function([pieces,prior_timesteps], cost, allow_input_downcast=True)
		self.generate = theano.function([piece], generated_piece, updates=rng_updates, allow_input_downcast=True)

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
	dataset, min_num, max_num, timestep_length = load_dataset("/usr/users/quota/students/18/sgoree/Honors/Data/train.p", "/usr/users/quota/students/18/sgoree/Honors/Data/validate.p")
	model = SimpleGenerative(max_num - min_num, len(dataset[0]), 3)
	output_dir = '/usr/users/quota/students/18/sgoree/Honors/Data/Output/generative/' + strftime("%a,%d,%H:%M", localtime())+ '/'
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
	train()