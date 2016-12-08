# generative.py
# simple generative model using theano_lstm by Jonathan Raiman

import sys
import os
sys.path.append('~/Honors/src/')

from Utilities.note import Note
from Utilities.midi_parser_random import output_midi

import fractions

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano_lstm

import numpy as np
np.set_printoptions(threshold=np.inf)

epsilon = 10e-9


class GenerativeLSTM:

	# initial try at this, I'll probably refactor later
	# input is one-hot encodings for voices from this timestep and the previous
	# the neural net should see all voices from the previous timestep and some number from this one and have to predict the next note

	# NEW: encoding_size is the input dimension of the lstm network
	# network_shape is the sizes of each internal layer of the neural network, not including input or output
	# prev_notes is a minibatched list of piece segments, where each piece segment is a list of timesteps and each timestep is an input to the NN
	# curr_notes is similar to prev_notes except it does not include the voice information we're looking to generate, should be the same shape as prev_notes, can be none
	# gen_input is a full piece tensor with one voice removed
	# output_encoding_size is the same is input_encoding_size if left None
	# gen_length is by default the length of gen_input, but if that is None, it should be a specified theano scalar specifying how many timesteps to generate.
	# returns a tensor of the same shape of prev_notes that is the outputted probabilties, a tensor of the same shape as all_but_one voice and the random variable updates for generation
	def __init__(self, input_encoding_size, output_encoding_size, network_shape, prev_notes, curr_notes, gen_input, gen_length=None, rng=None):
		print("Building a generative model")
		
		# handle encoding size modularity
		if output_encoding_size is None:
			input_encoding = input_encoding_size * 2
			output_encoding = input_encoding_size
		elif input_encoding_size is None:
			input_encoding = output_encoding_size
			output_encoding = output_encoding_size
		else:
			input_encoding = input_encoding_size  + output_encoding_size
			output_encoding = output_encoding_size
		


		self.model = theano_lstm.StackedCells(input_encoding, layers=network_shape, activation=T.tanh, celltype=theano_lstm.LSTM)
		self.model.layers[0].in_gate2_activation = lambda x: x
		self.model.layers.append(theano_lstm.Layer(network_shape[-1], output_encoding, lambda x: T.nnet.softmax(x.T).T if x.ndim > 1 else T.nnet.softmax(x)[0]))
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()


		# step function for theano scan - body of the symbolic for loop
		# prev_notes should be a one-hot encoding of the previous timestep
		if curr_notes is not None:
			def step(prev_notes, curr_notes, *prev_hiddens):
				# fire nn on them
				new_states = self.model.forward(T.concatenate([prev_notes, curr_notes], axis=1), prev_hiddens)
				# return new hiddens
				return new_states
			results, updates1 = theano.scan(step, n_steps=prev_notes.shape[1], 
				sequences=[prev_notes.dimshuffle(1,0,2), curr_notes.dimshuffle(1,0,2)],
				outputs_info=[T.extra_ops.repeat(layer.initial_hidden_state.dimshuffle('x',0), prev_notes.shape[0], axis=0)
				for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')] + [None])
		else: # if we only have previous timestep information
			def step(prev_notes, *prev_hiddens):
				# fire nn on them
				new_states = self.model.forward(prev_notes, prev_hiddens)
				# return new hiddens
				return new_states
			results, updates1 = theano.scan(step, n_steps=prev_notes.shape[1], 
				sequences=[prev_notes.dimshuffle(1,0,2)],
				outputs_info=[T.extra_ops.repeat(layer.initial_hidden_state.dimshuffle('x',0), prev_notes.shape[0], axis=0)
				for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')] + [None])
		# results is four dimensions, 0 is layers, 1 is time, 2 is piece, 3 is pitch
		generated_probs = results[-1].dimshuffle(1,0,2)

		# generative pass scan

		if curr_notes is not None:
			gen_length = gen_input.shape[0]
			# curr notes, prev_other_notes are two taps of the same sequence
			# prev_note and prev_hiddens are the recurrent inputs, specified in outputs_info
			def gen_step(prev_notes, prev_output, *prev_hiddens):
				new_states = self.model.forward(T.concatenate([prev_output, prev_notes]), prev_hiddens)
				# complicated part: sample from the distribution in new_states[-1] and return
				chosen_pitch = rng.choice(size=[1], a=output_encoding, p=new_states[-1])
				current_timestep_onehot = T.cast(int_to_onehot(chosen_pitch, output_encoding), 'int64')
				return [current_timestep_onehot] + new_states[:-1]

			gen_results, updates3 = theano.scan(gen_step, n_steps=gen_length,
				sequences=[dict(input=T.concatenate([T.zeros_like(gen_input[0]).dimshuffle('x', 0), gen_input]), taps=[0])],
				outputs_info=[dict(initial=T.cast(T.zeros([output_encoding]), 'int64'), taps=[-1])] + 
				[dict(initial=layer.initial_hidden_state, taps=[-1])
				for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')])
		else:
			def gen_step(prev_note, *prev_hiddens):
				new_states = self.model.forward(prev_note, prev_hiddens)
				# complicated part: sample from the distribution in new_states[-1] and return
				chosen_pitch = rng.choice(size=[1], a=output_encoding, p=new_states[-1])
				current_timestep_onehot = T.cast(int_to_onehot(chosen_pitch, output_encoding), 'int64')
				return [current_timestep_onehot] + new_states[:-1]

			gen_results, updates3 = theano.scan(gen_step, n_steps=gen_length,
				outputs_info=[dict(initial=T.cast(T.zeros([output_encoding]), 'int64'), taps=[-1])] + 
				[dict(initial=layer.initial_hidden_state, taps=[-1])
				for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')])

		return generated_probs, gen_results[0], updates3

	def train(self):
		raise NotImplementedError("Please use a subclass for your specific model!")
	def validate(self):
		raise NotImplementedError("Please use a subclass for your specific model!")
	def generate(self):
		raise NotImplementedError("Please use a subclass for your specific model!")

class SimpleGenerative(GenerativeLSTM):
	# pieces: minibatch of instances, each instance is a list of voices, each voice is a list of timesteps, each timestep is a 1-hot encoding
	# prior_timesteps: the timestep before the start of each piece in pieces, prior_timestep.shape[2] should be 1
	# piece: a full piece to generate the voice_to_predict of
	def __init__(self, encoding_size, network_shape, num_voices, voice_to_predict, pieces=T.itensor4(), prior_timesteps=T.itensor4(), piece=T.itensor3(), rng=None):
		print("Building Simple Generative Model")
		# variables for training

		full_pieces = T.sum(pieces, axis=1) # one-hot encoding of pitches for each timestep for each piece
		curr_notes = (T.sum(pieces[:,0:voice_to_predict], axis=1)
			+ T.sum(pieces[:,voice_to_predict+1:], axis=1)) if voice_to_predict + 1 < num_voices else T.sum(pieces[:,0:voice_to_predict], axis=1)

		# should be three dimensions, pieces, time, pitch
		prev_notes = T.concatenate([T.sum(prior_timesteps, axis=1), full_pieces[:, :-1]], axis=1)
	

		# stuff for generation
		
		full_piece = T.sum(piece, axis=0)
		gen_input = (T.sum(piece[0:voice_to_predict], axis=0) 
			+ T.sum(piece[voice_to_predict+1:], axis=0)) if voice_to_predict + 1 < num_voices else T.sum(piece[0:voice_to_predict], axis=0)

		self.generated_probs, generated_piece, rng_updates = super(SimpleGenerative, self).__init__(encoding_size, None, network_shape, prev_notes, curr_notes,gen_input, rng=rng)

		mask = pieces[:,voice_to_predict]
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')
		self.train_internal = theano.function([pieces, prior_timesteps], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([pieces,prior_timesteps], cost, allow_input_downcast=True)
		self.generate_internal = theano.function([piece], generated_piece, updates=rng_updates, allow_input_downcast=True)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def train(self, pieces, training_set, minibatch_size):
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
		return self.train_internal(minibatch, prior_timesteps)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		for i in pieces:
			start = 0
			prior_timestep = validation_set[i][None,:,None,start-1]
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
		return self.validate_internal(minibatch, prior_timesteps)

	def generate(self, piece):
		return self.generate_internal(piece)

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

