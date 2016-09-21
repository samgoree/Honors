# generative.py
# simple generative model using theano_lstm by Jonathan Raiman

import pickle
import fractions

import theano
import theano.tensor as T
import theano_lstm

# takes the path to a pickle file
# assumes the pickled object is python list of pieces, where each piece is a list of voices and each voice is a list of notes.
def load_dataset(filepath):
	# unpickle the file
	raw_dataset = pickle.load(filepath)
	# find the GCD of note lengths
	gcd = None
	min_num = 2^31
	max_num = -2^31
	for piece in raw_dataset:
		for voice in piece:
			for note in piece:
				length = note.stop_time - note.start_time
				if note.num < min_num: min_num = note.num
				if note.num > max_num: max_num = note.num
				if gcd is None: gcd = length
				else: gcd = fractions.gcd(gcd, length)
	# iterate through the voices and notes, convert each voice to a list of timesteps
	#TODO
	# return the list of lists of lists of timesteps

# initial try at this, I'll probably refactor later
# input is one-hot encodings for voices from this timestep and the previous
# the neural net should see all voices from the previous timestep and some number from this one and have to predict the next note
def build_model(min_num, max_num, num_voices, voice_to_predict):
	model = StackedCells(max_num-min_num * 2, layers=[20, 20], activation=T.tanh, celltype=LSTM)
	model.layers[0].in_gate2_activation = lambda x: x
	model.layers.append(Layer(20, max_num-min_num, lambda x: T.nnet.softmax(x)[0]))

	


	# step function for theano scan - body of the symbolic for loop
	# prev_notes should be a one-hot encoding of the previous timestep
	def step(prev_notes, curr_notes, prev_hiddens):
		# fire nn on them
		new_states = model.forward(T.concatenate([prev_notes, curr_notes]), prev_hiddens)
		# return new hiddens
		return new_states

	results, updates = theano.scan(step, n_steps=)