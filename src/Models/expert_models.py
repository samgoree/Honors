# expert_models.py
# contains several models and helper functions for use in the multi-expert contained within product_of_experts.py

from Models.generative import *
from Models.identity import Identity
from Utilities.visualizer import visualize_multiexpert
from sys import exit

def onehot_to_int(onehot):
	return T.argmax(onehot, axis=onehot.ndim-1)

# this is particularly awful to do (and a bit computationally expensive), avoid if possible
def int_matrix_to_onehot_tensor3(matrix, length):
	a = [T.zeros_like(matrix)] * length
	a = T.stack(a, axis=2)
	def step(matrix_value):
		return T.set_subtensor(T.zeros(length)[matrix_value], 1)
	r, u = theano.map(step, sequences=[matrix.reshape(matrix.shape[0] * matrix.shape[1],ndim=1)])
	return T.stack(r).reshape(a.shape)

# a posteriori way to do it in numpy, still slow until I get better at indexing magic
def int_vector_to_onehot_matrix(vector, length):
	a = np.zeros([len(vector),length])
	for i in range(len(vector)):
		a[i,vector[i]] = 1
	return a

# well now i gotta
# this is the above two functions modified to take int vector
def int_vector_to_onehot_symbolic(vector, length):
	def step(int_value):
		return T.set_subtensor(T.zeros(length)[int_value], 1)
	r,u = theano.map(step, sequences=[vector])
	return r

# roll each vector of the tensor according to the value in the corresponding place in the matrix
def roll_tensor(tensor, matrix):
	tensor_as_matrix = T.reshape(tensor, [tensor.shape[0]*tensor.shape[1], tensor.shape[2]], ndim=2)
	matrix_as_vector = T.reshape(matrix, matrix.shape[0]*matrix.shape[1], ndim=1)
	def step(tensor_vector, matrix_value):
		return T.roll(tensor_vector, matrix_value)
	r,u=theano.map(step, sequences=[tensor_as_matrix,matrix_as_vector])
	return T.stack(r).reshape(tensor.shape)

# cumulative sum where values above or below the max or min are bounded
def bounded_cumsum(vector, min_val, max_val):
	def step(val, total):
		new_total = val + total
		return T.maximum(T.minimum(new_total, max_val), min_val)
	results, updates = theano.scan(step, sequences=vector, outputs_info=[T.cast(T.as_tensor_variable(0), 'int64')])
	return results



class VoiceSpacingExpert(GenerativeLSTM):
	# encoding_size is the length of the third dimension of known voice, it should be the number of pitches
	# network_shape is the neural network shape
	# known_voice_number is the number of the voice we know, same for unknown, these are a priori ints

	# pieces and piece are for use with the product_of_experts model below, pieces is the training minibatch, piece is a piece to generate based on the known voice of
	def __init__(self, min_num, max_num, network_shape, known_voice_number, unknown_voice_number, pieces=None, piece=None, rng=None):
		print("Building a Voice Spacing Expert")

		self.known_voice_number = known_voice_number
		self.unknown_voice_number = unknown_voice_number
		self.encoding_size = encoding_size = max_num-min_num

		#handle missing parameters
		if pieces is None: pieces = T.itensor4()
		if piece is None: piece = T.itensor3()
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()

		 #pieces has four dimensions: pieces, voices, time, pitch

		known_voices = pieces[:,known_voice_number] # first dimension is piece, second is time, third is pitch
		unknown_voices = pieces[:,unknown_voice_number] # same as above

		spacing = T.maximum(T.minimum(onehot_to_int(unknown_voices) - onehot_to_int(known_voices) + encoding_size, encoding_size-1), 0)
		onehot_spacing = int_matrix_to_onehot_tensor3(spacing, encoding_size * 2)



		known_voice = piece[known_voice_number]
		gen_length = known_voice.shape[0] # length of a generated sample - parameter to generation call

		self.generated_probs, generated_spacing, rng_updates = super(VoiceSpacingExpert, self).__init__(None, encoding_size * 2, network_shape, onehot_spacing, None, None, gen_length, rng=rng)
		self.params = self.model.params
		self.layers = self.model.layers

		mask = onehot_spacing
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.params, method='adadelta')

		generated_voice = int_vector_to_onehot_symbolic(onehot_to_int(generated_spacing)+ onehot_to_int(known_voice) - encoding_size, encoding_size)



		self.train_internal = theano.function([pieces], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([pieces], cost, allow_input_downcast=True)
		self.generate = theano.function([piece], generated_voice, updates=rng_updates, allow_input_downcast=True)

	def steprec(self, concurrent_notes, prev_concurrent_notes, known_voices, prev_known_voices, current_beat, prev_note, prev_prev_note, prev_hiddens):
		known_voice = known_voices[self.known_voice_number if self.known_voice_number < self.unknown_voice_number else self.known_voice_number-1]
		prev_known_voice = prev_known_voices[self.known_voice_number if self.known_voice_number < self.unknown_voice_number else self.known_voice_number-1]
		# this one fires on the difference between prev_note and prev_concurrent_notes[known_voice_number]
		prev_spacing = T.set_subtensor(T.zeros(self.encoding_size * 2)[onehot_to_int(prev_note) 
					- onehot_to_int(prev_known_voice) + self.encoding_size], 1)
		model_states = self.model.forward(prev_spacing, prev_hiddens)
		new_states = model_states[:-1]

		min_num_index = self.encoding_size-onehot_to_int(known_voice)
		max_num_index = min_num_index + self.encoding_size
		# changes the generated probs (which here are voice spacing) into absolute pitch by rolling the subtensors corresponding to each timestep in each instance by the value of the reference voice
		final_probs = T.roll(model_states[-1], onehot_to_int(known_voice) - self.encoding_size)[min_num_index:max_num_index]
		return new_states, final_probs

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		for i in pieces:
			start =  0 if len(training_set[i][0]) == minibatch_size else np.random.randint(0, len(training_set[i][0])-(minibatch_size))
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.train_internal(minibatch)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		for i in pieces:
			start = 0
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.validate_internal(minibatch)

class VoiceContourExpert(GenerativeLSTM):

	# encoding size is the number of possible notes, we can't have contour that ascends or descends more than that
	# network_shape is the shape of our neural net
	# voice_number is the voice we are predicting
	# voices, gen_length and first_note are theano variables for the product model below
	# voices is the training minibatch, gen_length is the length of piece to generate and first note is its first note
	def __init__(self, min_num, max_num, network_shape, voice_number, voices=None, gen_length=None, first_note=None, rng=None):
		print("Building a Voice Contour Expert")
		# handle missing parameters
		if voices is None: voices = T.itensor3()
		if gen_length is None: gen_length = T.iscalar()
		if first_note is None: first_note = T.iscalar()
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()

		self.voice_number = voice_number
		self.max_num = max_num
		self.min_num = min_num
		self.encoding_size = encoding_size = max_num-min_num
		
		# voices' first dimension is piece, second is time, third is pitch
		prev_notes = voices[:,:-1]

		# take each note, subtract the value of the previous note
		contour = onehot_to_int(voices[:,1:]) - onehot_to_int(prev_notes)
		#contour = T.concatenate([T.zeros_like(contour[:,0]).dimshuffle(0, 'x'), contour], axis=1) # I think this was wrong
		onehot_contour = int_matrix_to_onehot_tensor3(contour + encoding_size, encoding_size * 2)

		self.generated_probs, generated_contour, rng_updates = super(VoiceContourExpert, self).__init__(None, encoding_size * 2, network_shape, onehot_contour, None, None, gen_length, rng=rng)
		self.params = self.model.params
		self.layers = self.model.layers

		# figure out the cost function
		mask = onehot_contour[:, 1:]
		cost = -T.sum(T.log((self.generated_probs[:,:-1] * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.params, method='adadelta')

		first_note_and_contour = T.concatenate([first_note.dimshuffle('x'), onehot_to_int(generated_contour) - encoding_size])
		#first_note = T.iscalar()
		generated_voice = bounded_cumsum(first_note_and_contour, min_num, max_num)

		self.train_internal = theano.function([voices], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([voices], cost, allow_input_downcast=True)
		self.generate_internal = theano.function([gen_length, first_note], generated_voice, updates=rng_updates, allow_input_downcast=True)

	def steprec(self, concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, prev_prev_note, prev_hiddens):
		# if we are on the first two timesteps, feed in prev_contour of 0
		prev_contour = T.switch(T.gt(T.max(prev_note), 0),
			T.set_subtensor(T.zeros([prev_note.shape[0]*2])[onehot_to_int(prev_note) - onehot_to_int(prev_prev_note) + self.max_num - self.min_num], 1),
			T.set_subtensor(T.zeros([prev_note.shape[0]*2])[self.max_num-self.min_num], 1))

		# this one fires on the difference between the previous two notes
		model_states = self.model.forward(prev_contour, prev_hiddens)
		new_states = model_states[:-1]
		min_num_index = self.encoding_size-onehot_to_int(prev_note)
		max_num_index = min_num_index + self.encoding_size
		final_probs = T.roll(model_states[-1], onehot_to_int(prev_note) - (self.max_num - self.min_num))[min_num_index:max_num_index]
		return new_states, final_probs

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		for i in pieces:
			start =  0 if len(training_set[i][0]) == minibatch_size else np.random.randint(0, len(training_set[i][0])-(minibatch_size))
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.train_internal(minibatch[:,self.voice_number])

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		for i in pieces:
			start = 0
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.validate_internal(minibatch[:,self.voice_number])

	# piece should be a 3d tensor, voice, time, pitch
	def generate(self, piece):
		return self.generate_internal(len(piece[0]), np.argmax(piece[0][0]))

# predicts pitches based on place in the measure
class RhythmExpert(GenerativeLSTM):
	
	# rhythm_encoding_size is the number of timestep-classes
	# pitch_encoding_size is the number of possible pitches
	# network_shape is the neural network shape
	# voice number is the voice we're predicting
	# timestep_info, prior_timestep_pitch_info, pieces and rhythm_info are theano variables for the product model below
	# timestep_info is the one-hot timestep of each note in each piece in the minibatch
	# prior_timestep_pitch_info is the single timestep of pitch data before the start of pieces, third dimension should have length 1
	# pieces is the pitch info that we are trying to predict for each piece in the minibatch
	# rhythm info is for generation, matrix of one-hot encodings of rhythm per timestep
	def __init__(self, rhythm_encoding_size, pitch_encoding_size, network_shape, voice_number, 
		timestep_info=None, prior_timestep_pitch_info=None, pieces=None, rhythm_info=None, rng=None):
		print("Building a Rhythm Expert")
		# handle missing params
		if timestep_info is None: timestep_info = T.itensor3()
		if prior_timestep_pitch_info is None: prior_timestep_pitch_info = T.itensor4()
		if pieces is None: pieces = T.itensor4()
		if rhythm_info is None: rhythm_info = T.imatrix()
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()
		self.voice_number = voice_number
		self.pitch_encoding_size = pitch_encoding_size
		self.rhythm_encoding_size = rhythm_encoding_size

		full_pieces = T.sum(pieces, axis=1)

		prev_notes = T.concatenate([T.sum(prior_timestep_pitch_info, axis=1), full_pieces[:, :-1]], axis=1) # one-hot encoding of pitches for each timestep for each piece

		self.generated_probs, generated_piece, rng_updates = super(RhythmExpert, self).__init__(rhythm_encoding_size, pitch_encoding_size, network_shape, prev_notes, timestep_info, rhythm_info, rng=rng)
		self.params = self.model.params
		self.layers = self.model.layers

		mask = pieces[:,self.voice_number]
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.params, method='adadelta')
		self.train_internal = theano.function([pieces, prior_timestep_pitch_info, timestep_info], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([pieces,prior_timestep_pitch_info, timestep_info], cost, allow_input_downcast=True)
		self.generate_internal = theano.function([rhythm_info], generated_piece, updates=rng_updates, allow_input_downcast=True)

	def steprec(self, concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, prev_prev_note, prev_hiddens):
		model_states= self.model.forward(T.concatenate([prev_note, current_beat]), prev_hiddens)
		return model_states[:-1], model_states[-1]

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start =  0 if len(training_set[i][0]) == minibatch_size else np.random.randint(0, len(training_set[i][0])-(minibatch_size))
			prior_timestep = training_set[i][None,:,None,start-1] if start > 0 else np.zeros_like(training_set[i][None,:,None,start])
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
				timestep_info = int_vector_to_onehot_matrix(np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
				timestep_info = np.append(timestep_info, int_vector_to_onehot_matrix(
					np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:], axis=0)
		return self.train_internal(minibatch, prior_timesteps, timestep_info)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start = 0
			prior_timestep = validation_set[i][None,:,None,start-1] if start > 0 else np.zeros_like(validation_set[i][None,:,None,start])
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
				timestep_info = int_vector_to_onehot_matrix(np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:]
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
				timestep_info = np.append(timestep_info, int_vector_to_onehot_matrix(
					np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:], axis=0)
		return self.validate_internal(minibatch, prior_timesteps, timestep_info)

	def generate(self, piece):
		rhythm_info = int_vector_to_onehot_matrix(np.arange(0, len(piece[0])) % self.rhythm_encoding_size, self.rhythm_encoding_size)
		return self.generate_internal(rhythm_info)