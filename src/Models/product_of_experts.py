# product_of_experts.py
# several generative models that I want to take the product of

from Models.generative import *

def onehot_to_int_matrix(onehot):
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

class VoiceSpacingExpert(GenerativeLSTM):
	# encoding_size is the length of the third dimension of known voice, it should be the number of pitches
	# network_shape is the neural network shape
	# known_voice_number is the number of the voice we know, same for unknown, these are a priori ints

	# pieces and piece are for use with the product_of_experts model below, pieces is the training minibatch, piece is a piece to generate based on the known voice of
	def __init__(self, encoding_size, network_shape, known_voice_number, unknown_voice_number, pieces=T.itensor4(), piece=T.itensor3()):
		print("Building a Voice Spacing Expert")

		 #pieces has four dimensions: pieces, voices, time, pitch

		known_voices = pieces[:,known_voice_number] # first dimension is piece, second is time, third is pitch
		unknown_voices = pieces[:,unknown_voice_number] # same as above

		spacing = onehot_to_int_matrix(unknown_voices) - onehot_to_int_matrix(known_voices)
		onehot_spacing = int_matrix_to_onehot_tensor3(spacing, encoding_size)



		known_voice = piece[known_voice_number]
		gen_length = known_voice.shape[0] # length of a generated sample - parameter to generation call

		self.generated_probs, generated_spacing, rng_updates = super(VoiceSpacingExpert, self).__init__(None, encoding_size, network_shape, onehot_spacing, None, None, gen_length)

		mask = onehot_spacing
		cost = -T.sum(T.log((generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')

		generated_voice = onehot_to_int_matrix(generated_spacing)+ onehot_to_int_matrix(known_voice)

		self.train_internal = theano.function([pieces], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([pieces], cost, allow_input_downcast=True)
		self.generate = theano.function([piece], generated_voice, updates=rng_updates, allow_input_downcast=True)

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		for i in pieces:
			start = np.random.randint(0, len(training_set[i][0])-(minibatch_size+1))
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.train_internal(minibatch)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		for i in pieces:
			start = np.random.randint(0, len(validation_set[i][0])-(minibatch_size+1))
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
	def __init__(self, encoding_size, network_shape, voice_number, voices=T.itensor3(), gen_length=T.iscalar(), first_note=T.iscalar()):
		print("Building a Voice Contour Expert")
		self.voice_number = voice_number
		
		# voices' first dimension is piece, second is time, third is pitch
		prev_notes = voices[:,1:]

		# take each note, subtract the value of the previous note
		contour = onehot_to_int_matrix(voices[:,:-1]) - onehot_to_int_matrix(prev_notes)
		contour = T.concatenate([T.zeros_like(contour[:,0]).dimshuffle(0, 'x'), contour], axis=1)
		onehot_contour = int_matrix_to_onehot_tensor3(contour + encoding_size, encoding_size * 2)

		generated_probs, generated_contour, rng_updates = super(VoiceContourExpert, self).__init__(None, encoding_size * 2, network_shape, onehot_contour, None, None, gen_length)

		# figure out the cost function
		mask = onehot_contour[:, 1:]
		cost = -T.sum(T.log((generated_probs[:,:-1] * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')

		generated_voice = T.extra_ops.cumsum(onehot_to_int_matrix(generated_contour) - encoding_size) + first_note

		self.train_internal = theano.function([voices], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([voices], cost, allow_input_downcast=True)
		self.generate_internal = theano.function([gen_length, first_note], generated_voice, updates=rng_updates, allow_input_downcast=True)

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		for i in pieces:
			start = np.random.randint(0, len(training_set[i][0])-(minibatch_size+1))
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.train_internal(minibatch[:,self.voice_number])

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		for i in pieces:
			start = np.random.randint(0, len(validation_set[i][0])-(minibatch_size+1))
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
	# timestep_info, prior_timestep_pitch_info
	def __init__(self, rhythm_encoding_size, pitch_encoding_size, network_shape, voice_number, 
		timestep_info=T.itensor3(), prior_timestep_pitch_info = T.itensor4(), pieces=T.itensor4(), rhythm_info=T.imatrix()):
		print("Building a Rhythm Expert")
		self.voice_number = voice_number
		self.pitch_encoding_size = pitch_encoding_size
		self.rhythm_encoding_size = rhythm_encoding_size
		# variables for training
		timestep_info = T.itensor3() # minibatch of instances, each instance is a list of timesteps, each timestep is a 1-hot encoding with timestep
		prior_timestep_pitch_info = T.itensor4() # the pitch timesteps immediately preceeding the pieces, third dimension should have length 1
		pieces = T.itensor4() # pitch info, just shifted forward a timestep
		full_pieces = T.sum(pieces, axis=1)

		prev_notes = T.concatenate([T.sum(prior_timestep_pitch_info, axis=1), full_pieces[:, :-1]], axis=1) # one-hot encoding of pitches for each timestep for each piece

		# stuff for generation
		rhythm_info = T.imatrix() # time, rhythm info

		generated_probs, generated_piece, rng_updates = super(RhythmExpert, self).__init__(rhythm_encoding_size, pitch_encoding_size, network_shape, prev_notes, timestep_info, rhythm_info)

		mask = pieces[:,self.voice_number]
		cost = -T.sum(T.log((generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')
		self.train_internal = theano.function([pieces, prior_timestep_pitch_info, timestep_info], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([pieces,prior_timestep_pitch_info, timestep_info], cost, allow_input_downcast=True)
		self.generate_internal = theano.function([rhythm_info], generated_piece, updates=rng_updates, allow_input_downcast=True)

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start = np.random.randint(0, len(training_set[i][0])-(minibatch_size+1))
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
			start = np.random.randint(0, len(validation_set[i][0])-(minibatch_size+1))
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

# Combines the "opinions" of multiple of the above experts
class Multi_expert:
	def __init__(experts):
		# check that the experts are all subclasses of GenerativeLSTM
		for e in experts:
			assert isinstance(e, GenerativeLSTM)
		self.experts = experts



	def train(self, pieces, training_set, minibatch_size):
		# figure out the training minibatch


