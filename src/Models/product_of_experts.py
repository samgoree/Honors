# product_of_experts.py
# several generative models that I want to take the product of

from Models.generative import *
from Models.identity import Identity
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



class VoiceSpacingExpert(GenerativeLSTM):
	# encoding_size is the length of the third dimension of known voice, it should be the number of pitches
	# network_shape is the neural network shape
	# known_voice_number is the number of the voice we know, same for unknown, these are a priori ints

	# pieces and piece are for use with the product_of_experts model below, pieces is the training minibatch, piece is a piece to generate based on the known voice of
	def __init__(self, encoding_size, network_shape, known_voice_number, unknown_voice_number, pieces=T.itensor4(), piece=T.itensor3(), rng=None):
		print("Building a Voice Spacing Expert")

		 #pieces has four dimensions: pieces, voices, time, pitch

		known_voices = pieces[:,known_voice_number] # first dimension is piece, second is time, third is pitch
		unknown_voices = pieces[:,unknown_voice_number] # same as above

		spacing = onehot_to_int(unknown_voices) - onehot_to_int(known_voices)
		onehot_spacing = int_matrix_to_onehot_tensor3(spacing, encoding_size)



		known_voice = piece[known_voice_number]
		gen_length = known_voice.shape[0] # length of a generated sample - parameter to generation call

		self.generated_probs, generated_spacing, rng_updates = super(VoiceSpacingExpert, self).__init__(None, encoding_size, network_shape, onehot_spacing, None, None, gen_length, rng=rng)

		mask = onehot_spacing
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')

		generated_voice = onehot_to_int(generated_spacing)+ onehot_to_int(known_voice)

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
	def __init__(self, encoding_size, network_shape, voice_number, voices=T.itensor3(), gen_length=T.iscalar(), first_note=T.iscalar(), rng=None):
		print("Building a Voice Contour Expert")
		self.voice_number = voice_number
		
		# voices' first dimension is piece, second is time, third is pitch
		prev_notes = voices[:,1:]

		# take each note, subtract the value of the previous note
		contour = onehot_to_int(voices[:,:-1]) - onehot_to_int(prev_notes)
		contour = T.concatenate([T.zeros_like(contour[:,0]).dimshuffle(0, 'x'), contour], axis=1)
		onehot_contour = int_matrix_to_onehot_tensor3(contour + encoding_size, encoding_size * 2)

		self.generated_probs, generated_contour, rng_updates = super(VoiceContourExpert, self).__init__(None, encoding_size * 2, network_shape, onehot_contour, None, None, gen_length, rng=rng)

		# figure out the cost function
		mask = onehot_contour[:, 1:]
		cost = -T.sum(T.log((self.generated_probs[:,:-1] * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')

		generated_voice = T.extra_ops.cumsum(onehot_to_int(generated_contour) - encoding_size) + first_note

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
	# timestep_info, prior_timestep_pitch_info, pieces and rhythm_info are theano variables for the product model below
	# timestep_info is the one-hot timestep of each note in each piece in the minibatch
	# prior_timestep_pitch_info is the single timestep of pitch data before the start of pieces, third dimension should have length 1
	# pieces is the pitch info that we are trying to predict for each piece in the minibatch
	# rhythm info is for generation, matrix of one-hot encodings of rhythm per timestep
	def __init__(self, rhythm_encoding_size, pitch_encoding_size, network_shape, voice_number, 
		timestep_info=T.itensor3(), prior_timestep_pitch_info = T.itensor4(), pieces=T.itensor4(), rhythm_info=T.imatrix(), rng=None):
		print("Building a Rhythm Expert")
		self.voice_number = voice_number
		self.pitch_encoding_size = pitch_encoding_size
		self.rhythm_encoding_size = rhythm_encoding_size

		full_pieces = T.sum(pieces, axis=1)

		prev_notes = T.concatenate([T.sum(prior_timestep_pitch_info, axis=1), full_pieces[:, :-1]], axis=1) # one-hot encoding of pitches for each timestep for each piece

		self.generated_probs, generated_piece, rng_updates = super(RhythmExpert, self).__init__(rhythm_encoding_size, pitch_encoding_size, network_shape, prev_notes, timestep_info, rhythm_info, rng=rng)

		mask = pieces[:,self.voice_number]
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

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
class MultiExpert:
	
	# expert_types should be a list of strings
	# num_voices is the number of voices
	# voice_to_predict is the voice we want to predict
	# known voice number only matters if we have a spacing expert, it is the 
	# min_num is the lowest midi number to expect
	# max_num is the highest to expect
	# timestep_length is the number of midi timesteps in a single encoded timestep here
	# pieces, prior_timesteps, timestep_info and piece are all for product of experts
	# pieces is a training minibatch, first dimension is pieces, second is voices, third is time, fourth is pitch
	# prior_timesteps is the timestep before the start of each minibatch instance
	# timestep_info is the rhythm information about pieces
	# piece is for generation, a full piece, first dimension is voice, second is time, third is pitch
	def __init__(self, expert_types, num_voices, voice_to_predict,  min_num, max_num, timestep_length,
					pieces=T.itensor4(), prior_timesteps=T.itensor4(), timestep_info=T.itensor3(), piece=T.itensor3(), rng=None):
		print("Building a multi-expert")
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()


		self.rhythm_encoding_size = 240*4//timestep_length
		self.pitch_encoding_size = max_num - min_num

		# do some symbolic variable manipulation so that we can compile a function with updates for all the models
		voices = pieces[:,voice_to_predict]
		gen_length = piece.shape[1]
		first_note = T.argmax(piece[voice_to_predict,0])
		rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=self.rhythm_encoding_size)[0]

		# instantiate all of our experts
		expert_models = []
		expert_probs = []
		hidden_partitions = [0]
		params = []
		for e in expert_types:
			if e == 'SimpleGenerative':
				model = SimpleGenerative(self.pitch_encoding_size, [100,200,100], num_voices, voice_to_predict, pieces, prior_timesteps, piece, rng=rng)
				expert_probs.append(model.generated_probs)
				
			elif e == 'VoiceSpacingExpert':
				model = VoiceSpacingExpert(self.pitch_encoding_size, [100,200,100], 0 if voice_to_predict != 0 else num_voices-1, voice_to_predict, pieces, piece, rng=rng)
				spacing_probs = model.generated_probs
				# changes the generated probs (which here are voice spacing) into absolute pitch by rolling the subtensors corresponding to each timestep in each instance by the value of the reference voice
				expert_probs.append(roll_tensor(spacing_probs, onehot_to_int(pieces[:,0 if voice_to_predict != 0 else num_voices-1])))
			elif e == 'Identity': # don't use this model for anything, it's just for testing -- an identity function
				model = Identity(self.pitch_encoding_size, [100,100], voice_to_predict, pieces, piece, rng=rng)
				expert_probs.append(model.generated_probs)
			elif e == 'VoiceContourExpert':
				model = VoiceContourExpert(self.pitch_encoding_size, [100,200,100], voice_to_predict, voices, gen_length, first_note, rng=rng)
				# we need to roll each subtensor corresponding to a timestep by the value of the previous note in the training data for that instance
				expert_probs.append(roll_tensor(model.generated_probs, onehot_to_int(voices))[:,:,self.pitch_encoding_size//2:(self.pitch_encoding_size*3)//2])

			elif e == 'RhythmExpert':
				model = RhythmExpert(self.rhythm_encoding_size, self.pitch_encoding_size, [100,200,100], voice_to_predict, timestep_info, prior_timesteps, pieces, rhythm_info, rng=rng)
				expert_probs.append(model.generated_probs)
			else:
				print("Unknown Model")
				sys.exit(1)
			hidden_partitions.append(hidden_partitions[-1])
			for l in model.model.layers:
				if hasattr(l, 'initial_hidden_state'):
					hidden_partitions[-1]+=1
			expert_models.append(model)
			params += model.model.params

		# get product weight shared variables
		product_weight = []
		for i in range(len(expert_probs)):
			product_weight.append(theano.shared(1.0)) # all experts are initially weighted the same, it's up to gradient descent to figure it out
		params += product_weight

		# calculate cost for the product symbolically
		un_normalized_product = T.ones_like(expert_probs[0])

		# take the product

		for i,tensor in enumerate(expert_probs):
			power = tensor ** product_weight[i]
			un_normalized_product = un_normalized_product * power
		# for each element of the product (remember, it's a tensor), figure out the normalizing factor
		
		normalizing_factors = T.stack([T.sum(un_normalized_product, axis=2)]*self.pitch_encoding_size, axis=2)

		self.generated_probs = un_normalized_product / normalizing_factors # god i hope this works

		# calculate error

		mask = pieces[:,voice_to_predict]
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, params, method='adadelta')

		# compile training and validation functions for the product model
		self.train_internal = theano.function([pieces, prior_timesteps, timestep_info], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([pieces,prior_timesteps, timestep_info], cost, allow_input_downcast=True)

		# generation is hard

		all_but_one_voice = (T.sum(piece[0:voice_to_predict], axis=0) 
			+ T.sum(piece[voice_to_predict+1:], axis=0)) if voice_to_predict + 1 < num_voices else T.sum(piece[0:voice_to_predict], axis=0)
		known_voice_number = 0 # beware - magic number! Maybe should change later??
		known_voice = piece[:,known_voice_number] # first dimension is time, then pitch
		unknown_voice = piece[:,voice_to_predict] # same as above
		beat_info = int_vector_to_onehot_symbolic(T.arange(0, piece.shape[0]) % self.rhythm_encoding_size, self.rhythm_encoding_size)


		def step(concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, prev_prev_note, *prev_hiddens):
			new_states = []
			probs = []
			#loop through models
			for i,model in enumerate(expert_models):
				e = expert_types[i]
				# fire model, multiply probs onto a running total scaled with product weight
				if e == 'SimpleGenerative': # this model wants sequences over all_but_one_voice and prev_note
					model_states=model.model.forward(T.concatenate([prev_note, concurrent_notes]), prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					probs.append(model_states[-1])
				elif e == 'VoiceSpacingExpert':
					# this one fires on the difference between prev_note and prev_concurrent_notes[known_voice_number]
					prev_spacing = T.set_subtensor(T.zeros_like(prev_note)[onehot_to_int(prev_note) 
								- onehot_to_int(prev_known_voice)], 1)
					model_states = model.model.forward(prev_spacing, prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					probs.append(model_states[-1])
					# changes the generated probs (which here are voice spacing) into absolute pitch by rolling the subtensors corresponding to each timestep in each instance by the value of the reference voice
					probs.append(T.roll(new_states[-1], onehot_to_int(known_voice)))
				elif e == 'VoiceContourExpert':
					# this one fires on the difference between the previous two notes
					prev_contour = T.set_subtensor(T.zeros_like(prev_note)[onehot_to_int(prev_note) - onehot_to_int(prev_prev_note) + self.pitch_encoding_size], 1)
					model_states = model.model.forward(prev_contour, prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					probs.append(T.roll(model_states[-1], onehot_to_int(prev_note) - self.pitch_encoding_size))

				elif e == 'RhythmExpert':
					print(prev_note, current_beat)
					model_states= model.model.forward(T.concatenate([prev_note, current_beat]), prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					probs.append(model_states[-1])
				else:
					print(e + " is not a known expert name, something went horribly wrong")
					exit(1)
			# multiply them together
			product = T.ones_like(probs[0])
			for p in probs:
				product *= p
			total = T.sum(product)
			final_product = product / total

			# sample from dist
			chosen_pitch = rng.choice(size=[1], a=self.pitch_encoding_size, p=final_product)
			current_timestep_onehot = T.cast(int_to_onehot(chosen_pitch, self.pitch_encoding_size), 'int64')
			return [current_timestep_onehot] + new_states

		all_layers = [layer for m in expert_models for layer in m.model.layers]
		results, gen_updates = theano.scan(step, n_steps=all_but_one_voice.shape[0], sequences = [dict(input=T.concatenate([T.zeros([1,all_but_one_voice.shape[1]]),all_but_one_voice], axis=0), taps=[0,-1]), 
									dict(input=T.concatenate([T.zeros([1,known_voice.shape[1]]),known_voice], axis=0), taps=[0,-1]), dict(input=beat_info, taps=[0])],
									outputs_info=[dict(initial=T.zeros([2, self.pitch_encoding_size], dtype='int64'), taps=[-1,-2])] + [dict(initial=layer.initial_hidden_state, taps=[-1])
				for layer in all_layers if hasattr(layer, 'initial_hidden_state')])

		self.generate_internal  = theano.function([piece], results, updates=gen_updates, allow_input_downcast=True)


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
		return self.generate_internal(piece)