# product_of_experts.py
# several generative models that I want to take the product of

from Models.generative import *
from Models.expert_models import *
from Models.identity import Identity
from Utilities.visualizer import visualize_multiexpert
from sys import exit



# Combines the "opinions" of multiple of the above experts
class MultiExpert:
	
	# expert_types should be a list of strings or lists, where a string will be resolved to an expert and a list will be a recursively instantiated multi-expert
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
					pieces=None, prior_timesteps=None, timestep_info=None, piece=None, rng=None, transparent=False):
		print("Building a multi-expert")
		# handle missing parameters
		if pieces is None: pieces = T.itensor4()
		if prior_timesteps is None: prior_timesteps = T.itensor4()
		if timestep_info is None: timestep_info = T.itensor3()
		if piece is None: piece = T.itensor3()
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()
		self.transparent = transparent

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
		self.params = []
		self.layers = []
		for e in expert_types:
			if type(e) is list: #that means we need a sub-multi-expert
				model = MultiExpert(e, num_voices, voice_to_predict, min_num, max_num, timestep_length, pieces,prior_timesteps, timestep_info, piece, rng, transparent)
				expert_probs.append(model.generated_probs)
			elif e == 'SimpleGenerative':
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
				model = VoiceContourExpert(min_num, max_num, [100,200,100], voice_to_predict, voices, gen_length, first_note, rng=rng)
				# we need to roll each subtensor corresponding to a timestep by the value of the previous note in the training data for that instance
				predicted_timesteps = roll_tensor(model.generated_probs, onehot_to_int(voices))[:,:,self.pitch_encoding_size//2:(self.pitch_encoding_size*3)//2]
				# we necessarily cannot make predictions about the first timestep of the training data since it has no contour on its own
				# this just gives equal probability of each note
				first_timestep = T.ones_like(predicted_timesteps[:,0]) / predicted_timesteps.shape[2]
				probs = T.concatenate([first_timestep.dimshuffle(0,'x',1), predicted_timesteps], axis=1)
				expert_probs.append(probs)

			elif e == 'RhythmExpert':
				model = RhythmExpert(self.rhythm_encoding_size, self.pitch_encoding_size, [100,200,100], voice_to_predict, timestep_info, prior_timesteps, pieces, rhythm_info, rng=rng)
				expert_probs.append(model.generated_probs)
			else:
				print("Unknown Model")
				sys.exit(1)
			self.hidden_partitions.append(self.hidden_partitions[-1])
			for l in model.layers:
				self.layers += model.layers
				if hasattr(l, 'initial_hidden_state'):
					self.hidden_partitions[-1]+=1
			expert_models.append(model)
			params += model.params

		# get product weight shared variables
		self.product_weight = []
		for i in range(len(expert_probs)):
			self.product_weight.append(theano.shared(1.0)) # all experts are initially weighted the same, it's up to gradient descent to figure it out
		params += self.product_weight

		# calculate cost for the product symbolically
		un_normalized_product = T.ones_like(expert_probs[0])

		# take the product

		for i,tensor in enumerate(expert_probs):
			power = tensor ** self.product_weight[i]
			un_normalized_product = un_normalized_product * power
		# for each element of the product (remember, it's a tensor), figure out the normalizing factor
		
		normalizing_factors = T.stack([T.sum(un_normalized_product, axis=2)]*self.pitch_encoding_size, axis=2)

		self.generated_probs = un_normalized_product / normalizing_factors

		# calculate error

		mask = pieces[:,voice_to_predict]
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, params, method='adadelta')

		# compile training and validation functions for the product model
		self.train_internal = theano.function([pieces, prior_timesteps, timestep_info], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
		self.validate_internal = theano.function([pieces,prior_timesteps, timestep_info], cost, allow_input_downcast=True, on_unused_input='ignore')

		# if we're being transparent (slightly slower setup), we make some more functions
		if transparent:
			self.internal_probs = []
			for model_name, probs in zip(expert_types, expert_probs):
				if type(model_name) is list: # if the model is a multi-expert, for now we just have the final probs from it TODO: make it transparent recursively
					self.internal_probs.append(("MultiExpert", theano.function([pieces, prior_timesteps, timestep_info], probs, allow_input_downcast=True, on_unused_input='ignore')))
				self.internal_probs.append((model_name, theano.function([pieces,prior_timesteps, timestep_info], probs, allow_input_downcast=True, on_unused_input='ignore')))
			self.internal_final_prob = theano.function([pieces, prior_timesteps, timestep_info], self.generated_probs, allow_input_downcast=True, on_unused_input='ignore')

		# generation is hard

		all_but_one_voice = (T.sum(piece[0:voice_to_predict], axis=0) 
			+ T.sum(piece[voice_to_predict+1:], axis=0)) if voice_to_predict + 1 < num_voices else T.sum(piece[0:voice_to_predict], axis=0)
		known_voice_number = 0 # beware - magic number! Maybe should change later??
		known_voice = piece[known_voice_number] # first dimension is time, then pitch
		unknown_voice = piece[voice_to_predict] # same as above
		beat_info = int_vector_to_onehot_symbolic(T.arange(0, unknown_voice.shape[0]) % self.rhythm_encoding_size, self.rhythm_encoding_size)


		def step(concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, prev_prev_note, *prev_hiddens):
			model_states, output_probs = self.steprec(concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, 
				prev_prev_note, prev_hiddens)

				"""e = expert_types[i]
				# fire model, multiply probs onto a running total scaled with product weight
				if type(e) is list:
					# this means we have a multi-expert, so we need to make step recursive
				elif e == 'SimpleGenerative': # this model wants sequences over all_but_one_voice and prev_note
					
				elif e == 'VoiceSpacingExpert':
					# this one fires on the difference between prev_note and prev_concurrent_notes[known_voice_number]
					prev_spacing = T.set_subtensor(T.zeros_like(prev_note)[onehot_to_int(prev_note) 
								- onehot_to_int(prev_known_voice)], 1)
					model_states = model.model.forward(prev_spacing, prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					# changes the generated probs (which here are voice spacing) into absolute pitch by rolling the subtensors corresponding to each timestep in each instance by the value of the reference voice
					probs.append(T.roll(model_states[-1], onehot_to_int(known_voice)))
				elif e == 'VoiceContourExpert':
					# if we are on the first two timesteps, feed in prev_contour of 0
					prev_contour = T.switch(T.gt(T.max(prev_note), 0),
						T.set_subtensor(T.zeros([prev_note.shape[0]*2])[onehot_to_int(prev_note) - onehot_to_int(prev_prev_note) + self.pitch_encoding_size], 1),
						T.set_subtensor(T.zeros([prev_note.shape[0]*2])[self.pitch_encoding_size], 1))

					# this one fires on the difference between the previous two notes
					model_states = model.model.forward(prev_contour, prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					probs.append(T.roll(model_states[-1], onehot_to_int(prev_note) - self.pitch_encoding_size)[self.pitch_encoding_size//2:(self.pitch_encoding_size*3)//2])

				elif e == 'RhythmExpert':
					model_states= model.model.forward(T.concatenate([prev_note, current_beat]), prev_hiddens[hidden_partitions[i]:hidden_partitions[i+1]])
					new_states += model_states[:-1]
					probs.append(model_states[-1])
				else:
					print(e + " is not a known expert name, something went horribly wrong")
					exit(1)
			# multiply them together
			product = T.ones_like(probs[0])
			for i,p in enumerate(probs):
				product *= (p ** self.product_weight[i])
			total = T.sum(product)
			final_product = product / total"""

			# sample from dist
			chosen_pitch = rng.choice(size=[1], a=self.pitch_encoding_size, p=output_probs)
			current_timestep_onehot = T.cast(int_to_onehot(chosen_pitch, self.pitch_encoding_size), 'int64')
			return [current_timestep_onehot] + new_states

		all_layers = [layer for m in expert_models for layer in m.model.layers]
		results, gen_updates = theano.scan(step, n_steps=all_but_one_voice.shape[0], sequences = [dict(input=T.concatenate([T.zeros([1,all_but_one_voice.shape[1]]),all_but_one_voice], axis=0), taps=[0,-1]), 
									dict(input=T.concatenate([T.zeros([1,known_voice.shape[1]]),known_voice], axis=0), taps=[0,-1]), dict(input=beat_info, taps=[0])],
									outputs_info=[dict(initial=T.zeros([2, self.pitch_encoding_size], dtype='int64'), taps=[-1,-2])] + [dict(initial=layer.initial_hidden_state, taps=[-1])
				for layer in all_layers if hasattr(layer, 'initial_hidden_state')])

		self.generate_internal  = theano.function([piece], results[0], updates=gen_updates, allow_input_downcast=True, on_unused_input='ignore')


	def steprec(self, concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, prev_prev_note, prev_hiddens):
		new_states = []
		probs = []
		#loop through models
		for i,model in enumerate(expert_models):
			model_states, output_probs = model.steprec(concurrent_notes, prev_concurrent_notes, known_voice, prev_known_voice, current_beat, prev_note, 
				prev_prev_note, prev_hiddens[self.hidden_partitions[i]:self.hidden_partitions[i+1]])
			new_states += model_states[:-1]
			probs += output_probs

		product = T.ones_like(probs[0])
		for i,p in enumerate(probs):
			product *= (p ** self.product_weight[i])
		total = T.sum(product)
		final_product = product / total
		return new_states, final_product


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
		print("Validating...")
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start=0
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
		if self.transparent: return self.validate_internal(minibatch, prior_timesteps, timestep_info), minibatch, prior_timesteps, timestep_info
		else: return self.validate_internal(minibatch, prior_timesteps, timestep_info)

	def generate(self, piece):
		return self.generate_internal(piece)