# product_of_experts.py
# several generative models that I want to take the product of

from Models.generative import *
from Models.expert_models import *
from Models.articulation_model import ArticulationModel
from Models.identity import Identity
from Utilities.visualizer import visualize_multiexpert
from sys import exit

# slice each subtensor of tensor from the corresponding starts to the length
def modular_slice(tensor, starts, length):
	assert tensor.ndim - 1 == starts.ndim
	tensor_as_matrix = T.reshape(tensor, [tensor.shape[0]*tensor.shape[1], tensor.shape[2]], ndim=2)
	starts_as_vector = T.reshape(starts, starts.shape[0]*starts.shape[1], ndim=1)
	def step(tensor_vector, start_value, length):
		return tensor_vector[start_value:start_value+length]
	r,u=theano.map(step, sequences=[tensor_as_matrix,starts_as_vector], non_sequences=length)
	return T.reshape(T.stack(r), [tensor.shape[0], tensor.shape[1], length], ndim=3)


# Combines the "opinions" of multiple of the above experts
class MultiExpert:
	
	# sub_experts should be a list of instantiated expert models. Currently, the three models in expert_models.py, the simple generative model in generative.py and this model are supported
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
	def __init__(self, sub_experts, num_voices, voice_to_predict,  min_num, max_num, timestep_length, rhythm_encoding_size,
					pieces=None, prior_timesteps=None, timestep_info=None, piece=None, gen_articulation=None, rng=None, transparent=False):
		print("Building a multi-expert")
		# handle missing parameters
		if pieces is None: pieces = T.itensor4()
		if prior_timesteps is None: prior_timesteps = T.itensor4()
		if timestep_info is None: timestep_info = T.itensor3()
		if piece is None: piece = T.itensor3()
		if gen_articulation is None: gen_articulation = T.itensor3()
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()

		training_mask = T.ones_like(pieces[:,voice_to_predict])

		self.transparent = transparent
		self.voice_to_predict = voice_to_predict

		self.rhythm_encoding_size = rhythm_encoding_size
		self.pitch_encoding_size = max_num - min_num

		# some symbolic manipulation necessary for expert models
		voices = pieces[:,voice_to_predict]
		gen_length = piece.shape[1]
		first_note = T.argmax(piece[voice_to_predict,0])
		rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=self.rhythm_encoding_size)[0]

		# instantiate all of our experts

		self.articulation_model = ArticulationModel(3, rhythm_encoding_size, [100,200,100], voice_to_predict, num_voices, timestep_info=timestep_info, 
			rhythm_info=rhythm_info, piece_articulation=gen_articulation, rng=rng)
		
		expert_probs = []
		self.hidden_partitions = [0]
		self.params = []
		self.layers = []
		self.expert_models = []
		for model in sub_experts:
			if type(model) is MultiExpert:
				expert_probs.append(model.generated_probs)
			elif type(model) is SimpleGenerative:
				expert_probs.append(model.generated_probs)
			elif type(model) is VoiceSpacingExpert:
				spacing_probs = model.generated_probs
				# changes the generated probs (which here are voice spacing) into absolute pitch by rolling the subtensors corresponding to each timestep in each instance by the value of the reference voice
				min_num_index = self.pitch_encoding_size-onehot_to_int(pieces[:,model.known_voice_number])
				expert_probs.append(modular_slice(roll_tensor(spacing_probs, onehot_to_int(pieces[:,model.known_voice_number]) - self.pitch_encoding_size), min_num_index, self.pitch_encoding_size))
			elif type(model) is Identity:
				expert_probs.append(model.generated_probs)
			elif type(model) is VoiceContourExpert:
				min_num_index = self.pitch_encoding_size - onehot_to_int(voices)
				# we need to roll each subtensor corresponding to a timestep by the value of the previous note in the training data for that instance
				predicted_timesteps = modular_slice(roll_tensor(model.generated_probs, onehot_to_int(voices)), min_num_index, self.pitch_encoding_size)
				# we necessarily cannot make predictions about the first timestep of the training data since it has no contour on its own
				# this just gives equal probability of each note
				first_timestep = T.ones_like(predicted_timesteps[:,0]) / predicted_timesteps.shape[2]
				probs = T.concatenate([first_timestep.dimshuffle(0,'x',1), predicted_timesteps], axis=1)
				expert_probs.append(probs)

			elif type(model) is RhythmExpert:
				expert_probs.append(model.generated_probs)
			else:
				print("Unknown Model")
				sys.exit(1)
			self.hidden_partitions.append(self.hidden_partitions[-1])
			self.layers += model.layers
			for l in model.layers:
				if hasattr(l, 'initial_hidden_state'):
					self.hidden_partitions[-1]+=1
			self.expert_models.append(model)
			self.params += model.params

		# get product weight shared variables
		self.product_weight = []
		for i in range(len(expert_probs)):
			self.product_weight.append(theano.shared(1.0)) # all experts are initially weighted the same, it's up to gradient descent to figure it out
		self.params += self.product_weight

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

		mask = pieces[:,voice_to_predict] * training_mask # elementwise product - each value of training_mask should be 1 or 0.
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.params, method='adadelta')

		# compile training and validation functions for the product model
		self.train_internal = theano.function([pieces, prior_timesteps, timestep_info, training_mask], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
		self.validate_internal = theano.function([pieces,prior_timesteps, timestep_info, training_mask], cost, allow_input_downcast=True, on_unused_input='ignore')

		# if we're being transparent (slightly slower setup), we make some more functions
		if transparent:
			self.internal_probs = []
			i=0
			for model, probs in zip(sub_experts, expert_probs):
				if type(model) is MultiExpert: # if the model is a multi-expert, for now we just have the final probs from it TODO: make it transparent recursively
					self.internal_probs.append(("MultiExpert", theano.function([pieces, prior_timesteps, timestep_info], probs, allow_input_downcast=True, on_unused_input='ignore')))
				else:
					self.internal_probs.append((str(type(model)).split("'")[1]+ str(i), theano.function([pieces,prior_timesteps, timestep_info], probs, allow_input_downcast=True, on_unused_input='ignore')))
				i+=1
			self.internal_final_prob = theano.function([pieces, prior_timesteps, timestep_info], self.generated_probs, allow_input_downcast=True, on_unused_input='ignore')

		# generation is hard

		all_but_one_voice = (T.sum(piece[0:voice_to_predict], axis=0) 
			+ T.sum(piece[voice_to_predict+1:], axis=0)) if voice_to_predict + 1 < num_voices else T.sum(piece[0:voice_to_predict], axis=0)
		known_voices = T.concatenate([piece[0:voice_to_predict], piece[voice_to_predict+1:]]) if voice_to_predict + 1 < num_voices else piece[0:voice_to_predict]
		known_articulations = T.concatenate([gen_articulation[0:voice_to_predict], gen_articulation[voice_to_predict+1:]]) if voice_to_predict + 1 < num_voices else gen_articulation[0:voice_to_predict]
		unknown_voice = piece[voice_to_predict] # same as above
		beat_info = int_vector_to_onehot_symbolic(T.arange(0, unknown_voice.shape[0]) % self.rhythm_encoding_size, self.rhythm_encoding_size)


		# if we're being transparent, we end up returning a probability vector for each model (in the same order as sub_experts) for each timestep
		def step(concurrent_notes, prev_concurrent_notes, known_voices, prev_known_voices, current_beat, current_articulations, prev_note, prev_prev_note, prev_articulation, *prev_hiddens):
			if self.transparent:
				model_states, output_probs, internal_probs = self.steprec(concurrent_notes, prev_concurrent_notes, known_voices, prev_known_voices, current_beat, prev_note, 
					prev_prev_note, prev_hiddens[:self.hidden_partitions[-1]])
			else:
				model_states, output_probs = self.steprec(concurrent_notes, prev_concurrent_notes, known_voices, prev_known_voices, current_beat, prev_note, 
					prev_prev_note, prev_hiddens[:self.hidden_partitions[-1]])

			# sample from dist
			chosen_pitch = rng.choice(size=[1], a=self.pitch_encoding_size, p=output_probs)[0]

			#fire articulation model
			articulation_states, articulation_probs = self.articulation_model.steprec(prev_articulation, current_beat, current_articulations, prev_hiddens[self.hidden_partitions[-1]:])
			if self.transparent:
				internal_probs.append(articulation_probs)
				internal_probs.append(output_probs)
			model_states += articulation_states
			articulation = rng.choice(size=[1], a=3, p=articulation_probs)[0]
			rest = T.zeros_like(prev_note)

			# if the articulation is sustain
			current_timestep_onehot = theano.ifelse.ifelse(T.eq(articulation, 0), T.cast(int_to_onehot(chosen_pitch, self.pitch_encoding_size), 'int64'),
				theano.ifelse.ifelse(T.eq(articulation,1), prev_note, rest))
			articulation_onehot = T.cast(int_to_onehot(articulation, 3), 'int64')

			if self.transparent: return [current_timestep_onehot, articulation_onehot] + internal_probs + model_states
			else: return [current_timestep_onehot, articulation_onehot] + model_states

		# I am dimshuffling the known_voices so that we loop over the right axis (time), so the new dimensions are time, voice, pitch
		if self.transparent:
			results, gen_updates = theano.scan(step, n_steps=all_but_one_voice.shape[0], sequences = [
										dict(input=T.concatenate([T.zeros([1,all_but_one_voice.shape[1]]),all_but_one_voice], axis=0), taps=[0,-1]), 
										dict(input=T.concatenate([T.zeros([known_voices.shape[0], 1 ,known_voices.shape[2]]),known_voices], axis=1).dimshuffle(1,0,2), taps=[0,-1]), 
										dict(input=beat_info, taps=[0]),
										dict(input=known_articulations.dimshuffle(1,0,2), taps=[0])],
									outputs_info=[dict(initial=T.zeros([2, self.pitch_encoding_size], dtype='int64'), taps=[-1,-2]), 
										dict(initial=T.zeros([3], dtype='int64'), taps=[-1])] + 
										[None] * (len(sub_experts) + 2) + 
										[dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in self.layers + self.articulation_model.layers if hasattr(layer, 'initial_hidden_state')])
			self.generate_internal = theano.function([piece, gen_articulation], results[0:4+len(sub_experts)], updates=gen_updates, allow_input_downcast=True, on_unused_input='ignore')
		else: 
			results, gen_updates = theano.scan(step, n_steps=all_but_one_voice.shape[0], sequences = [
										dict(input=T.concatenate([T.zeros([1,all_but_one_voice.shape[1]]),all_but_one_voice], axis=0), taps=[0,-1]), 
										dict(input=T.concatenate([T.zeros([known_voices.shape[0], 1 ,known_voices.shape[2]]),known_voices], axis=1).dimshuffle(1,0,2), taps=[0,-1]), 
										dict(input=beat_info, taps=[0]),
										dict(input=known_articulations.dimshuffle(1,0,2), taps=[0])],
									outputs_info=[dict(initial=T.zeros([2, self.pitch_encoding_size], dtype='int64'), taps=[-1,-2]), 
										dict(initial=T.zeros([3], dtype='int64'), taps=[-1])] + 
										[dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in self.layers + self.articulation_model.layers if hasattr(layer, 'initial_hidden_state')])

			self.generate_internal  = theano.function([piece, gen_articulation], (results[0], results[1]), updates=gen_updates, allow_input_downcast=True, on_unused_input='ignore')


	def steprec(self, concurrent_notes, prev_concurrent_notes, known_voices, prev_known_voices, current_beat, prev_note, prev_prev_note, prev_hiddens):
		new_states = []
		probs = []
		#loop through models
		for i,model in enumerate(self.expert_models):
			model_states, output_probs = model.steprec(concurrent_notes, prev_concurrent_notes, known_voices, prev_known_voices, current_beat, prev_note, 
				prev_prev_note, prev_hiddens[self.hidden_partitions[i]:self.hidden_partitions[i+1]])
			new_states += model_states
			probs.append(output_probs)

		product = T.ones_like(probs[0])
		for i,p in enumerate(probs):
			product *= (p ** self.product_weight[i])
		total = T.sum(product)
		final_product = product / total
		if self.transparent: return new_states, final_product, probs
		else: return new_states, final_product

	def train(self, pieces, training_set, articulation_training_set, minibatch_size):
		minibatch = None
		articulation_minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start = 0 if len(training_set[i][0]) == minibatch_size else np.random.randint(0, len(training_set[i][0])-(minibatch_size))
			prior_timestep = training_set[i][None,:,None,start-1] if start > 0 else np.zeros_like(training_set[i][None,:,None,start])
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
				articulation_minibatch = articulation_training_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
				timestep_info = int_vector_to_onehot_matrix(np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
				articulation_minibatch = np.append(articulation_minibatch, articulation_training_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
				timestep_info = np.append(timestep_info, int_vector_to_onehot_matrix(
					np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:], axis=0)
		training_mask = np.zeros_like(minibatch[:,self.voice_to_predict])
		training_mask[articulation_minibatch[:,self.voice_to_predict,:,0] == 1] = 1
		return self.train_internal(minibatch, prior_timesteps, timestep_info, training_mask)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, articulation_validation_set, minibatch_size):
		print("Validating...")
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start=0
			prior_timestep = validation_set[i][None,:,None,start-1] if start > 0 else np.zeros_like(validation_set[i][None,:,None,start])
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
				articulation_minibatch = articulation_validation_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
				timestep_info = int_vector_to_onehot_matrix(np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:]
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
				articulation_minibatch = np.append(articulation_minibatch, articulation_validation_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
				timestep_info = np.append(timestep_info, int_vector_to_onehot_matrix(
					np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:], axis=0)

		training_mask = np.zeros_like(minibatch[:,self.voice_to_predict])
		training_mask[articulation_minibatch[:,self.voice_to_predict,:,0] == 1] = 1
		if self.transparent: return self.validate_internal(minibatch, prior_timesteps, timestep_info, training_mask), minibatch, prior_timesteps, timestep_info
		else: return self.validate_internal(minibatch, prior_timesteps, timestep_info, training_mask)


	def generate(self, piece, articulation):
		if self.transparent:
			outputs = self.generate_internal(piece, articulation)
			return outputs[0], outputs[1], outputs[2:]
		else:
			pitches, articulations = self.generate_internal(piece, articulation)
			return pitches, articulations
