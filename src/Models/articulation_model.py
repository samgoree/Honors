# articulation_model.py
# uses generativeLSTM to model articulate/sustain/rest data

from Models.generative import *
from Models.expert_models import int_vector_to_onehot_matrix

SUSTAIN_DROPOUT = 0.5 # across the dataset, removing 68% of the sustain timesteps will, on average, make the same number of sustains as articulates

class ArticulationModel(GenerativeLSTM):
	
	# training mask is a tensor the same shape as articulation data that signals which values' prediction accuracy should be trained on. I multipy the articulation_data by those values before calculating loss
	# prev_articulation is the previous timestep's (before the start of each piece in the minibatch) articulation
	# articulation_data is the articulations for each timestep in the minibatch
	# timestep_info is the timestep encoding data
	def __init__(self, articulation_encoding_size, rhythm_encoding_size, network_shape, voice_number, prev_articulation=None, articulation_data=None, timestep_info=None, rhythm_info=None, rng=None):
		if articulation_data is None:
			articulation_data = T.itensor3()
		if prev_articulation is None:
			prev_articulation = T.tensor3()
		if timestep_info is None:
			timestep_info = T.itensor3()
		if rhythm_info is None:
			rhythm_info = T.imatrix()
		training_mask_A = T.ones_like(articulation_data)
		training_mask_B = T.ones_like(articulation_data)
		training_mask_C = T.ones_like(articulation_data)
		if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()

		prev_articulations = T.concatenate([prev_articulation, articulation_data[:,:-1]], axis=1)

		self.generated_probs, generated_articulation, rng_updates = super(ArticulationModel, self).__init__(rhythm_encoding_size, articulation_encoding_size, 
			network_shape, prev_articulations, timestep_info, rhythm_info, rng=rng)

		self.params = self.model.params
		self.layers = self.model.layers
		self.voice_number = voice_number
		self.articulation_encoding_size = articulation_encoding_size
		self.rhythm_encoding_size = rhythm_encoding_size

		maskA = articulation_data * training_mask_A # elementwise product - each value of training_mask should be 1 or 0.
		maskB = articulation_data * training_mask_B
		maskC = articulation_data * training_mask_C
		costA = -T.sum(T.log((self.generated_probs * maskA).nonzero_values()))
		costB = -T.sum(T.log((self.generated_probs * maskB).nonzero_values()))
		costC = -T.sum(T.log((self.generated_probs * maskC).nonzero_values()))
		cost = theano.printing.Print('costA:')(costA) + theano.printing.Print('costB:')(costB) + theano.printing.Print('costC:')(costC)

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.params, method='adadelta')

		self.train_internal = theano.function([articulation_data, prev_articulation, timestep_info, training_mask_A, training_mask_B, training_mask_C], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([articulation_data, prev_articulation, timestep_info], cost, allow_input_downcast=True)
		self.generate_internal = theano.function([rhythm_info], generated_articulation, updates=rng_updates, allow_input_downcast=True)

	def steprec(self, prev_articulation, current_beat, prev_hiddens):
		model_states= self.model.forward(T.concatenate([prev_articulation, current_beat]), prev_hiddens)
		return model_states[:-1], model_states[-1]

	def train(self, pieces, training_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start = np.random.randint(0, len(training_set[i][0])-(minibatch_size+1))
			prior_timestep = training_set[i][None,self.voice_number,None,start-1] if start > 0 else np.zeros_like(training_set[i][None,self.voice_number,None,start])
			if minibatch is None:
				minibatch = training_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
				timestep_info = int_vector_to_onehot_matrix(np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:]
			else:
				minibatch = np.append(minibatch, training_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
				timestep_info = np.append(timestep_info, int_vector_to_onehot_matrix(
					np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:], axis=0)
		training_maskA = np.zeros_like(minibatch[:,self.voice_number])
		training_maskA[minibatch[:,self.voice_number,:,0] == 1] = 1
		training_maskB = np.zeros_like(minibatch[:,self.voice_number])
		training_maskB[minibatch[:,self.voice_number,:,1] == 1] = np.random.choice(2, training_maskB[minibatch[:,self.voice_number,:,1] == 1].shape, p=[SUSTAIN_DROPOUT, 1-SUSTAIN_DROPOUT])
		training_maskC = np.zeros_like(minibatch[:,self.voice_number])
		training_maskC[minibatch[:,self.voice_number,:,2] == 1] = 1

		return self.train_internal(minibatch[:,self.voice_number], prior_timesteps, timestep_info, training_maskA, training_maskB, training_maskC)

	# pieces is an array of ints corresponding to indicies of the pieces selected from training_set
	def validate(self, pieces, validation_set, minibatch_size):
		minibatch = None
		prior_timesteps = None
		timestep_info = None
		for i in pieces:
			start = 0
			prior_timestep = np.zeros_like(validation_set[i][None,self.voice_number,None,start])
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
				prior_timesteps = prior_timestep
				timestep_info = int_vector_to_onehot_matrix(np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:]
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
				prior_timesteps = np.append(prior_timesteps, prior_timestep, axis=0)
				timestep_info = np.append(timestep_info, int_vector_to_onehot_matrix(
					np.arange(start, start+minibatch_size) % self.rhythm_encoding_size, self.rhythm_encoding_size)[None,:], axis=0)
		return self.validate_internal(minibatch[:, self.voice_number], prior_timesteps, timestep_info)

	def generate(self, piece):
		rhythm_info = int_vector_to_onehot_matrix(np.arange(0, len(piece[0])) % self.rhythm_encoding_size, self.rhythm_encoding_size)
		return self.generate_internal(rhythm_info)