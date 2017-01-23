# articulation_model.py
# uses generativeLSTM to model articulate/sustain/rest data

from Models.generative import *

class ArticulationModel(GenerativeLSTM):
	
	def __init__(self, encoding_size, network_shape, articulation_data=None, gen_length=None, rng=None):
		if articulation_data is None:
			articulation_data = T.itensor4()
		if gen_length is None:
			gen_length = T.iscalar()
			if rng is None: rng = theano.tensor.shared_randomstreams.RandomStreams()
		self.generated_probs, generated_articulation, rng_updates = super(VoiceSpacingExpert, self).__init__(None, encoding_size, network_shape, articulation_data, None, None, gen_length, rng=rng)

		self.params = self.model.params
		self.layers = self.model.layers

		mask = articulation_data
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.params, method='adadelta')

		self.train_internal = theano.function([articulation_data], cost, updates=updates, allow_input_downcast=True)
		self.validate_internal = theano.function([articulation_data], cost, allow_input_downcast=True)
		self.generate = theano.function([gen_length], generated_articulation, updates=rng_updates, allow_input_downcast=True)

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
			start = 0
			if minibatch is None:
				minibatch = validation_set[i][None,:,start:start+minibatch_size]
			else:
				minibatch = np.append(minibatch, validation_set[i][None,:,start:start+minibatch_size], axis=0)
		return self.validate_internal(minibatch)