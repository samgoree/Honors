# identity.py
# testing generative model, it should be able to figure out the identity function, right?

from Models.generative import *
from theano.compile.nanguardmode import NanGuardMode

class Identity(GenerativeLSTM):
	def __init__(self, encoding_size, network_shape, voice_to_predict, pieces=T.itensor4(), piece=T.itensor3()):
		print("Building Identity Model")
		# variables for training
		
		one_voice = pieces[:,voice_to_predict]

		# stuff for generation
		gen_length = piece.shape[1] # length of a generated sample - parameter to generation call

		self.generated_probs, generated_voice, rng_updates = super(Identity, self).__init__(encoding_size, network_shape, one_voice, None, None, gen_length)

		mask = pieces[:,voice_to_predict]
		cost = -T.sum(T.log((self.generated_probs * mask).nonzero_values()))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')
		self.train_internal = theano.function([pieces], cost, updates=updates, allow_input_downcast=True, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
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