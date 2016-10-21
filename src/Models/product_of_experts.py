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

class VoiceSpacingExpert(GenerativeLSTM):
	# encoding_size is the length of the third dimension of known voice, it should be the number of pitches
	# network_shape is the neural network shape
	def __init__(self, encoding_size, network_shape):
		print("Building Voice Spacing Expert")
		known_voice = T.itensor3() # first dimension is piece, second is time, third is pitch
		unknown_voice = T.itensor3() # these should be the same shape - testing a posteriori shapes is hard, so it's not going to happen here

		spacing = onehot_to_int_matrix(unknown_voice) - onehot_to_int_matrix(known_voice)
		onehot_spacing = int_matrix_to_onehot_tensor3(spacing, encoding_size)

		gen_length = T.iscalar() # length of a generated sample - parameter to generation call

		generated_probs, generated_spacing, rng_updates = super(VoiceSpacingExpert, self).__init__(encoding_size, network_shape, onehot_spacing, None, None, gen_length)

		cost = -T.sum(T.log(generated_probs[onehot_spacing==1]))

		updates, gsums, xsums, lr, max_norm  = theano_lstm.create_optimization_updates(cost, self.model.params, method='adadelta')

		generated_voice = onehot_to_int_matrix(generated_spacing)+ onehot_to_int_matrix(known_voice)

		self.train = theano.function([known_voice, unknown_voice], cost, updates=updates, allow_input_downcast=True)
		self.validate = theano.function([known_voice,unknown_voice], cost, allow_input_downcast=True)
		self.generate = theano.function([known_voice, gen_length], generated_voice, updates=rng_updates, allow_input_downcast=True)