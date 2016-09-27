# generative.py
# simple generative model using theano_lstm by Jonathan Raiman

import Utilities.note

import pickle
import fractions

import theano
import theano.tensor as T
import theano_lstm

# takes the path to a pickle file
# assumes the pickled object is python list of pieces, where each piece is a list of voices and each voice is a list of notes.
def load_dataset(filepath):
	# unpickle the file
	raw_dataset = pickle.load(open(filepath, 'rb'))
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
	# return the list of lists of lists of timesteps, the min num and max num

# initial try at this, I'll probably refactor later
# input is one-hot encodings for voices from this timestep and the previous
# the neural net should see all voices from the previous timestep and some number from this one and have to predict the next note
def build_model(min_num, max_num, num_voices, voice_to_predict):
	model = StackedCells(max_num-min_num * 2, layers=[20, 20], activation=T.tanh, celltype=LSTM)
	model.layers[0].in_gate2_activation = lambda x: x
	model.layers.append(Layer(20, max_num-min_num, lambda x: T.nnet.softmax(x)[0]))

	piece = T.itensor3() # list of voices, each voice is a list of timesteps, each timestep is a 1-hot encoding
	full_piece = T.sum(piece, axis=0) # one-hot encoding of pitches for each timestep
	all_but_one_voice = T.sum(piece[0:voice_to_predict], axis=0) + T.sum(piece[voice_to_predict+1:], axis=0)

	# step function for theano scan - body of the symbolic for loop
	# prev_notes should be a one-hot encoding of the previous timestep
	def step(prev_notes, curr_notes, *prev_hiddens):
		# fire nn on them
		new_states = model.forward(T.concatenate([prev_notes, curr_notes]), prev_hiddens)
		# return new hiddens
		return new_states

	prev_notes = T.concatenate([T.zeros_like(full_piece[0]), full_piece[:-1]])

	# we want the prev_notes to be all 0 at the first timestep, then full_piece[:-1]
	# we want curr_notes to be the full piece with a voice removed
	results, updates1 = theano.scan(step, n_steps=piece.shape[1], 
		sequences=[prev_notes, all_but_one_voice], 
		outputs_info=[dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')])

	# results is three dimensions, 0 is time, 1 is layers, 2 is pitch
	generated_probs = results[:,-1,:]
	
	# calculate cost - negative log liklihood of generating the correct piece
	cost = -T.sum(T.log(generated_probs[piece[voice_to_predict]==1]))

	# create optimization updates using adadelta
	updates2, _,_,_ = create_optimization_updates(cost, model.params, method='adadelta')


	# training function
	training = theano.function([piece], [cost], updates={**updates1, **updates2}, allow_input_downcast=True)

	# generative pass scan

	# generative function

	return training

def train():
	dataset, min_num, max_num = load_dataset("/usr/users/quota/students/18/sgoree/Honors/Data/train.p")
	train = build_model(min_num, max_num, len(dataset), 1)
	print(train(dataset))

train()