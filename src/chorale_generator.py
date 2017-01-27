# chorale_generator.py
# scripts the process for generating a chorale from scratch (composition) and from a melody (reharmonization)

from train import *
from scipy.stats import truncnorm

PPQ = 480 # pulses per quarter note -- a midi thing that specifies the length of a timestep

###### four functions to instantiate the models for the first chorale generation scheme ######

# dataset etc. are the same as usual
# soprano controls whether the model expects a soprano voice to exist
# output dir will be created if it is not supplied
# visualize controls whether the training process will be visualized at validation
def instantiate_and_train_bass_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):
	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,0]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[0,0])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]


	# instantiate expert models
	bass_soprano_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 3, 0, pieces=pieces, piece=piece, rng=rng)
	bass_alto_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 2, 0, pieces=pieces, piece=piece, rng=rng)
	bass_tenor_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 1, 0, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([bass_soprano_spacing_expert, bass_alto_spacing_expert, bass_tenor_spacing_expert], 4, 0, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=True)
	bass_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 0,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	bass_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 0, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	models = [spacing_multiexpert, bass_contour_expert, bass_rhythm_expert]
	bass_model = MultiExpert(models, 4, 0,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(bass_model, 'Bass_model', dataset, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return bass_model

def instantiate_and_train_soprano_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):
	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,3]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[3,0])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]


	# instantiate chorale models
	soprano_alto_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 2, 3, pieces=pieces, piece=piece, rng=rng)
	soprano_tenor_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 1, 3, pieces=pieces, piece=piece, rng=rng)
	soprano_bass_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 0, 3, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([soprano_alto_spacing_expert, soprano_tenor_spacing_expert, soprano_bass_spacing_expert], 4, 3, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=True)
	soprano_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 3,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	soprano_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 3, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	soprano_model = MultiExpert([spacing_multiexpert, soprano_contour_expert, soprano_rhythm_expert], 4, 3,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(soprano_model, 'Soprano_model', dataset, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return soprano_model


def instantiate_and_train_alto_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):

	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,2]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[2,0])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]


	# instantiate chorale models
	alto_soprano_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 3, 2, pieces=pieces, piece=piece, rng=rng)
	alto_tenor_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 1, 2, pieces=pieces, piece=piece, rng=rng)
	alto_bass_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 0, 2, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([alto_soprano_spacing_expert, alto_tenor_spacing_expert, alto_bass_spacing_expert], 4, 2, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=True)
	alto_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 2,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	alto_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 2, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	alto_model = MultiExpert([spacing_multiexpert, alto_contour_expert, alto_rhythm_expert], 4, 2,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(alto_model, 'Alto_model', dataset, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return alto_model

def instantiate_and_train_tenor_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):

	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,1]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[1,0])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]


	# instantiate chorale models
	tenor_soprano_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 3, 1, pieces=pieces, piece=piece, rng=rng)
	tenor_alto_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 2, 1, pieces=pieces, piece=piece, rng=rng)
	tenor_bass_spacing_expert = VoiceSpacingExpert(max_num-min_num, [100,200,100], 0, 1, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([tenor_soprano_spacing_expert, tenor_alto_spacing_expert, tenor_bass_spacing_expert], 4, 1, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=True)
	tenor_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 1,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	tenor_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 1, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	tenor_model = MultiExpert([spacing_multiexpert, tenor_contour_expert, tenor_rhythm_expert], 4, 1,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(tenor_model, 'Tenor_model', dataset, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return tenor_model

# empty_voice should be a zero-initialized array for a voice of the correct length and correct note encoding
# rng is a random number generator
# generated voice will have a gaussian distribution with standard deviation 1/4 of the note encoding from the middle note of the encoding
def generate_random_voice(empty_voice, rng):
	mean = empty_voice.shape[1]//2
	stddev = empty_voice.shape[1]//4
	notes = np.floor(truncnorm.rvs(-2, 2, loc=mean, scale=stddev, size=empty_voice.shape[0], random_state=rng)) # this takes truncation bounds in terms of multiples of the standard deviation away from mean of 0
	for i, note in enumerate(notes):
		empty_voice[i][int(note)] = 1



# generate a piece from "scratch" (a random first note and piece length for the bass in the dataset)
# dataset etc. are the same things I pass into everything
# num_to_generate is the number of pieces to output (from different first notes)
# soprano,alto,tenor,bass weights are string filepaths of pickled model files. If they are left as None, new models will be trained from the dataset
# if visualize is true, the training process will be visualized at every validation step
def generate_voice_by_voice(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/generate_voice_by_voice/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/generate_voice_by_voice'): os.mkdir('../Data/Output/generate_voice_by_voice')
	os.mkdir(output_dir)
	# train models
	models = []
	models.append(instantiate_and_train_bass_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if bass_weights is None else load_weights(bass_weights))
	models.append(instantiate_and_train_tenor_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
	models.append(instantiate_and_train_alto_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	models.append(instantiate_and_train_soprano_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece = dataset[np.random.randint(len(dataset))]
		generated_piece = np.zeros_like(sample_piece)
		for i in range(4):
			generate_random_voice(generated_piece[i], rng)
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,4,1))[0])

			generated_piece[v] = models[v].generate(generated_piece)

		output_midi([timesteps_to_notes(voice, min_num, timestep_length * PPQ) for voice in generated_piece], path=output_dir + 'output' + str(n) + '.mid')


def harmonize_melody(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/harmonize_melody/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/harmonize_melody'): os.mkdir('../Data/Output/harmonize_melody')
	os.mkdir(output_dir)
	# train models
	models = []
	models.append(instantiate_and_train_alto_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	models.append(instantiate_and_train_tenor_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
	models.append(instantiate_and_train_bass_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if bass_weights is None else load_weights(bass_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece = dataset[np.random.randint(len(dataset))]
		generated_piece = np.zeros_like(sample_piece)
		generated_piece[3] = sample_piece[3]
		for i in range(3):
			generate_random_voice(generated_piece[i], rng)
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,3,1))[0])

			generated_piece[v] = models[v].generate(generated_piece)



		output_midi([timesteps_to_notes(voice, min_num, timestep_length * PPQ) for voice in generated_piece], path=output_dir + 'output' + str(n) + '.mid')

def harmonize_bass(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/harmonize_bass/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/harmonize_bass'): os.mkdir('../Data/Output/harmonize_bass')
	os.mkdir(output_dir)
	# train models
	models = []
	models.append(instantiate_and_train_soprano_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))
	models.append(instantiate_and_train_alto_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	models.append(instantiate_and_train_tenor_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece = dataset[np.random.randint(len(dataset))]
		generated_piece = np.zeros_like(sample_piece)
		generated_piece[0] = sample_piece[0]
		for i in range(1,4):
			generate_random_voice(generated_piece[i], rng)
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,3,1))[0])

			generated_piece[v+1] = models[v].generate(generated_piece)

		output_midi([timesteps_to_notes(voice, min_num, timestep_length * PPQ) for voice in generated_piece], path=output_dir + 'output' + str(n) + '.mid')

def harmonize_melody_and_bass(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/harmonize_melody_and_bass/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/harmonize_melody_and_bass'): os.mkdir('../Data/Output/harmonize_melody_and_bass')
	os.mkdir(output_dir)
	# train models
	models = []
	models.append(instantiate_and_train_alto_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	models.append(instantiate_and_train_tenor_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece = dataset[np.random.randint(len(dataset))]
		generated_piece = np.zeros_like(sample_piece)
		generated_piece[0] = sample_piece[0]
		generated_piece[3] = sample_piece[3]
		for i in range(1,3):
			generate_random_voice(generated_piece[i], rng)
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,2,1))[0])

			generated_piece[v+1] = models[v].generate(generated_piece)

		output_midi([timesteps_to_notes(voice, min_num, timestep_length * PPQ) for voice in generated_piece], path=output_dir + 'output' + str(i) + '.mid')

# an experiment for my own sanity
# generates a piece feeding in the actual other voices to each model, but assembling the final piece out of just the outputs
def generate_informed(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/generate_voice_by_voice/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/generate_voice_by_voice'): os.mkdir('../Data/Output/generate_voice_by_voice')
	os.mkdir(output_dir)
	# train models
	models = []
	models.append(instantiate_and_train_bass_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if bass_weights is None else load_weights(bass_weights))
	models.append(instantiate_and_train_tenor_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
	models.append(instantiate_and_train_alto_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	models.append(instantiate_and_train_soprano_model(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir, 
		visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece = dataset[np.random.randint(len(dataset))]
		generated_piece = np.zeros_like(sample_piece)
		for i in range(4):
			generated_piece[i] = models[i].generate(sample_piece)

		output_midi([timesteps_to_notes(voice, min_num, timestep_length * PPQ) for voice in generated_piece], path=output_dir + 'output' + str(n) + '.mid')
		output_midi([timesteps_to_notes(voice, min_num, timestep_length * PPQ)  for voice in sample_piece], path=output_dir + 'sample_for_output' + str(n) + '.mid')

# load dataset
if __name__ == '__main__':
	dataset, min_num, max_num, timestep_length = pickle.load(open('../Data/music21.dataset', 'rb'))
	rhythm_encoding_size = int(4//timestep_length)
	generate_informed(dataset, min_num, max_num, timestep_length, rhythm_encoding_size, 10,
		soprano_weights='../Data/Output/generate_voice_by_voice/Thu,26,21:27/Soprano_model160.p',
		alto_weights='../Data/Output/generate_voice_by_voice/Thu,26,21:27/Alto_model180.p',
		tenor_weights='../Data/Output/generate_voice_by_voice/Thu,26,21:27/Tenor_model200.p',
		bass_weights='../Data/Output/generate_voice_by_voice/Thu,26,21:27/Bass_model320.p')