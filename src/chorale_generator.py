# chorale_generator.py
# scripts the process for generating a chorale from scratch (composition) and from a melody (reharmonization)

from train import *
from Models.articulation_model import ArticulationModel
from scipy.stats import truncnorm

PPQ = 480 # pulses per quarter note -- a midi thing that specifies the length of a timestep

###### four functions to instantiate the models for the first chorale generation scheme ######

# dataset etc. are the same as usual
# soprano controls whether the model expects a soprano voice to exist
# output dir will be created if it is not supplied
# visualize controls whether the training process will be visualized at validation
def instantiate_and_train_bass_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):
	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,0]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[3,0])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]


	# instantiate expert models
	bass_soprano_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 0, 3, pieces=pieces, piece=piece, rng=rng)
	bass_alto_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 1, 3, pieces=pieces, piece=piece, rng=rng)
	bass_tenor_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 2, 3, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([bass_soprano_spacing_expert, bass_alto_spacing_expert, bass_tenor_spacing_expert], 4, 3, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=False)
	bass_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 3,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	bass_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 3, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	models = [spacing_multiexpert, bass_contour_expert, bass_rhythm_expert]
	bass_model = MultiExpert(models, 4, 3,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(bass_model, 'Bass_model', dataset, articulation_data, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)

	return bass_model

def instantiate_and_train_soprano_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):
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


	# instantiate chorale models
	soprano_alto_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 1, 0, pieces=pieces, piece=piece, rng=rng)
	soprano_tenor_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 2, 0, pieces=pieces, piece=piece, rng=rng)
	soprano_bass_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 3, 0, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([soprano_alto_spacing_expert, soprano_tenor_spacing_expert, soprano_bass_spacing_expert], 4, 0, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=False)
	soprano_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 0,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	soprano_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 0, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	soprano_model = MultiExpert([spacing_multiexpert, soprano_contour_expert, soprano_rhythm_expert], 4, 0,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(soprano_model, 'Soprano_model', dataset, articulation_data, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return soprano_model


def instantiate_and_train_alto_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):

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
	alto_soprano_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 0, 1, pieces=pieces, piece=piece, rng=rng)
	alto_tenor_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 2, 1, pieces=pieces, piece=piece, rng=rng)
	alto_bass_spacing_expert = VoiceSpacingExpert(min_num,max_num, [100,200,100], 3, 1, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([alto_soprano_spacing_expert, alto_tenor_spacing_expert, alto_bass_spacing_expert], 4, 1, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=False)
	alto_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 1,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	alto_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 1, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	alto_model = MultiExpert([spacing_multiexpert, alto_contour_expert, alto_rhythm_expert], 4, 1,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(alto_model, 'Alto_model', dataset, articulation_data, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return alto_model

def instantiate_and_train_tenor_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=None, visualize=False):

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
	tenor_soprano_spacing_expert = VoiceSpacingExpert(min_num, max_num, [100,200,100], 0, 2, pieces=pieces, piece=piece, rng=rng)
	tenor_alto_spacing_expert = VoiceSpacingExpert(min_num, max_num, [100,200,100], 1, 2, pieces=pieces, piece=piece, rng=rng)
	tenor_bass_spacing_expert = VoiceSpacingExpert(min_num, max_num, [100,200,100], 3, 2, pieces=pieces, piece=piece, rng=rng)
	spacing_multiexpert = MultiExpert([tenor_soprano_spacing_expert, tenor_alto_spacing_expert, tenor_bass_spacing_expert], 4, 2, min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=False)
	tenor_contour_expert = VoiceContourExpert(min_num, max_num, [100,200,100], 2,
		voices=voices, gen_length=gen_length, first_note=first_note, rng=rng)
	tenor_rhythm_expert = RhythmExpert(rhythm_encoding_size, max_num-min_num, [100,200,100], 2, 
		timestep_info=timestep_info, prior_timestep_pitch_info=prior_timesteps, pieces=pieces, rhythm_info=rhythm_info, rng=rng)
	tenor_model = MultiExpert([spacing_multiexpert, tenor_contour_expert, tenor_rhythm_expert], 4, 2,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(tenor_model, 'Tenor_model', dataset, articulation_data, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return tenor_model

# builds a multiexpert of one simple generative model for comparison
def instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, voice_to_predict, rhythm_encoding_size, output_dir=None, visualize=False):

	timestep_info = T.itensor3()
	prior_timesteps=T.itensor4()
	pieces=T.itensor4()
	piece=T.itensor3()
	rng = theano.tensor.shared_randomstreams.RandomStreams()
	# do some symbolic variable manipulation so that we can compile a function with updates for all the models
	voices = pieces[:,voice_to_predict]
	gen_length = piece.shape[1]
	first_note = T.argmax(piece[voice_to_predict,0])
	rhythm_info = theano.map(lambda a, t: T.set_subtensor(T.zeros(t)[a % t], 1), sequences=T.arange(gen_length), non_sequences=rhythm_encoding_size)[0]

	simple_generative = SimpleGenerative(max_num-min_num, [100,200,100], 4,voice_to_predict,
		pieces=pieces, prior_timesteps=prior_timesteps, piece=piece, rng=rng)
	model = MultiExpert([simple_generative], 4, voice_to_predict,  min_num, max_num, timestep_length, rhythm_encoding_size,
		pieces=pieces, prior_timesteps=prior_timesteps, timestep_info=timestep_info, piece=piece, rng=rng, transparent=visualize)
	train(model, 'Just_simple_generative', dataset, articulation_data, min_num, max_num, timestep_length, output_dir=output_dir, visualize=visualize)
	return model

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
def generate_voice_by_voice(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, simple=False, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/generate_voice_by_voice/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/generate_voice_by_voice'): os.mkdir('../Data/Output/generate_voice_by_voice')
	os.mkdir(output_dir)
	os.mkdir(output_dir + 'soprano/')
	os.mkdir(output_dir + 'alto/')
	os.mkdir(output_dir + 'tenor/')
	os.mkdir(output_dir + 'bass/')
	# train models
	models = []
	if simple:
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 0, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
			visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 1, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize) if alto_weights is None else load_weights(alto_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 2, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 3, rhythm_encoding_size, output_dir=output_dir + 'bass/', 
			visualize=visualize)if bass_weights is None else load_weights(bass_weights))

		
	else:
		models.append(instantiate_and_train_soprano_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
			visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))
		models.append(instantiate_and_train_alto_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize)if alto_weights is None else load_weights(alto_weights))
		models.append(instantiate_and_train_tenor_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_bass_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'bass/', 
			visualize=visualize)if bass_weights is None else load_weights(bass_weights))


	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece_number = np.random.randint(len(dataset))
		sample_piece = dataset[sample_piece_number]
		sample_articulation = articulation_data[sample_piece_number]
		generated_piece = np.zeros_like(sample_piece)
		generated_articulation = np.zeros_like(sample_articulation)
		for i in range(4):
			generate_random_voice(generated_piece[i], rng)
			generate_random_voice(generated_articulation, rng) #this just works
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,4,1))[0])

			generated_piece[v], generated_articulation[v] = models[v].generate(generated_piece, generated_articulation)

		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(generated_piece, generated_articulation)], path=output_dir + 'output' + str(n) + '.mid')


def harmonize_melody(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, simple=False, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/harmonize_melody/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/harmonize_melody'): os.mkdir('../Data/Output/harmonize_melody')
	os.mkdir(output_dir)
	os.mkdir(output_dir + 'alto/')
	os.mkdir(output_dir + 'tenor/')
	os.mkdir(output_dir + 'bass/')
	# train models
	models = []
	if simple:
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 1, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize) if alto_weights is None else load_weights(alto_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 2, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 3, rhythm_encoding_size, output_dir=output_dir + 'bass/', 
			visualize=visualize)if bass_weights is None else load_weights(bass_weights))
		
	else:
		models.append(instantiate_and_train_alto_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize)if alto_weights is None else load_weights(alto_weights))
		models.append(instantiate_and_train_tenor_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_bass_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'bass/', 
			visualize=visualize)if bass_weights is None else load_weights(bass_weights))
		
	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece_number = np.random.randint(len(dataset))
		sample_piece = dataset[sample_piece_number]
		sample_articulation = articulation_data[sample_piece_number]
		generated_piece = np.zeros_like(sample_piece)
		generated_articulation = np.zeros_like(sample_articulation)
		generated_piece[0] = sample_piece[0]
		generated_articulation[0] = sample_articulation[0]
		for i in range(1,4):
			generate_random_voice(generated_piece[i], rng)
			generate_random_voice(generated_articulation, rng) #this just works
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(1,4,1))[0])

			generated_piece[v], generated_articulation[v] = models[v-1].generate(generated_piece, generated_articulation)



		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(generated_piece, generated_articulation)], path=output_dir + 'output' + str(n) + '.mid')
		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(sample_piece, sample_articulation)], path=output_dir + 'sample_for_output' + str(n) + '.mid')

"""def harmonize_bass(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, simple=False, 
	soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/harmonize_bass/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/harmonize_bass'): os.mkdir('../Data/Output/harmonize_bass')
	os.mkdir(output_dir)
	os.mkdir(output_dir + 'soprano/')
	os.mkdir(output_dir + 'alto/')
	os.mkdir(output_dir + 'tenor/')
	# train models
	models = []
	if simple:
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 2, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 1, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize)if alto_weights is None else load_weights(alto_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 0, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
			visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))
	else:
		models.append(instantiate_and_train_tenor_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_alto_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize)if alto_weights is None else load_weights(alto_weights))
		models.append(instantiate_and_train_soprano_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
			visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))


	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece_number = np.random.randint(len(dataset))
		sample_piece = dataset[sample_piece_number]
		sample_articulation = articulation_data[sample_piece_number]
		generated_piece = np.zeros_like(sample_piece)
		generated_articulation = np.zeros_like(sample_articulation)
		generated_piece[0] = sample_piece[0]
		generated_articulation[0] = sample_articulation[0]
		for i in range(1,4):
			generate_random_voice(generated_piece[i], rng)
			generate_random_voice(generated_articulation, rng) #this just works
		for i in range(100): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,3,1))[0])

			generated_piece[v+1], generated_articulation[v+1] = models[v].generate(generated_piece, generated_articulation)

		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(generated_piece, generated_articulation)], path=output_dir + 'output' + str(n) + '.mid')
		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(sample_piece, sample_articulation)], path=output_dir + 'sample_for_output' + str(n) + '.mid')

def harmonize_melody_and_bass(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, simple=False, soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/harmonize_melody_and_bass/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/harmonize_melody_and_bass'): os.mkdir('../Data/Output/harmonize_melody_and_bass')
	os.mkdir(output_dir)
	os.mkdir(output_dir + 'alto/')
	os.mkdir(output_dir + 'tenor/')
	# train models
	models = []
	if simple:
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 2, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 1, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	else:
		models.append(instantiate_and_train_tenor_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
			visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
		models.append(instantiate_and_train_alto_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
			visualize=visualize)if alto_weights is None else load_weights(alto_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece_number = np.random.randint(len(dataset))
		sample_piece = dataset[sample_piece_number]
		sample_articulation = articulation_data[sample_piece_number]
		generated_piece = np.zeros_like(sample_piece)
		generated_articulation = np.zeros_like(sample_articulation)
		generated_piece[0] = sample_piece[0]
		generated_articulation[0] = sample_articulation[0]
		generated_piece[3] = sample_piece[3]
		generated_articulation[3] = sample_articulation[3]
		for i in range(1,3):
			generate_random_voice(generated_piece[i], rng)
			generate_random_voice(generated_articulation, rng) #this just works
		for i in range(10): # arbitrary number of generation runs
			v = int(np.floor(rng.uniform(0,2,1))[0])

			generated_piece[v+1], generated_articulation[v+1] = models[v].generate(generated_piece, generated_articulation)

			output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(generated_piece, generated_articulation)], path=output_dir + 'output' + str(n) + 'pass ' + str(i) + '.mid')
		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(sample_piece, sample_articulation)], path=output_dir + 'sample_for_output' + str(n) + '.mid')

# an experiment for my own sanity
# generates a piece feeding in the actual other voices to each model, but assembling the final piece out of just the outputs
# TODO: update with articulation
def generate_informed(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, 
	soprano_weights=None, alto_weights=None, tenor_weights=None, bass_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/generate_informed/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/generate_informed'): os.mkdir('../Data/Output/generate_informed')
	os.mkdir(output_dir)
	os.mkdir(output_dir + 'soprano/')
	os.mkdir(output_dir + 'alto/')
	os.mkdir(output_dir + 'tenor/')
	os.mkdir(output_dir + 'bass/')
	# train models
	models = []
	models.append(instantiate_and_train_bass_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'bass/', 
		visualize=visualize)if bass_weights is None else load_weights(bass_weights))
	models.append(instantiate_and_train_tenor_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'tenor/', 
		visualize=visualize)if tenor_weights is None else load_weights(tenor_weights))
	models.append(instantiate_and_train_alto_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'alto/', 
		visualize=visualize)if alto_weights is None else load_weights(alto_weights))
	models.append(instantiate_and_train_soprano_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
		visualize=visualize) if soprano_weights is None else load_weights(soprano_weights))

	for n in range(num_to_generate):
		# choose a random bach piece to steal the length of

		sample_piece_number = np.random.randint(len(dataset))
		sample_piece = dataset[sample_piece_number]
		sample_articulation = articulation_data[sample_piece_number]
		generated_piece = np.zeros_like(sample_piece)
		generated_articulation = np.zeros_like(sample_articulation)
		for i in range(4):
			generated_piece[i], generated_articulation[i] = models[i].generate(sample_piece, sample_articulation)

		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(generated_piece, generated_articulation)], path=output_dir + 'output' + str(n) + '.mid')
		output_midi([timesteps_to_notes(voice, artic, min_num, timestep_length * PPQ) for voice,artic in zip(sample_piece, sample_articulation)], path=output_dir + 'sample_for_output' + str(n) + '.mid')
"""
def generate_melody(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, num_to_generate, simple=False, 
	soprano_weights=None, visualize=False):
	rng = np.random.RandomState()
	# make output directory
	output_dir = '../Data/Output/generate_melody/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/generate_melody'): os.mkdir('../Data/Output/generate_melody')
	os.mkdir(output_dir)
	os.mkdir(output_dir + 'soprano/')

	if simple: model =instantiate_and_train_simple_model(dataset, articulation_data, min_num, max_num, timestep_length, 0, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
			visualize=visualize) if soprano_weights is None else load_weights(soprano_weights)
	else: model = instantiate_and_train_soprano_model(dataset, articulation_data, min_num, max_num, timestep_length, rhythm_encoding_size, output_dir=output_dir + 'soprano/', 
			visualize=visualize) if soprano_weights is None else load_weights(soprano_weights)
	for n in range(num_to_generate):
		sample_piece_number = np.random.randint(len(dataset))
		sample_piece = dataset[sample_piece_number]
		sample_articulation = articulation_data[sample_piece_number]
		if visualize:
			generated_melody, generated_articulation, probs = model.generate(sample_piece, sample_articulation)
			i = 0
			for expert_model,p in zip(model.expert_models, probs):
				visualize_probs(p, title=expert_model.__class__.__name__ + 'GeneratedOutput' + str(n), 
					path=output_dir + expert_model.__class__.__name__ + str(i)+ 'GeneratedOutput' + str(n))
				i+=1
			visualize_probs(probs[-2], title= 'ArticulationModelGeneratedOutput' + str(n), 
					path=output_dir + 'ArticulationModelGeneratedOutput' + str(n))
			visualize_probs(probs[-1], title= 'FinalGeneratedOutput' + str(n), 
					path = output_dir + 'FinalGeneratedOutput' + str(n))
		else:
			generated_melody, generated_articulation = model.generate(sample_piece, sample_articulation)
		output_midi([timesteps_to_notes(generated_melody, generated_articulation, min_num, timestep_length * PPQ)], path=output_dir + 'output' + str(n) + '.mid')

# load dataset
if __name__ == '__main__':
	dataset, articulation, min_num, max_num, timestep_length = pickle.load(open('../Data/music21_articulation_dataset.p', 'rb'))
	rhythm_encoding_size = int(4//timestep_length) # modified for music21: units are no longer midi timesteps (240 to a quarter note) but quarterLengths (1 to a quarter note)
	#generate_voice_by_voice(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 10, visualize=False)#, soprano_weights='../Data/Output/Soprano_model/Tue,17,09:18/320.p', alto_weights='../Data/Output/Alto_model/Tue,17,09:23/240.p',
	#																			#tenor_weights='../Data/Output/Tenor_model/Tue,17,09:30/280.p', bass_weights='../Data/Output/Bass_Model/Tue,17,09:37/60.p')
	#
	#generate_informed(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 10, visualize=False)
	#harmonize_melody_and_bass(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 1, visualize=False)
	output_dir = '../Data/Output/simple_generative/' + strftime("%a,%d,%H:%M", localtime())+ '/'
	if not os.path.exists('../Data/Output/simple_generative'): os.mkdir('../Data/Output/simple_generative')
	os.mkdir(output_dir)
	model = instantiate_and_train_simple_model(dataset, articulation, min_num, max_num, timestep_length, 0, rhythm_encoding_size, output_dir=output_dir, visualize=True)

	#harmonize_melody(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 10, visualize=False)
	#harmonize_bass(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 10, visualize=False)
	
