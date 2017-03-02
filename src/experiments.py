# experiments.py
# contains all of the experiment code to generate a melody, harmonization and original chorale for the model name in argv[1]

from chorale_generator import *

if len(sys.argv) == 1:
	print("Include your model name as an argument! Choices are 'simple' and 'multi' right now")
	sys.exit(1)
else:
	MODEL_NAME = sys.argv[1]
	simple = (MODEL_NAME == 'simple')



dataset, articulation, min_num, max_num, timestep_length = pickle.load(open('../Data/music21_articulation_dataset.p', 'rb'))
rhythm_encoding_size = int(4//timestep_length) # modified for music21: units are no longer midi timesteps (240 to a quarter note) but quarterLengths (1 to a quarter note)
#generate_melody(dataset, articulation, min_num, max_num, timestep_length, rhythm_encoding_size, 10, simple=simple, visualize=True)
#generate_voice_by_voice(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 10, simple=simple, visualize=False)
#generate_melody(dataset, articulation, min_num,  max_num, timestep_length, rhythm_encoding_size, 10, simple=simple, 
	#soprano_weights='../Data/Final_output/multi_melody/Thu,23,12:08/soprano/700.p',
	#tenor_weights='../Data/Output/generate_voice_by_voice/Wed,22,20:13/tenor/700.p',
	#bass_weights='../Data/Output/generate_voice_by_voice/Wed,22,20:13/bass/700.p',
	#visualize=True)

harmonize_melody(dataset, articulation, min_num, max_num, timestep_length, rhythm_encoding_size, 10, simple=simple, 
	alto_weights='../Data/Output/harmonize_melody/Fri,24,10:14/alto/1500.p',
	tenor_weights='../Data/Output/harmonize_melody/Fri,24,10:14/tenor/1200.p',
	bass_weights='../Data/Output/harmonize_melody/Fri,24,10:14/bass/800.p',
	visualize=True)