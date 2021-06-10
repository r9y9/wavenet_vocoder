import os
import errno

target_dir = '/Tacotron-2/tacotron_output'

try:
    os.makedirs(target_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

speakers = ['Regina', 'Aiste', 'Edvardas', 'Vladas']

target = os.path.join(target_dir, 'train.txt')

#training_data_Regina/audio/speech-audio-05124.npy|training_data_Regina/mels/speech-mel-05124.npy|tacotron_output_Regina/gta/mel-speech-05124.npy|<no_g>|vienúolika eũrų
#/Tacotron-2/tacotron_output_Regina/gta/speech-mel-04883.npy

with open(target, 'wt', encoding='utf-8') as fpw:
	for i, name in enumerate(speakers):
		speaker_train = '/Tacotron-2/training_data_' + name + '/train.txt'

		with open(speaker_train, 'rt', encoding='utf-8') as fp:
			for line in fp:
				audio,mel,_,length,_,text = line.strip().split('|')
				mel = '/Tacotron-2/training_data_' + name + '/mels/' + mel
				audio = '/Tacotron-2/training_data_' + name + '/audio/' + audio
				line = '|'.join([audio, mel, length, text, str(i)]) + '\n'
				fpw.write(line)

