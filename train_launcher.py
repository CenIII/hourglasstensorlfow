"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from inputgen import SFSDataProvider as DataGenerator

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config('config.cfg')
	
	print('--Creating Dataset')
	# dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
	# dataset._create_train_table()
	# dataset._randomize()
	# dataset._create_sets()
	
	data_gen = DataGenerator()

	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], outputDim=params['num_joints'], batch_size=params['batch_size'], training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'],  modif=False) # w_loss=params['weighted_loss'] ,joints= params['joint_list'],nModules=params['nmodules'], 
	#nLow=params['nlow'],attention = params['mcam'],
	model.generate_model()
	model.training_init(data_gen, nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)
	
