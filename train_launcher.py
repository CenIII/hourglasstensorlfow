"""
TRAIN LAUNCHER 

"""

# import configparser
from hourglass_tiny import HourglassModel
from inputgen import SFSDataProvider as DataGenerator

# def process_config(conf_file):
# 	"""
# 	"""
# 	params = {}
# 	config = configparser.ConfigParser()
# 	config.read(conf_file)
# 	for section in config.sections():
# 		if section == 'DataSetHG':
# 			for option in config.options(section):
# 				params[option] = eval(config.get(section, option))
# 		if section == 'Network':
# 			for option in config.options(section):
# 				params[option] = eval(config.get(section, option))
# 		if section == 'Train':
# 			for option in config.options(section):
# 				params[option] = eval(config.get(section, option))
# 		if section == 'Validation':
# 			for option in config.options(section):
# 				params[option] = eval(config.get(section, option))
# 		if section == 'Saver':
# 			for option in config.options(section):
# 				params[option] = eval(config.get(section, option))
# 	return params


if __name__ == '__main__':
	# print('--Parsing Config File')
	# params = process_config('config.cfg')
	
	print('--Creating Dataset')
	# dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
	# dataset._create_train_table()
	# dataset._randomize()
	# dataset._create_sets()
	
	data_gen = DataGenerator()

	model = HourglassModel(nFeat=128, nStack=4, nLow=4, outputDim=3, batch_size=16,training=True, drop_rate=0.2, lear_rate=1e-6, decay=0.96, decay_step=2000,logdir_train='./logdir_train', logdir_test='./logdir_test', tiny=True, w_loss=False,modif=False)
	model.generate_model()
	model.training_init(data_gen, nEpochs = 10, epochSize = 1000, batchSize=8, saveStep = 500, load = None)

