"""
TRAIN LAUNCHER 

"""

# import configparser
from hourglass_tiny import HourglassModel
from inputgen import SFSDataProvider, SFSTestDataProvider
import shutil
import os
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

	training = True
	data_dir = "/home/chuancen/PJDATA/"

	model_load_dir = "/home/chuancen/PJDATA/model/tiny_hourglass_1"
	save_dir = "/home/chuancen/PJDATA/test/normal/"
	shutil.rmtree(save_dir,ignore_errors=True)
	os.mkdir(save_dir)

	if training==True:
		data_gen = SFSDataProvider(str(data_dir+"train/"))
		# initial learning rate is 2.5*1e-4
		model = HourglassModel(nFeat=256, nStack=4, nLow=4, outputDim=3, batch_size=4,training=True, drop_rate=0.2, lear_rate=9*1e-5, decay=0.96, decay_step=5000,logdir_train='./logdir_train', logdir_test='./logdir_test', tiny=False, w_loss=False,modif=False)
		model.generate_model()
		model.training_init(data_gen, nEpochs = 2, epochSize = 20000, batchSize=4, saveStep = 10000, load = model_load_dir)
	else:
		data_gen = SFSTestDataProvider(str(data_dir+"test/"))
		model_test = HourglassModel(nFeat=256, nStack=4, nLow=4, outputDim=3, batch_size=4,training=False, drop_rate=0.2, lear_rate=9*1e-5, decay=0.95, decay_step=10000,logdir_train='./logdir_train', logdir_test='./logdir_test', tiny=False, w_loss=False,modif=False)
		model_test.generate_model()
		model_test.test_init(data_gen, load = model_load_dir, save = save_dir)
