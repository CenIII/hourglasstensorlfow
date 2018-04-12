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
	
	training = True
	data_dir = "/home/chuancen/PJDATA/"
	model_load_dir = None #"/home/chuancen/PJDATA/model/tiny_hourglass_3"
	pred_save_dir = "/home/chuancen/PJDATA/test/normal/"

	if training==True:
		data_gen = SFSDataProvider(str(data_dir+"train/"))
		# initial learning rate is 2.5*1e-4
		model = HourglassModel(nFeat=384, nStack=5, nLow=4, outputDim=3, batch_size=4,training=True, drop_rate=0.2, lear_rate=2.5*1e-4, decay=0.96, decay_step=1000,logdir_train='./logdir_train', logdir_test='./logdir_test', tiny=False, w_loss=False,modif=False)
		model.generate_model()
		model.training_init(data_gen, nEpochs = 5, epochSize = 20000, batchSize=4, saveStep = 10000, load = model_load_dir)
	else:

		shutil.rmtree(pred_save_dir,ignore_errors=True)
		os.mkdir(pred_save_dir)
		data_gen = SFSTestDataProvider(str(data_dir+"test/"))
		model_test = HourglassModel(nFeat=384, nStack=5, nLow=4, outputDim=3, batch_size=4,training=False, drop_rate=0.2, lear_rate=2.5*1e-4, decay=0.96, decay_step=10000,logdir_train='./logdir_train', logdir_test='./logdir_test', tiny=False, w_loss=False,modif=False)
		model_test.generate_model()
		model_test.test_init(data_gen, load = model_load_dir, save = pred_save_dir)
