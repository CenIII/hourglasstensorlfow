####################################################
Surface normals prediction from a single image
EECS 442 Team Project (2018 Fall)
University of Michigan, Ann Arbor

@author: Chuan Cen, Siqi Shen, Jiacheng Zhu, Huajing Zhao
####################################################

I. Data Augmentation

	In this version, you need to use data augmentor to generate a set of data sets first. To do that, change "root_dirt" in "dataAug.py" to the root path of the original training set, define how many sets of data set you want to generate. Also you can define how the data is going to augmented, such as the probability of shifting and flipping. And then simply run
	
	`
	python3 dataAug.py
	
	`
II. Training

	- Train without pre-trained model
		In file "launcher.py", assign "None" to variable "model_load_dir". Make sure "training=True". Then run
		`
		python3 launcher.py
		
		`
	- Train with a pre-trained model
		In file "launcher.py", assign the path of your pre-trained model to the variable "model_load_dir". Make sure "training=True". Then run
		`
		python3 launcher.py
		
		`
III. Model saving
	
	It will save the trained model everytime an epoch ends. You can decide how big the epoch size is so as to make the saving period shorter or longer. It'll save to './' (where "launcher.py" is located) directly. 

IV. Testing

	Change the variable "training" to "False" in "launcher.py". Assign the path of your trained model to the variable "model_load_dir". Run 
	`
	python3 launcher.py

	`
