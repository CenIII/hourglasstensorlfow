import sys
import Augmentor
import os
import shutil

root_dir = "/home/chuancen/PJDATA/train/"
dataset_num = 10
num_samples_each = 20000


shutil.rmtree(root_dir+"color/output", ignore_errors=True)
os.mkdir(root_dir+"color/output")
p = Augmentor.Pipeline(root_dir+"color")
p.ground_truth(root_dir+"mask",root_dir+"normal")
#p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.shift_random(probability=0.75)

for i in range(dataset_num):
	os.mkdir(root_dir+"color/output/0")
	os.mkdir(root_dir+"color/output/1")
	os.mkdir(root_dir+"color/output/2")
	p.sample(num_samples_each)
	os.rename(root_dir+"color/output/0", root_dir+"color/output/color"+str(i))
	os.rename(root_dir+"color/output/1", root_dir+"color/output/mask"+str(i))
	os.rename(root_dir+"color/output/2", root_dir+"color/output/normal"+str(i))
	shutil.move(root_dir+"color/output/color"+str(i), root_dir)
	shutil.move(root_dir+"color/output/mask"+str(i), root_dir)
	shutil.move(root_dir+"color/output/normal"+str(i), root_dir)

shutil.rmtree(root_dir+"color/output", ignore_errors=True)
