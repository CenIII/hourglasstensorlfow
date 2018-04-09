import sys
# sys.path.insert(0, '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/Augmentor')
import Augmentor
import os

root_dir = "/home/chuancen/PJDATA/train/"

os.mkdir(root_dir+"color/output")

p = Augmentor.Pipeline(root_dir+"color")
p.ground_truth(root_dir+"mask",root_dir+"normal")

# p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)

p.flip_top_bottom(probability=0.5)

#p.shift_random(probability=0.5)


for i in range(10):
	os.mkdir(root_dir+"color/output/0")
	os.mkdir(root_dir+"color/output/1")
	os.mkdir(root_dir+"color/output/2")

	p.sample(20000)
	os.rename(root_dir+"color/output/0", root_dir+"color/output/color"+str(i))
	os.rename(root_dir+"color/output/1", root_dir+"color/output/mask"+str(i))
	os.rename(root_dir+"color/output/2", root_dir+"color/output/normal"+str(i))

