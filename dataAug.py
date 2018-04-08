import sys
sys.path.insert(0, '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/Augmentor')
import Augmentor
import os


os.mkdir("./train/color/output")

p = Augmentor.Pipeline("./train/color")
p.ground_truth("./train/mask","./train/normal")

# p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)

p.flip_top_bottom(probability=0.5)

p.shift_random(probability=0.5)


for i in range(10):
	os.mkdir("./train/color/output/0")
	os.mkdir("./train/color/output/1")
	os.mkdir("./train/color/output/2")

	p.sample(200)
	os.rename("./train/color/output/0", "./train/color/output/color"+str(i))
	os.rename("./train/color/output/1", "./train/color/output/mask"+str(i))
	os.rename("./train/color/output/2", "./train/color/output/normal"+str(i))

