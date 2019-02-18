import glob
import os
fw = open('vgg_5000.txt', 'a')
for i, base_path in enumerate(os.listdir('./train/')):
	for img in glob.glob('train/%s/*'%base_path):
		fw.write('%s/%s %s\n'%(base_path, img.split('/')[-1], str(i)))
fw.close()

