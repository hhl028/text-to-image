import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

# DID NOT TRAIN IT ON MS COCO YET
def save_caption_vectors_ms_coco(data_dir, split, batch_size):
	meta_data = {}
	ic_file = join(data_dir, 'annotations/captions_{}2014.json'.format(split))
	with open(ic_file) as f:
		ic_data = json.loads(f.read())

	meta_data['data_length'] = len(ic_data['annotations'])
	with open(join(data_dir, 'meta_{}.pkl'.format(split)), 'wb') as f:
		pickle.dump(meta_data, f)

	model = skipthoughts.load_model()
	batch_no = 0
	print "Total Batches", len(ic_data['annotations'])/batch_size

	while batch_no*batch_size < len(ic_data['annotations']):
		captions = []
		image_ids = []
		idx = batch_no
		for i in range(batch_no*batch_size, (batch_no+1)*batch_size):
			idx = i%len(ic_data['annotations'])
			captions.append(ic_data['annotations'][idx]['caption'])
			image_ids.append(ic_data['annotations'][idx]['image_id'])

		print captions
		print image_ids
		# Thought Vectors
		tv_batch = skipthoughts.encode(model, captions)
		h5f_tv_batch = h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no)), 'w')
		h5f_tv_batch.create_dataset('tv', data=tv_batch)
		h5f_tv_batch.close()

		h5f_tv_batch_image_ids = h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no)), 'w')
		h5f_tv_batch_image_ids.create_dataset('tv', data=image_ids)
		h5f_tv_batch_image_ids.close()

		print "Batches Done", batch_no, len(ic_data['annotations'])/batch_size
		batch_no += 1


def save_caption_vectors_flowers(data_dir):
	import time
	
	img_dir = join(data_dir, 'flowers/jpg')
	image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
	print image_files[300:400]
	print len(image_files)
	image_captions = { img_file : [] for img_file in image_files }

	caption_dir = join(data_dir, 'flowers/text_c10')
	class_dirs = []
	for i in range(1, 103):
		class_dir_name = 'class_%.5d'%(i)
		class_dirs.append( join(caption_dir, class_dir_name))

	for class_dir in class_dirs:
		caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
		for cap_file in caption_files:
			with open(join(class_dir,cap_file)) as f:
				captions = f.read().split('\n')
			img_file = cap_file[0:11] + ".jpg"
			# 5 captions per image
			image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

	print len(image_captions)

	model = skipthoughts.load_model()
	encoded_captions = {}


	for i, img in enumerate(image_captions):
		st = time.time()
		encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
		print i, len(image_captions), img
		print "Seconds", time.time() - st
		
	
	h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
	for key in encoded_captions:
		h.create_dataset(key, data=encoded_captions[key])
	h.close()

def save_caption_vectors_shapes(data_dir):
	import time
	
	img_dir = join(data_dir, 'shapes/images')
	image_files = [f for f in os.listdir(img_dir) if 'png' in f]
	print image_files[300:400]
	print len(image_files)
	image_captions = { img_file : [] for img_file in image_files }

	caption_dir = join(data_dir, 'shapes/texts')
        caption_files = [f for f in os.listdir(caption_dir) if 'txt' in f]
        for cap_file in caption_files:
                with open(join(caption_dir,cap_file)) as f:
                        captions = f.read().split('\n')
                img_file = cap_file[0:5] + ".png"
                # 5 captions per image
                image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

	print len(image_captions)

	model = skipthoughts.load_model()
	encoded_captions = {}

	for i, img in enumerate(image_captions):
		st = time.time()
		encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
		print i, len(image_captions), img
		print "Seconds", time.time() - st
		
	
	h = h5py.File(join(data_dir, 'shapes_tv.hdf5'))
	for key in encoded_captions:
		h.create_dataset(key, data=encoded_captions[key])
	h.close()

def save_caption_vectors_flickr(data_dir):
	import time
	
	print("BEGIN LOADING FLICKR")
	img_dir = join('/home/hhl028/WINNtranslation/flickr', 'cropped_images')

	def get_images_path_in_directory(path):
	    '''
	    Get path of all images recursively in directory filtered by extension list.
	        path: Path of directory contains images.
	    Return path of images in selected directory.
	    '''
	    images_path_in_directory = []
	    image_extensions = ['.png', '.jpg']
	    
	    for root_path, directory_names, file_names in os.walk(path):
	        for file_name in file_names:
	            lower_file_name = file_name.lower()
	            if any(map(lambda image_extension: 
	                       lower_file_name.endswith(image_extension), 
	                       image_extensions)):
	                images_path_in_directory.append(os.path.join(root_path, file_name))

	    return images_path_in_directory

	# # read the labels into some data structure
	img_to_text = {}
	with open("/home/hhl028/WINNtranslation/flickr/image_labels.txt", 'r') as file:
	    i = 0
	    for line in file:
	        tmp = line.split("|")
	        label = '_'.join(tmp[:2])
	        img_to_text[label] = tmp[2:]
	        if i % 25000 == 0:
	            print("Finished iteration %d" % (i))
	        i += 1

	def remove_invalid_paths(pos_all_images_path, img_to_text):
	    '''
	    Some images don't have text. Don't use those.
	    '''
	    res = []
	    for path in pos_all_images_path:
	        # get image name from path
	        img_name = path.split('/')[-1].replace('.png','').replace('.jpg','')
	        label = '_'.join((img_name).split("_")[:2])
	        # don't use image unless we have phrases for it
	        if label in img_to_text:
	            res.append(path)
	    return res

	# remove images with invalid paths
	image_files = remove_invalid_paths(image_files, img_to_text)
	image_captions = { img_file : [] for img_file in image_files }

	# remove images without one of the target terms in it 
	def get_image_texts(pos_all_images_path, target_phrases, img_to_text):
	    '''
	    Given a list of positive ground truth images/text, return two lists, 
	    one of image paths and one of texts.
	    
	    At least one of the phrases for the given image must contain the 
	    target text.
	    '''
	    return_labels = []
	    return_texts = []
	    for image_path in pos_all_images_path:
	        img_name = image_path.split('/')[-1].replace('.png','').replace('.jpg','')
	        label = '_'.join((img_name).split("_")[:2])
	        for phrase in img_to_text[label]:
	            for target_phrase in target_phrases:
	                if target_phrase in phrase + '\n':
	                    return_labels.append(label)
	                    return_texts.append(phrase.strip())
	    return zip(return_labels, return_texts)

	# class 1: man
	target_phrases = [' man\n']
    label_text_1 = get_image_texts(pos_all_images_paths_all, target_phrases, img_to_text)
    label_text_1 = label_text_1[:len(pos_all_images_text_1)/4]

    # class 2: dog
    target_phrases = ['dog\n']
    label_text_2 = get_image_texts(pos_all_images_paths_all, target_phrases, img_to_text)

    # combine classes
    label_text_all = label_text_1 + label_text_2

	model = skipthoughts.load_model()
	encoded_captions = {}

	for label, text in label_text_all:
		encoded_captions[label] = skipthoughts.encode(model, text)
		
	print("END ENCODING")

	h = h5py.File(join(data_dir, 'flickr_tv.hdf5'))
	for key in encoded_captions:
		h.create_dataset(key, data=encoded_captions[key])
	h.close()
	print("DONE WRITING H5PY")
			
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train',
                       help='train/val')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
	parser.add_argument('--data_set', type=str, default='flowers',
                       help='Data Set : Flowers, MS-COCO')
	args = parser.parse_args()
	
	if args.data_set == 'flowers':
		save_caption_vectors_flowers(args.data_dir)
	elif args.data_set == 'shapes':
		save_caption_vectors_shapes(args.data_dir)
	elif args.data_set == 'flickr':
		save_caption_vectors_flickr(args.data_dir)
	else:
		save_caption_vectors_ms_coco(args.data_dir, args.split, args.batch_size)

if __name__ == '__main__':
	main()
