import cv2
import numpy as np
import os
import argparse
import random
import time
import shutil
from io import BytesIO
from collections import Counter
from PIL import Image
from itertools import combinations 

# Args
parser = argparse.ArgumentParser(
    description="Generate your own synthetic LoL minimap images for training or testing")

parser.add_argument(
    '-nc',
    '--no_clear',
    help="don't clear the output folder (Default: False)",
    action='store_false')

subparsers = parser.add_subparsers(help = 'sub-command help')
parser_generate = subparsers.add_parser('generate', help='generate .npz files')
parser_generate.set_defaults(which='generate')
parser_dump = subparsers.add_parser('dump', help='dumps minimap images from .npz file(s)')
parser_dump.set_defaults(which='dump')

parser_generate.add_argument(
    '-ni',
    '--number_of_images',
    help="number of minimap images to generate, to avoid wasting clusters ensure you align on 32 images (Default: 119808)",
    metavar='N',
	type=int,
	default=119808)

parser_generate.add_argument(
	'-cn',
	'--number_of_champs_per_minimap',
	help="number of champions to place on a single minimap image (Default: 30)",
	metavar='N',
	type=int,
	default=20)

parser_generate.add_argument(
	'-txt',
	'--text_files',
    help="don't create NPZ files, instead generate .txt files alongside .jpg files (Default: false)",
	action='store_true')

parser_generate.add_argument(
    '-npz',
    '--number_of_npz_files',
    help="number of .npz files to create (Default: 8)",
    metavar='N',
	type=int,
	default=8)

parser_generate.add_argument(
	'-o',
	'--output_directory',
	help="the folder to which minimap images are generated (Default: '.\\output')",
	type=str,
	default='output')

parser_dump.add_argument(
    '-i',
    '--input_directory',
    help="the folder from which to read the .npz files (Default: '.')",
	type=str,
	default='output')

parser_dump.add_argument(
	'-o',
	'--output_directory',
	help="the folder to which minimap images are dumped (Default: '.\\images')",
	type=str,
	default='dump')

def composite_icons(icon_dict, mapping_dict):
	i = random.randint(0, 17)
	minimap = Image.open("base_minimap/minimap_blue_%d.png" % i)
	minimap = minimap.convert('RGBA')

	# We resize the images such that we train on various size minimaps because players can resize their minimaps
	# We use this https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py to calculate the anchors before training
	size = random.randint(150, 200)
	ratio = 295 / size
	npdata = []
	data = []

	for champion, value in icon_dict.items():
		icon = Image.fromarray(np.uint8(value[0]))
		icon = icon.convert('RGBA')  

	    # Choose a random x,y position for the icon
		paste_position = (value[1], value[2])
	    
	    # Create a new icon image as large as the minimap and paste it on top
		new_icon = Image.new('RGBA', minimap.size, color = (0, 0, 0, 0))
		new_icon.paste(icon, paste_position)
	        
	    # Extract the alpha channel from the icon and paste it into a new image the size of the minimap
		alpha_mask = icon.getchannel(3)
		new_alpha_mask = Image.new('L', minimap.size, color=0)
		new_alpha_mask.paste(alpha_mask, paste_position)
		composite = Image.composite(new_icon, minimap, new_alpha_mask)
		minimap = composite
	    
		# Grab the alpha pixels above a specified threshold
		hard_mask = Image.fromarray(np.uint8(new_alpha_mask) * 255, 'L')   
	    
	 	# Get the smallest & largest non-zero values in each dimension and calculate the bounding box
		nz = np.nonzero(hard_mask)
		x_min, x_max = np.min(nz[1]) / ratio, np.max(nz[1]) / ratio
		y_min, y_max = np.min(nz[0]) / ratio, np.max(nz[0]) / ratio
		# bbox = [x_min, y_min, x_max, y_max] 
		# cntr = (x_min+ int((x_max-x_min)/2), y_min+int((y_max-y_min)/2))
		npdata.append((mapping_dict[champion[:-4]], int(x_min + ((x_max - x_min) / 2)), int(y_min + ((y_max - y_min) / 2)), int(28 / ratio), int(28 / ratio)))
		data.append((mapping_dict[champion[:-4]], (x_min + ((x_max - x_min) / 2)) / size, (y_min + ((y_max - y_min) / 2)) / size, (28 / ratio) / size, (28 / ratio) / size))
	data_array = np.asarray(npdata)
	composite = composite.resize((size, size), resample=Image.BILINEAR)
	return composite, data_array, data

def load_icons(input_directory):
	mapping, rsze_icon_dict, rsze_icon_list, count = {}, {}, [], 0
	outfile = open("league_classes.txt", "w+")
	for file in os.scandir(input_directory):
		if file.name[-4:] == ".png":
			icon = Image.open(file.path)
			icon = icon.convert('RGBA')
			rsze_icon_dict[file.name] = icon
			rsze_icon_list.append(file.name)
			mapping[file.name[:-4]] = count
			count+=1
			outfile.write((file.name[:-4] + '\n'))
	return rsze_icon_dict, rsze_icon_list, mapping, count

def randomize_icons(icon_dict, icon_list, number_of_champs_per_minimap):
	# Grab X random champs from dictionary:
	random_champions = random.sample(icon_list, number_of_champs_per_minimap)
	random_champions_dict = {}
	for champion in random_champions:
		champion_frame = icon_dict[champion]
		random_champions_dict[champion] = (champion_frame, random.randint(5, 260), random.randint(5, 260))
	return random_champions_dict

def generate_txt(number_of_images, number_of_champs_per_minimap, output_directory):
	icon_dict, icon_list, mapping_dict, number_of_champs = load_icons("minimap_icons_28x28")
	max_num_per_champ = int(number_of_images / number_of_champs) * number_of_champs_per_minimap
	print("Generating %d minimap(s) with %d champions per minimap (max %d) from %d champions" % (number_of_images, number_of_champs_per_minimap, max_num_per_champ, number_of_champs))
	champion_counter = Counter()
	output_dir = os.path.join(output_directory, 'img0')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Split minimap generation across number_of_npz_files
	for i in range(1, number_of_images):
		# Dynamically reduced icon_list cannot be less than number_of_champs_per_minimap champions:
		# Hack to create balanced classes - UGLY but works fine for now.
		# TODO: Balanced data set creation needs to be cleaned up.
		# https://stackoverflow.com/questions/45624395/evenly-distributed-random-strings-in-python
		for key, value in champion_counter.items():
			if value >= max_num_per_champ:
				if (key + '.png') in icon_list:
					icon_list.remove((key + '.png'))
		if len(icon_list) >= number_of_champs_per_minimap:
			random_icon_dict = randomize_icons(icon_dict, icon_list, number_of_champs_per_minimap)
			composite_image, _, bbox = composite_icons(random_icon_dict, mapping_dict)
			composite_image = composite_image.convert('RGB')

			# Print out stats every 100th iteration:
			if i%100 == 0:
				print('.', end="", flush=True)
			for key in random_icon_dict:
				champion_counter[key[:-4]]+=1

			outfile = os.path.join(output_dir, "minimap_" + str(i))
			with open((outfile + '.txt'), 'w') as fp:
				fp.write('\n'.join('%s %f %f %f %f' % x for x in bbox))
			composite_image.save(outfile + '.jpg', "JPEG", quality=random.randint(45, 65), optimize=True, progressive=True)
			if i%5000 == 0:
				output_dir = os.path.join(output_directory, ('img' + str(i)))
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				print("\n %d minimap(s) generated with %d champions left in list" % (i, len(icon_list)))
				print(champion_counter, "\n")
		else:
			print("\n%d minimap(s) generated which is less than the requested %d" % (i, number_of_images))
			break
	print("Final champion count in data set:")
	for key, value in champion_counter.most_common():
		print(key, ":", value)

def generate(number_of_npz_files, number_of_images, number_of_champs_per_minimap, output_directory):
	icon_dict, icon_list, mapping_dict, number_of_champs = load_icons("minimap_icons_28x28")
	max_num_per_champ = int(number_of_images / number_of_champs) * number_of_champs_per_minimap
	print("Generating %d minimap(s) across %d .npz file(s) with %d champions per minimap (max %d) from %d champions" % 
		(number_of_images, number_of_npz_files, number_of_champs_per_minimap, max_num_per_champ, number_of_champs))
	champion_counter = Counter()
	for j in range(number_of_npz_files):
		start_time = time.time()
		# Split minimap generation across number_of_npz_files
		for i in range(int(number_of_images / number_of_npz_files)):
			# Dynamically reduced icon_list cannot be less than number_of_champs_per_minimap champions:
			# Hack to create balanced classes - UGLY but works fine for now.
			# TODO: Balanced data set creation needs to be cleaned up.
			# https://stackoverflow.com/questions/45624395/evenly-distributed-random-strings-in-python
			for key, value in champion_counter.items():
				if value >= max_num_per_champ:
					if (key + '.png') in icon_list:
						icon_list.remove((key + '.png'))
			if len(icon_list) >= number_of_champs_per_minimap:
				random_icon_dict = randomize_icons(icon_dict, icon_list, number_of_champs_per_minimap)
				composite_image, icon_bbox, _ = composite_icons(random_icon_dict, mapping_dict)
				composite_image = composite_image.convert('RGB')

				sample = random.randint(200,240)
				composite_image_pix = composite_image.resize((sample, sample), resample=Image.BILINEAR)
				composite_image = composite_image_pix.resize(composite_image.size, Image.NEAREST)
				# Hack creates stackable array of arrays :
				composite_array = np.array(np.asarray(composite_image))
				icon_bbox       = np.array(np.asarray(icon_bbox))

				# Print out stats every 100th iteration:
				if i%100 == 0 and i > 0:
					print('.', end="", flush=True)
				for key in random_icon_dict:
					champion_counter[key[:-4]]+=1

				# NPZ files generated:
				if i == 0:
					stacked_composite_array = np.copy([composite_array])
					stacked_icon_bbox = np.copy([icon_bbox])
				else:
					composite_array         = np.copy([composite_array])
					stacked_composite_array = np.append(stacked_composite_array, composite_array, 0) 
					icon_bbox         = np.copy([icon_bbox])
					stacked_icon_bbox = np.append(stacked_icon_bbox, icon_bbox, 0)			
			else:
				print("\n%d minimap(s) generated which is less than the requested %d" % (i, number_of_images))
				break
		outfile = os.path.join(output_directory, ("data_" + str(j) + ".npz"))
		np.savez(outfile, images=stacked_composite_array, boxes=stacked_icon_bbox)
		print("\nFile:", outfile, "generated in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), "with", i, "minimap(s), from", len(icon_list), "champion(s) with", len(champion_counter.keys()), "unique champion(s)")
		print(champion_counter, "\n")
	print("\nFinal champion count in data set:\n")
	for key, value in champion_counter.most_common():
		print(key, ":", value)

def dump_images(input_directory, output_directory):
	for file in os.scandir(input_directory):
		if file.name[-4:] == ".npz":
			print("Processing: " + file.path)
			image_array = np.load(file.path)
			for i in range(image_array['images'].shape[0]):
				cv2.imwrite(os.path.join(output_directory, "cluster_0_image_%d.png" % i), cv2.cvtColor(image_array['images'][i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
				#with open(os.path.join(output_directory, "cluster_0_image_%d.txt" % i), 'w') as fp:
				#	fp.write('\n'.join('%s %f %f %f %f' % x for x in image_array['boxes'][i]))
				image_array['boxes'][i].tofile(os.path.join(output_directory, "cluster_0_image_%d.txt" % i), ' ')
				if i%100 == 0:
					print('.', end="", flush=True)
			print("\n Dumped %d images" % i)

def _main(args):
	output_dir = os.path.expanduser(args.output_directory)
	if args.no_clear and os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	if args.which == 'generate':
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		if args.text_files:
			generate_txt(args.number_of_images, args.number_of_champs_per_minimap, output_dir)
		else:
			generate(args.number_of_npz_files, args.number_of_images, args.number_of_champs_per_minimap, output_dir)
	elif args.which == 'dump':
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		input_dir = os.path.expanduser(args.input_directory)
		dump_images(input_dir, output_dir)
	else:
		args.print()

##### MAIN: #####
if __name__ == '__main__':
    args = parser.parse_args()
    _main(args)
