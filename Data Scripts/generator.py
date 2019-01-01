import cv2
import numpy as np
import os
import argparse
import random
import time
from collections import Counter
from PIL import Image
from itertools import combinations 

# Args
parser = argparse.ArgumentParser(
    description="Generate your own synthetic LoL minimap images for training or testing")

subparsers = parser.add_subparsers(help = 'sub-command help')
parser_generate = subparsers.add_parser('generate', help='generate .npz files')
parser_generate.set_defaults(which='generate')
parser_dump = subparsers.add_parser('dump', help='dumps minimap images from .npz file(s)')
parser_dump.set_defaults(which='dump')

parser_generate.add_argument(
    '-ni',
    '--number_of_images',
    help="number of minimap images to generate (Default: 120000)",
    metavar='N',
	type=int,
	default=120000)

parser_generate.add_argument(
    '-cn',
    '--number_of_champs_per_minimap',
    help="number of champions to place on a single minimap image (Default: 30)",
    metavar='N',
	type=int,
	default=30)

parser_generate.add_argument(
    '-npz',
    '--number_of_npz_files',
    help="number of .npz files to create (Default: 8)",
    metavar='N',
	type=int,
	default=8)

parser_dump.add_argument(
    '-i',
    '--input_directory',
    help="the folder from which to read the .npz files (Default: '.')",
	type=str,
	default='.')

parser_dump.add_argument(
    '-o',
    '--output_directory',
    help="the folder to which minimap images are dumped (Default: '.\\images')",
    type=str,
	default='images')

def composite_icons(ten_icon_dict, mapping_dict):
	minimap = Image.open("base_minimap/minimap_295x295.png")
	minimap = minimap.convert('RGBA')
	data    = []

	for champion, value in ten_icon_dict.items():
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
		x_min, x_max = np.min(nz[1]), np.max(nz[1])
		y_min, y_max = np.min(nz[0]), np.max(nz[0])
		bbox = [x_min, y_min, x_max, y_max] 
		cntr = (x_min+ int((x_max-x_min)/2), y_min+int((y_max-y_min)/2))
		data.append((mapping_dict[champion[:-4]], x_min, y_min, x_max, y_max))
	data_array = np.asarray(data)
	return composite, data_array


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
	# Grab 20 random champs from dictionary:
	random_champions = random.sample(icon_list, number_of_champs_per_minimap)
	random_champions_dict = {}
	for champion in random_champions:
		champion_frame = icon_dict[champion]
		sample = random.randint(21,25)
		sampled_champion_frame = champion_frame.resize((sample, sample),resample=Image.BILINEAR)
		sampled_champion_frame.resize(champion_frame.size, Image.NEAREST)
		random_champions_dict[champion] = (sampled_champion_frame, random.randint(5, 260), random.randint(5, 260))
	return random_champions_dict

def generate(number_of_npz_files, number_of_images, number_of_champs_per_minimap):
	print("Generating %d minimap(s) across %d .npz file(s) with %d champions per minimap" % (number_of_images, number_of_npz_files, number_of_champs_per_minimap))
	icon_dict, icon_list, mapping_dict, number_of_champs = load_icons("minimap_icons_28x28")
	max_num_per_champ = int(number_of_images / number_of_champs)
	for j in range(number_of_npz_files):
		start_time = time.time()
		champion_counter = Counter()
		# Split minimap generation across number_of_npz_files
		for i in range(int(number_of_images / number_of_npz_files)):
			# Dynamically reduced icon_list cannot be less than number_of_champs_per_minimap champions:
			if len(icon_list) >= number_of_champs_per_minimap:
				random_icon_dict = randomize_icons(icon_dict, icon_list, number_of_champs_per_minimap)
				composite_image, icon_bbox = composite_icons(random_icon_dict, mapping_dict)
				composite_image = composite_image.convert('RGB')

				# Hack creates stackable array of arrays :
				composite_array = np.array(np.asarray(composite_image))
				icon_bbox       = np.array(np.asarray(icon_bbox))

				# Hack to create balanced classes - UGLY but works fine for now.
				# TODO: Balanced data set creation needs to be cleaned up.
				for champ_number, count in champion_counter.items():
					if count >= max_num_per_champ:
						for champ_name, mapped_number in mapping_dict.items():
							if champ_number == mapped_number:
								file_name = champ_name + '.png'
								if file_name in icon_list:
									icon_list.remove(file_name)

				# Print out stats every 100th iteration:
				if i%100 == 0:
					print('.', end="", flush=True)
				champion_counter[icon_bbox[0][0]]+=1

				# NPZ files generated:
				if i == 0:
					stacked_composite_array = np.copy([composite_array])
					stacked_icon_bbox = np.copy([icon_bbox])
				else:
					composite_array         = np.copy([composite_array])
					stacked_composite_array = np.append(stacked_composite_array, composite_array, 0) 
					icon_bbox         = np.copy([icon_bbox])
					stacked_icon_bbox = np.append(stacked_icon_bbox, icon_bbox, 0)			

		outfile = "data_" + str(j) + ".npz"
		np.savez(outfile, images=stacked_composite_array, boxes=stacked_icon_bbox)
		print("\nFile:", outfile, "generated in", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), "with", i, "minimap(s), from", len(icon_list), "champion(s) with", len(champion_counter.keys()), "unique champion(s)")
		print(champion_counter, "\n")

def dump_images(input_directory, output_directory):
	for file in os.scandir(input_directory):
		if file.name[-4:] == ".npz":
			print("Dumping: " + file.path)
			image_array = np.load(file.path)
			for i in range(image_array['images'].shape[0]):
				cv2.imwrite(os.path.join(output_directory, "cluster_0_image_%d.png" % i), cv2.cvtColor(image_array['images'][i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
				if i%100 == 0:
					print('.', end="", flush=True)
			print("\n")

def _main(args):
	if args.which == 'generate':
		generate(args.number_of_npz_files, args.number_of_images, args.number_of_champs_per_minimap)
	elif args.which == 'dump':
		if not os.path.exists(os.path.expanduser(args.output_directory)):
			os.mkdir(os.path.expanduser(args.output_directory))
		dump_images(os.path.expanduser(args.input_directory), os.path.expanduser(args.output_directory))
	else:
		args.print()

##### MAIN: #####
if __name__ == '__main__':
    args = parser.parse_args()
    _main(args)
