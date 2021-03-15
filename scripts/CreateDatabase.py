import os
import sys

if not __package__:
	sys.path.append('./')	# for debugging
    sys.path.append('../')

import cv2
import time
import argparse
import traceback
import numpy as np


from model import  OcrHandle



import config
from config import dbnet_max_size


import database.features as df




# import doc_img.toolset
# import doc_img.fsvd
from scripts import bin_utils
# from doc_img.other.vis import draw_candidates
from other.report import HTMLReport
# from doc_img.other.utils import cur_date_str, cur_time_str


ocrhandle = OcrHandle()



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='./picture', help="输入图片文件夹")
parser.add_argument('-a', '--action', default='create_add', choices=['create_add', 'create', 'remove'])
# parser.add_argument('-d', '--database_path', type=str, default='./feature/feature_trial.txt')
parser.add_argument('-p', '--client_port', type=int, default=50051)
parser.add_argument('-n', '--num_of_kernels', type=int, default=4)
args = parser.parse_args()





if not os.path.exists(os.path.dirname(args.database_path)):
	os.mkdir(os.path.dirname(args.database_path))




if not os.path.isdir(args.input_dir):
	print('请输入正确的图片文件夹路径')
	parser.print_help(sys.stderr)
	exit(-1)













def GetFeature(f, fsvd_client, infile, dx, m, n, address, report):
    '''

    '''
	try:
		img_buffer, img = bin_utils.read_img(infile)
		boxes = bin_utils.cut_line(img)
		feature_string = fsvd_client.ext_feature_from_buffer(img_buffer, boxes, infile)
		if feature_string is not None:
			f.write(feature_string)

			report.write_title(infile+" w:%d,h:%d"%(img.shape[1], img.shape[0]))
			with report.indent():
				report.write_image(draw_candidates(img, boxes))
			return True
		return False
	except Exception as e:
		traceback.print_exc()
		return False







def GetFeature2(infile, report):
    '''
    '''
	try:
		img_buffer, img = bin_utils.read_img(infile)
		boxes = bin_utils.cut_line(img)
		feature_string = fsvd_client.ext_feature_from_buffer(img_buffer, boxes, infile)
		if feature_string is not None:
			f.write(feature_string)
			report.write_title(infile+" w:%d,h:%d"%(img.shape[1], img.shape[0]))
			with report.indent():
				report.write_image(draw_candidates(img, boxes))
			return True
		return False
	except Exception as e:
		traceback.print_exc()
		return False






def main():

	if args.action == 'create':
		f = open(args.database_path, 'w')
	elif args.action == 'create_add':
		f = open(args.database_path, 'a')
	elif args.action == 'remove':
		os.remove(args.database_path)
		exit(0)






	# fsvd_client = doc_img.fsvd.FSVDClient(port=args.client_port)
	report = HTMLReport()
	with report.start():
		image_list = os.listdir(args.input_dir)
		image_list = [fn for fn in image_list if fn.split('.')[-1] in ['jpeg', 'jpg', 'png', 'tif', 'bmp', 'gif']]
		num_images = len(image_list)
		print('正在建立数据库...')
		for ind, image_filename in enumerate(image_list):
			img_dir = os.path.abspath(os.path.join(args.input_dir, image_filename))





			# if GetFeature(f, fsvd_client, img_dir, 32, 0, args.num_of_kernels, args.database_path, report):
				# print("[{}/{}] : {} 已提取".format(ind, num_images, img_dir))
			# else:
				# print("[{}/{}] : {} 失败".format(ind, num_images, img_dir))




		print('数据库建立完毕!')

	f.close()

	report.write_to('Database_%s.html'%cur_time_str())






if __name__ == '__main__':	
	main()
