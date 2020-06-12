#some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('~/322GDrive/Insight_Project/My_Project/Virtural_TryOn')
#sys.path.append('~/Insight_Project/Virtual_TryOn')
from MyWebApp.static.scripts.utils import random_person, random_cloth

from flask import render_template, request, send_from_directory
import os
from flaskexample import app

from inference import build_model, try_on_database
import time
from collections import OrderedDict
from options.test_options import TestOptions
from dataset.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import numpy as np
import torch
from torch.autograd import Variable
import cv2


config = {'database': 'flaskexample/static/database',
		  'person_url': "flaskexample/static/database/person/000001_0.jpg",
		  'person_folder':'flaskexample/static/database/person',
		  'cloth_folder':'flaskexample/static/database/cloth',
		  'result_folder':'flaskexample/static/database/result',
		  'title':"Virtual TryOn - by Qingyun Wang",
		  'person_name':"",
		  'cloth_name':"",
		  'result_name':""}
opt = TestOptions().parse()
opt.checkpoints_dir = '../checkpoints'
model = create_model(opt)

# config['person_name'] = random_person(config)
# config['cloth_name'] = random_cloth(config)
# person_path = os.path.join('../MyWebApp', config['database'], 'person', config['person_name'])
# cloth_path = os.path.join('../MyWebApp', config['database'], 'cloth', config['cloth_name'])
# result = try_on_database(opt, model, person_path, cloth_path)

# data preprocessing
# def changearm(old_label):
# 	label = old_label
# 	arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int))
# 	arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int))
# 	noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int))
# 	label = label * (1 - arm1) + arm1 * 4
# 	label = label * (1 - arm2) + arm2 * 4
# 	label = label * (1 - noise) + noise * 4
# 	return label

# def data_preprocessing(person_name, cloth_name, config):
# 	data['image'] =
# 	mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
# 	mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
# 	img_fore = data['image'] * mask_fore
# 	img_fore_wc = img_fore * mask_fore
# 	all_clothes_label = changearm(data['label'])
# 	data ={}












# Create the application object
#app = Flask(__name__)

# # load index and model
# input_dataset = ImageDataset('notebooks/data')
# bs = 100
# image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=bs)
# model, model_full = load_pretrained_model()
#
# pd_files = input_dataset.get_file_df()
# annoy_path = 'notebooks/annoy_idx.annoy'
# if os.path.exists(annoy_path):
# 	annoy_idx_loaded = AnnoyIndex(512, metric='angular')
# 	annoy_idx_loaded.load(annoy_path)


@app.route('/')
@app.route('/index')
def index():

	# Pick a random person
	config['person_name'] = random_person(config)
	config['cloth_name'] = random_cloth(config)
	config['result_name'] = os.path.join("../static/database/person", config['person_name'])

	print("person_name", config['person_name'])
	print("cloth_name", config['cloth_name'])
	# person_url = os.path.join(person_folder, person_names[idx])

	return render_template('index.html',
						   person_name = config['person_name'],
						   cloth_name = config['cloth_name'],
						   result_name = config['result_name'],
						   title = config['title'],
						   username = 'chooch')


@app.route('/random_person', methods = ['POST', 'GET'])
def get_person():
	if request.method == 'POST':

		# #Pick a random person
		config['person_name'] = random_person(config)
		print(config['person_name'])
		return render_template('index.html',
							   person_name = config['person_name'],
							   cloth_name = config['cloth_name'],
							   result_name=config['result_name'],
							   title=config['title'],
							   username='chooch')


@app.route('/random_cloth', methods=['POST', 'GET'])
def get_cloth():
	if request.method == 'POST':

		# #Pick a random person

		cloth_name = random_cloth(config)
		config['cloth_name'] = cloth_name
		print("person_name", config['person_name'])
		print("cloth_name", config['cloth_name'])
		# person_url = os.path.join(person_folder, person_names[idx])

		return render_template('index.html',
							   person_name=config['person_name'],
							   cloth_name = config['cloth_name'],
							   result_name=config['result_name'],
							   title=config['title'],
							   username='chooch')

@app.route('/output', methods=['POST', 'GET'])
def get_result():
	if request.method == 'GET':
		person_path = os.path.join(config['person_folder'], config['person_name'])
		cloth_path = os.path.join(config['cloth_folder'], config['cloth_name'])
		result = try_on_database(opt, model, person_path, cloth_path)
		result_name = config['person_name'].replace('.jpg', '+') + config['cloth_name']
		config['result_name'] = result_name

		result_path = os.path.join(config['result_folder'], result_name)
		cv2.imwrite(result_path, result)
		return render_template('index.html',
							   person_name=config['person_name'],
							   cloth_name = config['cloth_name'],
							   result_name=config['result_name'],
							   title=config['title'],
							   username='chooch')

# @app.route('/output')
# def output():
# 	title_text = 'NextPick - by Isaac Chung'
# 	selection = request.args.get("selection")
# 	input_location = request.args.get("input_location")
#
# 	# Case if empty
# 	if selection != " ":
# 		print("..tag not empty")
# 		if selection == "ski":
# 			test_img = 'notebooks/ski-test-img.png'
# 			in_img = "assets/img/ski-test-img.png"
# 		elif selection == "war_mem":
# 			test_img = 'notebooks/test-img-war-mem.jpg'
# 			in_img = "assets/img/test-img-war-mem.jpg"
# 		elif selection == "banff":
# 			test_img = "static/assets/img/banff.jpg"
# 			in_img = "assets/img/banff.jpg"
# 		# searches = eval_test_image(test_img, model, annoy_idx_loaded, top_n=30) # returns more than top 5 for processing
# 		# df = create_df_for_map_plot(searches, pd_files)
# 		# input_latlon = get_input_latlon(input_location)
# 		# df = get_distances(input_latlon, df)
# 		# df = get_top5_distance(df)
# 		# bar = create_plot(df)
#
# 		return render_template("results.html", title=title_text, flag="1", sel_input=selection,
# 							   input_location=input_location, input_pic=in_img
# 							   )
# 		# return render_template("results.html", title=title_text, flag="1", sel_input=selection,
# 		# 					   df=df, plot=bar, input_location=input_location,
# 		# 					   input_latlon=input_latlon, input_pic=in_img
# 		# 					   )
# 	else:
# 		print("..tag empty")
# 	return render_template("index.html",
# 						   title=title_text, flag="0", sel_input=selection,
# 						   sel_form_result="Empty"
# 						   )


# @app.route('/<path:filename>')
# def download_file(filename):
# 	return send_from_directory(DATA_FOLDER, filename, as_attachment=True)


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
