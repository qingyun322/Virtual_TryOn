import sys
import shutil
import time
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.append('~/322GDrive/Insight_Project/My_Project/Virtual_TryOn')
#sys.path.append('~/Insight_Project/Virtual_TryOn')
#sys.path.append('~/322GDrive/Insight_Project/My_Project/Virtural_TryOn')
sys.path.append('/home/ubuntu/Insight_Project/Virtural_TryOn')
from flaskexample.static.scripts.utils import random_person, random_cloth

from flask import render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename

import os
from flaskexample import app

from inference import build_model, try_on
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
from PIL import Image

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


config = {'database': 'flaskexample/static/database',
          'person_url': "flaskexample/static/database/person/000001_0.jpg",
	  #'person_folder':'/home/ubuntu/Insight_Project/Virtural_TryOn/application/flaskexample/static/database/person',
          	  'person_folder':'./application/flaskexample/static/database/person',
	  'cloth_folder':'./application/flaskexample/static/database/cloth',
	  'result_folder':'./application/flaskexample/static/database/result',
	  'title':"Virtual TryOn - by Qingyun Wang",
	  'person_name':"",
	  'cloth_name':"",
	  'result_name':"",
          'from_user': False,
          'person_index': 0,
          'cloth_index': 0
         }

opt = TestOptions().parse()
opt.checkpoints_dir = './checkpoints'
model = create_model(opt)

@app.route('/')
@app.route('/index')
def index():

	# Pick a random person
	config['person_name'] = random_person(config)
	config['cloth_name'] = random_cloth(config)
	config['result_name'] = 'blank.jpg'
	print("person_name", config['person_name'])
	print("cloth_name", config['cloth_name'])
	# person_url = os.path.join(person_folder, person_names[idx])

	return render_template('index.html',
						   person_name = config['person_name'],
						   cloth_name = config['cloth_name'],
						   result_name = config['result_name'],
						   title = config['title'],
						   username = 'chooch')



@app.route('/person_upload', methods=['POST', 'GET'])
def upload_person():
	if request.method == 'POST':
		if 'person_name' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['person_name']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
                        config['from_user'] = True
                        config['person_name'] = str(config['person_index']) + '_user_person.jpg'
                        filename = secure_filename(file.filename)
                        person_url = os.path.join(config['person_folder'], config['person_name'])
                        file.save(person_url)
                        time.sleep(1)
                        person = Image.open(person_url).convert('RGB')
                        person.resize((192, 256))
                        person.save(person_url)
                        print(config)
                        config['person_index'] += 1
                        if config['person_index'] == 10:
                                config['person_index'] = 0
                        shutil.copyfile(person_url, os.path.join('../openpose/examples/AppIn', config['person_name']))
		return render_template('index.html',
							   person_name=config['person_name'],
							   cloth_name=config['cloth_name'],
							   result_name=config['result_name'],
							   title=config['title'],
							   username='chooch')






@app.route('/random_person', methods = ['POST', 'GET'])
def get_person():
	if request.method == 'POST':

		# #Pick a random person
		config['person_name'] = random_person(config)
		config['from_user'] = False
		return render_template('index.html',
							   person_name = config['person_name'],
							   cloth_name = config['cloth_name'],
							   result_name=config['result_name'],
							   title=config['title'],
							   username='chooch')


@app.route('/cloth_upload', methods = ['POST', 'GET'])
def upload_cloth():
	if request.method == 'POST':
		print(request.files)
		if 'shirt_name' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['shirt_name']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
                        config['from_user'] = True
                        config['cloth_name'] = str(config['cloth_index']) + '_user_cloth.jpg'
                        filename = secure_filename(file.filename)
                        cloth_url = os.path.join(config['cloth_folder'], config['cloth_name'])
                        file.save(cloth_url)
                        time.sleep(1)
                        cloth = Image.open(cloth_url).convert('RGB')
                        cloth.resize((192, 256))
                        cloth.save(cloth_url)
                        config['cloth_index'] += 1
                        if config['cloth_index'] == 10:
                            config['cloth_index'] = 0
		return render_template('index.html',
							   person_name=config['person_name'],
							   cloth_name=config['cloth_name'],
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
		if not config['from_user']:
                        pose_path = person_path.replace('.jpg', '_keypoints.json').replace('person', 'person_pose')
                        result = try_on(opt, config['person_name'], model, person_path, cloth_path, pose_path, config['from_user'])
		else:
                        print(config['person_name'])
                        os.chdir("../openpose")
                        os.system("./build/examples/openpose/openpose.bin --image_dir examples/AppIn --write_json examples/AppOut/ -model_pose='COCO' --display 0 --render_pose 0 ")
                        os.chdir("../Virtural_TryOn")
                        pose_path = os.path.join('../openpose/examples/AppOut/', config['person_name'].replace('.jpg', '_keypoints.json'))
                        result = try_on(opt, config['person_name'], model, person_path, cloth_path, pose_path, config['from_user'])
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
#if __name__ == "__main__":
#	app.run(debug=True) #will run locally http://127.0.0.1:5000/
