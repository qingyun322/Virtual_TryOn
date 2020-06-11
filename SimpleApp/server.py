import os
from flask import Flask, render_template, request, flash, redirect, request, url_for

UPLOAD_FOLDER = 'static/assets/img/user_input'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Create the application object
app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload', methods=["GET", "POST"])  # we are now using these methods to get user input
# def upload_file():
#     if request.method == 'POST':
#         f1 = request.files['file0']
#         f2 = request.files['file1']
#         f1.save('static/assets/img/user_input')
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file',
#                                     filename=filename))
#     return render_template('index.html')

def home_page():
	return render_template('index.html')  # render a template


@app.route('/output', methods=["GET", "POST"])



def recommendation_output():

	#
	# Pull input
	#some_input = request.args.get('user_input')
	if 'file0' not in request.files:
		some_input = 0
	else:
		some_input = 1
	# Case if empty
	if some_input == 0:
		return render_template("index.html",
							   my_input=some_input,
							   my_form_result="Empty")
	else:
		some_input = "yeay!"
		some_number = len(request.files)
		some_image = "assets/img/000001_0.jpg"
	return render_template("index.html",
						   my_input=some_input,
						   #my_output=some_output,
						   my_number=some_number,
						   my_img_name=some_image,
						   my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True)  # will run locally http://127.0.0.1:5000/

