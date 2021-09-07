# coding: utf-8
"""
Created on Fri Sep 03 15:40:29 2021

@author: Jay Satija
"""

from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from flask_mail import Mail, Message
import sqlite3
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import json
import datetime
from firebase_admin import credentials, firestore, initialize_app, storage
from google.cloud import storage as gcp_storage

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = gcp_storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesn't exist

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="minor-project-43e33-edc3fb7b3ae6.json"

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

mail = Mail() # instantiate the mail class

# configuration of mail
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'jaysatija712@gmail.com'
app.config['MAIL_PASSWORD'] = 'ogxjvdvmplfgxewv'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail.init_app(app)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'storageBucket': 'minor-project-43e33.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()
users_ref = db.collection('users')
images_ref = db.collection('images')

# No cacheing at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')

@app.route('/upload_ct.html')
def upload_ct():
   return render_template('upload_ct.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))
            
   # Load TFLite model and allocate tensors.
   interpreter = tf.lite.Interpreter(model_path = "models/inceptionv3_chest.tflite")
    
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
    
   interpreter.allocate_tensors()
    
   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.array(image, dtype=np.float32)
   interpreter.set_tensor(input_details[0]['index'], [image])
    
   interpreter.invoke()
    
   output_data = interpreter.get_tensor(output_details[0]['index'])
    
   probability = output_data[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(inception_chest_pred)
   
   blob_image_name = 'upload_chest ' + str(datetime.datetime.now()) + '.jpg'
   
   try:
      upload_blob(bucket.name, './flask app/assets/images/upload_chest.jpg', blob_image_name)
   except:
      print("Upload Failed...")
      
   firestore_entry = {}
   firestore_entry['id'] = blob_image_name
   firestore_entry['prediction'] = inception_chest_pred
   json_string = json.dumps(firestore_entry)
   json_object = json.loads(json_string)
        
   try:
      images_ref.document(blob_image_name).set(json_object)
   except Exception as e:
      print(e)

   return render_template('results_chest.html', inception_chest_pred=inception_chest_pred)

@app.route('/uploaded_ct', methods = ['POST', 'GET'])
def uploaded_ct():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

   # Load TFLite model and allocate tensors.
   interpreter = tf.lite.Interpreter(model_path = "models/inception_ct.tflite")
    
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
    
   interpreter.allocate_tensors()
    
   image = cv2.imread('./flask app/assets/images/upload_ct.jpg') # read file
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.array(image, dtype=np.float32)
   interpreter.set_tensor(input_details[0]['index'], [image])
    
   interpreter.invoke()
    
   output_data = interpreter.get_tensor(output_details[0]['index'])
    
   probability = output_data[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      inception_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(inception_ct_pred)
   
   blob_image_name = 'upload_ct ' + str(datetime.datetime.now()) + '.jpg'
   
   try:
      upload_blob(bucket.name, './flask app/assets/images/upload_ct.jpg', blob_image_name)
   except:
      print("Upload Failed...")
      
   firestore_entry = {}
   firestore_entry['id'] = blob_image_name
   firestore_entry['prediction'] = inception_ct_pred
   json_string = json.dumps(firestore_entry)
   json_object = json.loads(json_string)
        
   try:
      images_ref.document(blob_image_name).set(json_object)
   except Exception as e:
      print(e)

   return render_template('results_ct.html',inception_ct_pred=inception_ct_pred)
   
  
@app.route('/form.html')
def form():
   return render_template('form.html')
   
@app.route('/show.html')
def show():
   return render_template('show.html')

@app.route('/uploaded_form', methods=["GET", "POST"])
def uploaded_form():
    if request.method == "POST":
        file = open('models/model.pkl', 'rb')
        clf = pickle.load(file)
        file.close()
        myDict = request.form
        email = myDict['email']
        fever = float(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        # Code for Inference
        inputFeatures = [fever, pain, age, runnyNose, diffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        logistic_reg_pred = str('%.2f' % (infProb*100) + '% COVID-19 Infection Probability') 
        print(infProb)
        
        conn = get_db_connection()
        conn.execute('INSERT INTO users (email, fever, bodyPain, age, runnyNose, diffBreath, infectionProb) VALUES (?,?,?,?,?,?,?)',
            (email, fever, pain, age, runnyNose, diffBreath, round(infProb*100, 2)))
        conn.commit()
        conn.close()
        
        firestore_entry = request.form.to_dict()
        firestore_entry['infectionProb'] = round(infProb*100, 2)
        firestore_entry['created'] = str(datetime.datetime.now())
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)
        
        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)
        
        return render_template('show.html', logistic_reg_pred=logistic_reg_pred)
    return render_template('form.html')

@app.route('/thanks.html')
def thanks():
   return render_template('thanks.html')
   
@app.route('/sendemail', methods=["GET", "POST"])
def sendemail():
    if request.method == "POST":
        myDict = request.form.to_dict()
        msg = Message(myDict['subject'], sender='jaysatija712@gmail.com', recipients=['jaysatija712@gmail.com'])
        msg.body = """
        From: %s <%s>
        Phone: %s 
        Discussion: %s
        Message: %s
        """ % (myDict['name'], myDict['email'], myDict['phone'], myDict['discuss'], myDict['message'])
        mail.send(msg)
        print("---Mail Sent---")
        return render_template('thanks.html', user_name = myDict['name'])
    return render_template('contact.html')
  
if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)