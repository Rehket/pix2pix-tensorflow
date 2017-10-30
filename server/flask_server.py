from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import socket
import time
import argparse
import os
import json
import traceback
import threading
import multiprocessing
import random
import sys
import base64


parser = argparse.ArgumentParser()
parser.add_argument("--local_models_dir", help="directory containing local models to serve (either this or --cloud_model_names must be specified)")
parser.add_argument("--cloud_model_names", help="comma separated list of cloud models to serve (either this or --local_models_dir must be specified)")
parser.add_argument("--origin", help="allowed origin")
parser.add_argument("--addr", default="", help="address to listen on")
parser.add_argument("--port", default=8000, type=int, help="port to listen on")
parser.add_argument("--wait", default=0, type=int, help="time to wait for each request")
parser.add_argument("--credentials", help="JSON credentials for a Google Cloud Platform service account, generate this at https://console.cloud.google.com/iam-admin/serviceaccounts/project (select \"Furnish a new private key\")")
parser.add_argument("--project", help="Google Cloud Project to use, only necessary if using default application credentials")
a = parser.parse_args()

jobs = threading.Semaphore(multiprocessing.cpu_count() * 4)
models = {}
ml = None
project_id = None
build_cloud_client = None
UPLOAD_FOLDER = 'C:\\Workspace\\Python_Practice\\images\\upload'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Specifies the max content length of a file upload.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_data, a_model):
    if a_model not in models:
        raise Exception("invalid model")

    variants = models[name]  # "cloud" and "local" are the two possible variants

    input_b64data = base64.urlsafe_b64encode(image_data.read())

    print(input_b64data)

    time.sleep(a.wait)

    output_b64data = None

    if output_b64data is None and "local" in variants and jobs.acquire(blocking=False):
        m = variants["local"]
        try:
            output_b64data = m["sess"].run(m["output"], feed_dict={m["input"]: [input_b64data]})[0]
        finally:
            jobs.release()

    if output_b64data is None:
        raise Exception("too many requests")

    # add any missing padding
    output_b64data += b"=" * (-len(output_b64data) % 4)
    return base64.urlsafe_b64decode(output_b64data)



@app.route('/')
def index():
    return "Hello, World!"


@app.route('/api/image', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if 'model' not in request.form:
            print('No model selected.')
            flash('No file part')
            return redirect(request.url)
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No File')
            flash('No file part')
            return redirect(request.url)
        model = request.form['model']
        print('The Model is ' + model)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            process_image(file, model)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
        <input type=text name=model>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    if a.local_models_dir is None and a.cloud_model_names is None:
        raise Exception("must specify --local_models_dir or --cloud_model_names")

    if a.local_models_dir is not None:
        import tensorflow as tf
        for name in os.listdir(a.local_models_dir):
            if name.startswith("."):
                continue

            print("loading model", name)

            with tf.Graph().as_default() as graph:
                sess = tf.Session(graph=graph)
                saver = tf.train.import_meta_graph(os.path.join(a.local_models_dir, name, "export.meta"))

                saver.restore(sess, os.path.join(a.local_models_dir, name, "export"))

                if sys.version_info.major == 3:
                    input_vars = json.loads(tf.get_collection("inputs")[0].decode("utf-8"))
                    output_vars = json.loads(tf.get_collection("outputs")[0].decode("utf-8"))
                else:
                    input_vars = json.loads(tf.get_collection("inputs")[0])
                    output_vars = json.loads(tf.get_collection("outputs")[0])

                input = graph.get_tensor_by_name(input_vars["input"])
                output = graph.get_tensor_by_name(output_vars["output"])

                if name not in models:
                    models[name] = {}

                models[name]["local"] = dict(
                    sess=sess,
                    input=input,
                    output=output,
                )

    app.run(debug=False)