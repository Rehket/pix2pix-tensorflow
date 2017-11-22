import socket
import time
import argparse
import base64
import os
import json
import traceback
import threading
import multiprocessing
import sys
from flask import Flask

app = Flask(__name__)

# https://github.com/Nakiami/MultithreadedSimpleHTTPServer/blob/master/MultithreadedSimpleHTTPServer.py
if sys.version_info.major == 3:
    # Python 3
    from socketserver import ThreadingMixIn
    from http.server import HTTPServer, BaseHTTPRequestHandler
else:
    # Python 2
    from SocketServer import ThreadingMixIn
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler


socket.setdefaulttimeout(30)

parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", help="directory containing the configuration json")
parser.add_argument("--origin", help="allowed origin")
parser.add_argument("--addr", default="", help="address to listen on")
parser.add_argument("--port", default=8000, type=int, help="port to listen on")
parser.add_argument("--wait", default=0, type=int, help="time to wait for each request")

a = parser.parse_args()

jobs = threading.Semaphore(multiprocessing.cpu_count() * 4)
ml = None
project_id = None

network_model = None


class RateCounter(object):
    def __init__(self, window_us, granularity=1000):
        self.granularity = granularity
        self.width_us = int(window_us) // self.granularity
        self.buckets = [0] * self.granularity
        self.updated = 0
        self.lock = threading.RLock()

    def incr(self, amt=1):
        with self.lock:
            now_us = int(time.time() * 1e6)
            now = now_us // self.width_us

            if now > self.updated:
                # need to clear any buckets between the update time and now
                if now - self.updated > self.granularity:
                    self.buckets = [0] * self.granularity
                    self.updated = now
                else:
                    while self.updated <= now:
                        self.updated += 1
                        self.buckets[self.updated % self.granularity] = 0

            self.buckets[now % self.granularity] += amt

    def value(self):
        with self.lock:
            # update any expired buckets
            self.incr(amt=0)
            return sum(self.buckets)


successes = RateCounter(1 * 60 * 1e6)
failures = RateCounter(1 * 60 * 1e6)

class Status(object):
    def __init__(self, status, message):
        self.status = status
        self.message = message

    def toDict(self):
        temp = {}
        temp['message'] = self.message
        temp['status'] = self.status
        return temp


OK = Status("Ok", "All systems are go.")

# Get the models as a json string.
def getModels(inputDir):
    model_key_vals = []
    my_models = os.listdir(inputDir)
    index = 0
    for model in my_models:
        model_key_vals.append((model.replace("_", " "), model))
        index += 1
    return_dict = dict(model_key_vals)
    return_dict = [return_dict]
    model_json = json.JSONEncoder().encode(return_dict)

    return model_json


# Get Status
def get_Status():
    network_info = json.load(open('network.json'))
    print(network_info)
    print(network_info[0])
    return



# Model Class Object. The Model contains all the variables associated with a model.
class Model(object):

    def __init__(self, _name, _description, _uses, _target):
        self.name = _name
        self.description = _description
        self.uses = _uses
        self.target = _target
        self.session = None
        self.graph = None
        self.saver = None
        self.input_vars = None
        self.output_vars = None
        self.input = None
        self.output = None

    # Get the Model as a dictionary to be serialized to json.
    def toDict(self):
        temp = {}
        temp['name'] = self.name
        temp['description'] = self.description
        temp['uses'] = self.uses
        temp['target'] = self.target
        return temp

    # Initialize the tensor flow graphs. The inputs and outputs of the graph
    # are saved to variables in the model class.
    def init_TF_Grapg(self, _models_dir):
        with tf.Graph().as_default() as self.graph:
            self.session = tf.Session(graph=self.graph)
            self.saver = tf.train.import_meta_graph(os.path.join(_models_dir, self.target, "export.meta"))

            self.saver.restore(self.session, os.path.join(_models_dir, self.target, "export"))

            if sys.version_info.major == 3:  # For use with python 3
                self.input_vars = json.loads(tf.get_collection("inputs")[0].decode("utf-8"))
                self.output_vars = json.loads(tf.get_collection("outputs")[0].decode("utf-8"))
            else:  # For use with python 2
                self.input_vars = json.loads(tf.get_collection("inputs")[0])
                self.output_vars = json.loads(tf.get_collection("outputs")[0])

            self.input = self.graph.get_tensor_by_name(self.input_vars["input"])
            self.output = self.graph.get_tensor_by_name(self.output_vars["output"])

    def run_session(self, _input):

        if _input is not None:  # Make sure we have input.
            if jobs.acquire(blocking=False):     # Get a semaphore to process the job, by passing False,
                                                # in the event that there is not a semaphore available,
                                                # it just gives up.
                try:
                    result = self.session.run(self.output,
                                              feed_dict={self.input: [base64.urlsafe_b64encode(_input)]})[0]
                finally:
                    jobs.release()

                # Add padding
                result += b"=" * (-len(result) % 4)

                self.uses = self.uses + 1

                # Decode the base 64 result
                return base64.urlsafe_b64decode(result)

            else:  # If we were not able to get a semaphore to process the job.
                network_model.add_failure()
                print("Too many requests.")
                return None


# A class to hold the information pertaining to the neural network. It makes for easy parsing into json.
class Network(object):
    name = None  # The name of the nueral network. Does not really do anything.
    models = {}  # The models contained in this neural network. The key for each network is it's target.
    images_processed = 0  # Whenever an image is processed, the counter increments.
    description = None  # The description of the Neural Network
    production = False  # Production or testing network.
    start_time = 0  # The start time for the network
    total_time = 0  # The total uptime for the network.
    is_gpu = False  # Are we using GPU processing for the network.
    successful_uses = 0  # How many images have been successfully translated.
    failed_uses = 0  # How many uses were unsuccessfully translated.
    uses = 0  # What is the total number of uses.

    def __init__(self, _name, _description, _model_dir):
        self.name = _name
        self.description = _description
        self.start_time = time.time()
        self.model_dir = _model_dir

    # Get a list of dicts containing the models for the network
    def getModelsDict(self):
        temp = []
        if self.models is not None:
            for key in self.models:
                temp.append(self.models[key].toDict())
            return temp
        return temp

    # Get the time since the network was running
    def getUptime(self):
        return time.time() - self.start_time

    # Get a dict of the network.
    def toDict(self):
        temp = {}
        temp['name'] = self.name
        temp['images_processed'] = self.images_processed
        temp['description'] = self.description
        temp['uptime'] = self.getUptime()
        temp['production'] = self.production
        temp['models'] = self.getModelsDict()
        temp['uses'] = self.uses
        temp['successful_uses'] = self.successful_uses
        temp['failed_uses'] = self.failed_uses
        return temp

    # Adds a model to the model dict in the network class.
    def add_model(self, _model):
        self.models[_model.name] = _model

    # Creates a new model from a model dict and adds it to the dict of models in the network class.
    def create_model(self, _model):
        self.models[_model['target']] = Model(_model['name'],
                                            _model['description'],
                                            _model['uses'],
                                            _model['target'])

    def init_models(self):
        for value in self.models:
            self.models[value].init_TF_Grapg(self.model_dir)

    def process_image(self, _model, _input):
        print(self.models)
        return self.models[_model].run_session(_input)

    def add_successes(self):
        self.successful_uses = self.successful_uses + 1
        self.uses = self.uses + 1

    def add_failure(self):
        self.failed_uses = self.failed_uses + 1
        self.uses = self.uses + 1


class Handler(BaseHTTPRequestHandler):

    def API_Post(self, urls):
        if urls[0] == "image":
            self.process_image(urls[1])

        else:
            print("Exception", traceback.format_exc())
            status = 500
            body = "Server error".encode('utf-8')

            self.send_response(status)
            for key, value in self.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)


    def process_image(self, target_model):
        start = time.time()
        status = 201
        headers = {}

        if "origin" in self.headers:
            headers = {"access-control-allow-origin": self.headers["origin"]}
        try:
            content_len = int(self.headers.get("content-length", "0"))

            if content_len > 1 * 1024 * 1024:
                raise Exception("Post body too large")

            input_data = self.rfile.read(content_len)

            output_data = network_model.process_image(target_model, input_data)

            headers["content-type"] = "image/png"

            body = output_data

        except Exception as e:
            failures.incr()  # TODO: Switch to network failure counter.
            print("Exception", traceback.format_exc())
            status = 500
            body = "Server error".encode('utf-8')

        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

        print(
            "finished in %0.1fs successes=%d failures=%d" % (time.time() - start, successes.value(), failures.value()))

    def do_GET(self):

        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(json.JSONEncoder().encode(OK.toDict()).encode("utf-8"))
            return

        if self.path == "/models":
            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(getModels("C:\Workspace\pix2pix-tensorflow\Models").encode("utf-8"))

            return

        if self.path == "/network":
            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(json.JSONEncoder().encode(network_model.toDict()).encode("utf-8"))

            return

        self.send_response(404)
        return


    def do_OPTIONS(self):
        self.send_response(200)
        if "origin" in self.headers:
            if a.origin is not None and self.headers["origin"] != a.origin:
                print("invalid origin %s" % self.headers["origin"])
                self.send_response(400)
                return
            self.send_header("access-control-allow-origin", "*")

        allow_headers = self.headers.get("access-control-request-headers", "*")
        self.send_header("access-control-allow-headers", allow_headers)
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.send_header("access-control-max-age", "3600")
        self.end_headers()

    def do_POST(self):
        print(self.path)
        _path = self.path.split('/')
        print(_path)

        if _path[1] == "api":
            self.API_Post(_path[2:])



class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


if __name__ == "__main__":
    if a.config_dir is None:
        raise Exception("must specify --config_dir")

    # Load Config from file.
    model_config = json.load(open(a.config_dir))[0]

    # Check that config was loaded correctly.
    if model_config is not None:
        print(model_config)

    network_model = Network(model_config['name'],
                            model_config['description'],
                            model_config['model_directory'])

    print(network_model.name)
    print(network_model.description)

    if network_model is not None:
        import tensorflow as tf
        # See whether we are using GPU or CPU for TF
        if tf.test.is_built_with_cuda():
            network_model.is_gpu = True

        for model in model_config['models']:
            network_model.create_model(model)

        network_model.init_models()

        # For every model in the models directory, load the model and get it ready to process images.

    #print("listening on %s:%s" % (a.addr, a.port))
    # ThreadedHTTPServer((a.addr, a.port), Handler).serve_forever()
    app.run(host='127.0.0.1', port=8000)


