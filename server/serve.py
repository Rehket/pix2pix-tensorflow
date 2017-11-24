from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
parser.add_argument("--local_models_dir", help="directory containing local models to serve (either this or --cloud_model_names must be specified)")
parser.add_argument("--origin", help="allowed origin")
parser.add_argument("--addr", default="", help="address to listen on")
parser.add_argument("--port", default=8000, type=int, help="port to listen on")
parser.add_argument("--wait", default=0, type=int, help="time to wait for each request")

a = parser.parse_args()

jobs = threading.Semaphore(multiprocessing.cpu_count() * 4)
models = {}
ml = None
project_id = None
is_gpu = False



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


def process_Image():
    # Get the time the process starts
    start = time.time()


class Model(object):
    name = None
    description = None
    uses = None
    successes = None
    failures = None

    def __init__(self, _name, _description, _uses, _successes,  _failures):
        self.name = _name
        self.description = _description
        self.uses = _uses
        self.successes = _successes
        self.failures = _failures

    # Get the Model as a dictionary to be serialized to json.
    def toDict(self):
        temp = {}
        temp["name"] = self.name
        temp["description"] = self.description
        temp["uses"] = self.uses
        temp["successes"] = self.successes
        temp["failures"] = self.failures
        return temp


# A class to hold the information pertaining to the neural network. It makes for easy parsing into json.
class Network(object):
    name = None
    models = []
    images_processed = 0
    description = None
    production = False
    startTime = 0

    def __init__(self, _name, _description):
        self.name = _name
        self.description = _description
        self.startTime = time.time()

    # Get a list of dicts containing the models for the network
    def getModelsDict(self):
        temp = []
        for model in self.models:
            temp.append(model.toDict())
        return temp

    # Get the time since the network was running
    def getUptime(self):
        return time.time() - self.startTime

    # Get a dict of the network.
    def toDict(self):
        temp = {}
        temp["name"] = self.name
        temp["images_processed"] = self.images_processed
        temp["description"] = self.description
        temp["uptime"] = self.getUptime()
        temp["production"] = self.production
        temp["models"] = self.getModelsDict()




class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self.end_headers()
            self.wfile.write("OK")
            return

        if self.path == "/models":
            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(getModels(a.local_models_dir).encode("utf-8"))

            return

        if not os.path.exists("static"):
            self.send_response(404)
            return

        '''
                if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("static/index.html", "rb") as f:
                self.wfile.write(f.read())
            return
        '''

        filenames = [name for name in os.listdir("static") if not name.startswith(".")]
        path = self.path[1:]
        if path not in filenames:
            self.send_response(404)
            return

        self.send_response(200)
        if path.endswith(".png"):
            self.send_header("Content-Type", "image/png")
        elif path.endswith(".jpg"):
            self.send_header("Content-Type", "image/jpeg")
        else:
            self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        with open("static/" + path, "rb") as f:
            self.wfile.write(f.read())

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
        start = time.time()

        status = 201
        headers = {}
        if "origin" in self.headers:
            headers = {"access-control-allow-origin": self.headers["origin"]}


        try:
            name = self.path[2:]
            print(name)
            if name not in models:
                raise Exception("invalid model")

            content_len = int(self.headers.get("content-length", "0"))
            if content_len > 1 * 1024 * 1024:
                raise Exception("post body too large")
            input_data = self.rfile.read(content_len)
            input_b64data = base64.urlsafe_b64encode(input_data)

            output_b64data = None

            # Try to queue the image to be processed.
            if output_b64data is None and jobs.acquire(blocking=False):

                try:
                    output_b64data = models[name]["sess"].run(models[name]["output"], feed_dict={models[name]["input"]: [input_b64data]})[0]
                finally:
                    jobs.release()

            if output_b64data is None:
                raise Exception("too many requests")

            output_b64data += b"=" * (-len(output_b64data) % 4)

            output_data = base64.urlsafe_b64decode(output_b64data)

            headers["content-type"] = "image/png"

            body = output_data
            print(output_data)
            successes.incr()

        except Exception as e:
            failures.incr()
            print("exception", traceback.format_exc())
            status = 500
            body = "server error".encode('utf-8')

        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

        print("finished in %0.1fs successes=%d failures=%d" % (time.time() - start, successes.value(), failures.value()))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def main():
    if a.local_models_dir is None:
        raise Exception("must specify --local_models_dir")

    if a.local_models_dir is not None:
        import tensorflow as tf
        if tf.test.is_built_with_cuda():
            is_gpu = True

        # For every model in the models directory, load the model and get it ready to process images.
        for name in os.listdir(a.local_models_dir):
            if name.startswith("."):
                continue

            print("loading model ", name)

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

                models[name] = dict(
                    sess=sess,
                    input=input,
                    output=output,
                )

    print("listening on %s:%s" % (a.addr, a.port))
    ThreadedHTTPServer((a.addr, a.port), Handler).serve_forever()


main()