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
import random
import sys
import base64

# https://github.com/Nakiami/MultithreadedSimpleHTTPServer/blob/master/MultithreadedSimpleHTTPServer.py
try:
    # Python 2
    from SocketServer import ThreadingMixIn
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    # Python 3
    from socketserver import ThreadingMixIn
    from http.server import HTTPServer, BaseHTTPRequestHandler

socket.setdefaulttimeout(30)

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
cloud_requests = RateCounter(5 * 60 * 1e6)
cloud_accepts = RateCounter(5 * 60 * 1e6)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write("OK")
            return

        if not os.path.exists("static"):
            self.send_response(404)
            return

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("static/index.html", "rb") as f:
                self.wfile.write(f.read())
            return

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
            self.send_header("access-control-allow-origin", self.headers["origin"])

        allow_headers = self.headers.get("access-control-request-headers", "*")
        self.send_header("access-control-allow-headers", allow_headers)
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.send_header("access-control-max-age", "3600")
        self.end_headers()


    def do_POST(self):
        start = time.time()

        status = 200
        headers = {}
        if "origin" in self.headers:
            headers = {"access-control-allow-origin": self.headers["origin"]}
        body = ""

        try:
            url = self.path.strip("\n").split("/")
            print(url[1])
            name = url[1]

            print(self)

            variants = models[name]  # "cloud" and "local" are the two possible variants

            content_len = int(self.headers.get("content-length", "0"))
            if content_len > 1 * 1024 * 1024:
                raise Exception("post body too large")
            input_data = self.rfile.read(content_len)
            input_b64data = base64.urlsafe_b64encode(input_data)

            time.sleep(a.wait)

            output_b64data = None

            # add any missing padding

        except Exception as e:
            failures.incr()
            print("exception", traceback.format_exc())
            status = 500
            body = "server error"

        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

        print("finished in %0.1fs successes=%d failures=%d" % (time.time() - start, successes.value(), failures.value()))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def main():

    print("listening on %s:%s" % (a.addr, a.port))
    ThreadedHTTPServer((a.addr, a.port), Handler).serve_forever()

main()