# pix2pix-tensorflow server

Host pix2pix-tensorflow models which will accept binary encoded 64 image data for images of size 256 by 256. 

## Exporting

You can export a model to be served with `--mode export`. As with testing, you should specify the checkpoint to use with `--checkpoint`.

```sh
python ../pix2pix.py \
  --mode export \
  --output_dir models/facades \
  --checkpoint ../facades_train
```

## Local Serving

Using the [pix2pix-tensorflow Docker image](https://hub.docker.com/r/affinelayer/pix2pix-tensorflow/):

```sh
# export a model to upload (if you did not export one above)
python ../tools/dockrun.py python tools/export-example-model.py --output_dir models/example
# process an image with the model using local tensorflow
python ../tools/dockrun.py python tools/process-local.py \
    --model_dir models/example \
    --input_file static/facades-input.png \
    --output_file output.png
# run local server
python ../tools/dockrun.py --port 8000 python serve.py --port 8000 --local_models_dir models
# test the local server
python tools/process-remote.py \
    --input_file static/facades-input.png \
    --url http://localhost:8000/example \
    --output_file output.png
```


# serve model locally
cd server
python serve.py --port 8000 --local_models_dir models

# open http://localhost:8000 in a browser, and scroll to the bottom, you should be able to process an edges2cat image and get a bunch of noise as output

# serve model remotely

export GOOGLE_PROJECT=<project name>

# build image
# make sure models are in a directory called "models" in the current directory
docker build --rm --tag us.gcr.io/$GOOGLE_PROJECT/pix2pix-server .

# test image locally
docker run --publish 8000:8000 --rm --name server us.gcr.io/$GOOGLE_PROJECT/pix2pix-server python -u serve.py \
    --port 8000 \
    --local_models_dir models

# run this while the above server is running
python tools/process-remote.py \
    --input_file static/edges2cats-input.png \
    --url http://localhost:8000/edges2cats_AtoB \
    --output_file output.png

# publish image to private google container repository
python tools/upload-image.py --project $GOOGLE_PROJECT --version v1

# create a google cloud server
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars to put your cloud info in there
# get the service-account.json from the google cloud console
# make sure GCE is enabled on your account as well
python terraform plan
python terraform apply

# get name of server
gcloud compute instance-groups list-instances pix2pix-manager
# ssh to server
gcloud compute ssh <name of instance here>
# look at the logs (can take awhile to load docker image)
sudo journalctl -f -u pix2pix
# if you have never made an http-server before, apparently you may need this rule
gcloud compute firewall-rules create http-server --allow=tcp:80 --target-tags http-server
# get ip address of load balancer
gcloud compute forwarding-rules list
# open that in the browser, should see the same page you saw locally

# to destroy the GCP resources, use this
terraform destroy
```