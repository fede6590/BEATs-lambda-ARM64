# Python Built-Ins:
import logging
import sys
import os
import json

# External Dependencies:
import torch
import torchaudio
import boto3

# Local Dependencies:
from BEATs import BEATs, BEATsConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3')

# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initiating global variables
model = None
json_dict = None

# Environment variables
ENV_PATH = os.environ['LAMBDA_TASK_ROOT']
BUCKET = os.environ['BUCKET']
KEY = os.environ['KEY']


def download_model(bucket='', key=''):
    location = os.path.join(ENV_PATH, os.path.basename(key))
    if not os.path.isfile(location):
        logger.info("Downloading model...")
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, key).download_file(location)
        logger.info(f"Done: {location}")
    else:
        logger.info("Model already downloaded")
    return location


def model_load():
    global model
    if model is None:
        location = download_model(BUCKET, KEY)
        logger.info("Loading model...")
        checkpoint = torch.load(location)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    return model.to(device)


def download_audio(event):
    logger.info("Downloading audio...")
    input_bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    local_input_temp_file = "/tmp/" + file_key.replace('/', '-')
    logger.info(f"Filename: {local_input_temp_file}")
    s3.download_file(input_bucket_name, file_key, local_input_temp_file)
    audio_path = local_input_temp_file
    return audio_path


def pre_process(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        logger.info("Resampling...")
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.to(device)


def get_labels(pred, k):
    if 'json_dict' not in globals():
        global json_dict
        with open("labels.json", "r") as f:
            json_dict = json.load(f)
    labs = pred.topk(k)[1].tolist()[0]
    probs = pred.topk(k)[0].tolist()[0]
    labels = {}
    for i, lab in enumerate(labs):
        labels[json_dict[str(lab)]] = probs[i]
    return labels


def lambda_handler(event, context):
    model = model_load()
    logger.info("Model ready")
    audio_path = download_audio(event)
    data = pre_process(audio_path)
    logger.info("Data ready")
    try:
        with torch.no_grad():
            logger.info("Sending to model...")
            pred = model.extract_features(data, padding_mask=None)[0]
        logger.info("Inference done")
        labels = get_labels(pred, 5)
        logger.info(f"Labels: {labels}")
        return {
            'statusCode': 200,
            'class': labels
        }
    except:
        return {
            'statusCode': 404,
            'class': None
        }
