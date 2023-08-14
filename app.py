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

# Logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Environment variables
TASKROOT = os.environ['LAMBDA_TASK_ROOT']
BUCKET = os.environ['BUCKET']
KEY = os.environ['KEY']

# Global variables
s3 = boto3.client('s3')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_labels = None


def download_model(bucket, key):
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.isfile(location):
        logger.info(f'Downloading {key} from {bucket} bucket to {location}')
        try:
            s3_resource = boto3.resource('s3')
            s3_resource.Object(bucket, key).download_file(location)
            logger.info(f"Model downloaded to {location}")
        except Exception as e:
            logger.error("An error occurred while downloading model: %s", e)
    else:
        logger.info("Model already downloaded")
    return location


def load_model():
    global model
    if model is None:
        location = download_model(BUCKET, KEY)
        logger.info(f"Loading model to {device}...")
        checkpoint = torch.load(location)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    return model.to(device)

 
# Loading model as global variable
model = load_model()
logger.info("Model ready")


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


def load_json_file(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def filter_from_mask(labels_json, mask_json):
    labels = load_json_file(labels_json)
    mask = load_json_file(mask_json)
    final_labels = {key: mask.get(value) for key, value in labels.items()}
    return final_labels


def get_labels(pred, k, masked):
    if masked == 'y':
        final_labels = filter_from_mask("labels.json", "mask.json")
    else:
        final_labels = load_json_file("labels.json")
    labs = pred.topk(k)[1].tolist()[0]
    probs = pred.topk(k)[0].tolist()[0]
    labels = {}
    for lab, prob in zip(labs, probs):
        final_lab = final_labels.get(str(lab))
        if final_lab is not None:
            labels[final_lab] = prob
    return labels


def lambda_handler(event, context):
    try:
        audio_path = download_audio(event)
        data = pre_process(audio_path)
        logger.info("Data ready")

        with torch.no_grad():
            logger.info("Sending to model...")
            pred = model.extract_features(data, padding_mask=None)[0]
        logger.info("Inference done")

        labels = get_labels(pred, 5, masked='y')
        first = next(iter(labels.items()), (None, None))

        logger.info(f"Most probable: {first}")
        logger.info(f"Labels: {labels}")

        return {
            'statusCode': 200,
            'class': labels
        }
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return {
            'statusCode': 500,
            'class': None
        }
