# Python Built-Ins:
import logging
import sys
import os

# External Dependencies:
import torch
import torchaudio
import boto3

# Local Dependencies:
from model.BEATs import BEATs, BEATsConfig

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
model = None


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


def load_model(location):
    global model
    if model is None:
        logger.info(f"Loading model to {device}...")
        checkpoint = torch.load(location)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        logger.info("Model already loaded")
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
    logger.info(f"Sample rate = {sr}")
    if sr != 16000:
        logger.info("Resampling...")
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.to(device)


def get_label(label_pred):
    indices_list = label_pred[1][0].tolist()
    for value in indices_list:
        if value in [20, 404, 520, 151, 515, 522, 429, 199, 50, 433, 344, 34, 413, 244, 155, 245, 242]:
            return "Speech"
        elif value in [284, 19, 473, 498, 395, 81, 431, 62, 410]:
            return "Baby Crying"
        elif value in [323, 149, 339, 480, 488, 400, 150, 157]:
            return "Dog"
        elif value in [335, 221, 336, 277]:
            return "Cat"
        else:
            return "No Value"


def lambda_handler(event, context):
    try:
        location = download_model(BUCKET, KEY)
        model = load_model(location)
        logger.info("Model ready")
        audio_path = download_audio(event)
        data = pre_process(audio_path)
        logger.info("Data ready")

        with torch.no_grad():
            logger.info("Sending to model...")
            pred = model.extract_features(data, padding_mask=None)[0]
        logger.info("Inference done")

        label_pred = pred.topk(k=5)
        label = get_label(label_pred)
        logger.info(f"Label: {label}")

        return {
            'statusCode': 200,
            'class': label,
            'filename': event['Records'][0]['s3']['object']['key']
        }
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return {
            'statusCode': 500,
            'class': None
        }
