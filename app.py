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
final_labels = None

# Catching environment variables
TASKROOT = os.environ['LAMBDA_TASK_ROOT']
BUCKET = os.environ['BUCKET']
KEY = os.environ['KEY']


def download_model(bucket, key):
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.isfile(location):
        logger.info(f'Downloading {key} from {bucket} bucket to {location}')
        try:
            s3_resource = boto3.resource('s3')
            s3_resource.Object(bucket, key).download_file(location)
            print(f"Model downloaded to '{location}")
        except Exception as e:
            print("An error occurred: ", e)
    else:
        logger.info("Model already downloaded")
    return location


def model_load():
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


def get_key_from_value(input_value, data_dict):
    for key, values in data_dict.items():
        if input_value in values:
            return key
    return None


def filter_from_mask(labels_json, mask_json):
    with open(labels_json, "r") as f:
        json_labs = json.load(f)
        with open(mask_json, "r") as g:
            mask = json.load(g)
    final_labels = {key: get_key_from_value(value, mask) for key, value in json_labs.items()}
    return final_labels


def get_labels(pred, k, masked):
    if 'final_labels' not in globals():
        global final_labels
        if masked == 'y':
            final_labels = filter_from_mask("labels.json", "mask.json")
        else:
            with open("labels.json", "r") as f:
                final_labels = json.load(f)
    labs = pred.topk(k)[1].tolist()[0]
    probs = pred.topk(k)[0].tolist()[0]
    labels = {}
    for i, lab in enumerate(labs):
        final_lab = final_labels[str(lab)]
        if final_lab is not None:
            labels[final_lab] = probs[i]
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
        labels = get_labels(pred, 5, masked='y')
        first = list(labels.items())
        first = first[0] if first else (None, None)
        logger.info(f"Most probable: {first}")
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
