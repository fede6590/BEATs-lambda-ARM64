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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_load(model_path):
    logger.info("Loading model...")
    checkpoint = torch.load(model_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


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
    torchaudio.set_audio_backend("soundfile")
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        logger.info("Resampling...")
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform


def get_label(label_pred):
    if 'json_dict' not in globals():
        global json_dict
        with open("labels.json", "r") as f:
            json_dict = json.load(f)
    label = json_dict[str(label_pred)]
    return label


def lambda_handler(event, context):
    if 'model' not in globals():
        # Load model
        model_path = os.path.join(os.environ['LAMBDA_TASK_ROOT'], 'model.pt')
        global model
        model = model_load(model_path)
    logger.info("Model ready")
    # Download .wav
    audio_path = download_audio(event)
    # Pre-process audio
    data = pre_process(audio_path)
    logger.info("Data ready")
    # Classify image
    try:
        with torch.no_grad():
            probs = model.extract_features(data, padding_mask=None)[0]
            label_pred = probs.topk(k=1)[1].tolist()[0][0]
            label = get_label(label_pred)
            logger.info(f"Label: {label}")
        return {
            'statusCode': 200,
            'class': label
        }
    except:
        return {
            'statusCode': 404,
            'class': None
        }
