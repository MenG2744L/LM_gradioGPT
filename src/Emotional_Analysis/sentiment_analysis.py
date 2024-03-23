from transformers import pipeline

from src import llm_run


def audio_sentiment(path) -> dict:
    audio_to_list = pipeline('audio-classification', model='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    result = audio_to_list(path)
    output_dict = {}
    for item in result:
        label = item['label']
        score = item['score']
        output_dict[label] = score
    return output_dict


def img_sentiment(path) -> dict:
    audio_to_list = pipeline('image-classification', model='ntnxx2/vit-base-patch16-224-finetuned-Visual-Emotional')
    result = audio_to_list(path)
    output_dict = {}
    for item in result:
        label = item['label']
        score = item['score']
        output_dict[label] = score
    return output_dict
