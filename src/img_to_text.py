from transformers import pipeline

from src import llm_run


def img2text(path, history=[]):
    image_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large')
    text = image_to_text(path)[0]["generated_text"]
    return llm_run.story_run(text, history)
