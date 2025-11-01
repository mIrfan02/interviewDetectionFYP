import keras
import tensorflow as tf
import google.generativeai as genai
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
api_key="#"


genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

def load_model():
    # Load the Keras model
    model = keras.models.load_model('ver2.h5')
    return model

def video_model():
    video_model = tf.keras.models.load_model('FER01.h5')
    return video_model

def fluency_model(transcript):
    prompt=f"""
            Your task is to check the fluency in the transcription of the interviewee interview.
            you should check weather interviewee is confused and is missing the flow or interviewee is normal and relax.
            You should provide feedback in a paragraph.
            transcription:
            {
                transcript
            }

    """
    response=model.generate_content(prompt)
    return response.text


def feedback_model(FaceEmotions,VoiceEmotions):
    prompt=f"""
            your task is to give feedback on interviewee emotion. does the ineterviewee perform well or not and how can interviewee improve.
            analyze the provided Voice and Face emotion at different interval of time.
            voice emotion and face emotion of intervieww and different interval of time is provided.
            you should provide feedback in one paraghraph.
            you should only include the feedback.
            Voice Emotion:
            {
                VoiceEmotions
            }
            Face Emotions:
            {
                FaceEmotions
            }


    """
    response=model.generate_content(prompt)
    print(response.text)
    return response.text

def summarization_model(text):
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    loaded_model = T5ForConditionalGeneration.from_pretrained("t5_model")
    loaded_tokenizer = T5Tokenizer.from_pretrained("t5_model")
    device = torch.device('cpu')
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenized_text = loaded_tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = loaded_model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)
    output = loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output
    pass
