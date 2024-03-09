
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import os
import requests
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text


def generate_story(scenario):
    template = """
    You are story teller;
    You can generate  a short story based on a simple narrative, the story should be no more than 30 words.
    CONTEXT: {scenario}
    STORY:
    """
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
    payloads = {
        "inputs": scenario
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    return response.json()[0]['generated_text']


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as f:
        f.write(response.content)

def main():
    st.set_page_config(page_title='img to audio story', page_icon='🤗')
    st.header('Turn image into audio story')
    uploaded_file = st.file_uploader('Choose an image...', type='jpg')

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as f:
            f.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander('scenario'):
            st.write(scenario)
        with st.expander('story'):
            st.write(story)
        
        st.audio('audio.flac')

if __name__ == '__main__':
    main()

