import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertModel,DistilBertTokenizer
import warnings
warnings.filterwarnings('ignore')

# read our dataset with images paths
df1 = pd.read_csv('sample.csv')
#we import the dataset
df2 = pd.read_csv('datatext.csv')

#we load the model and the tokenizer
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#we use the tfidfvectorizer to get the vectors of our corpus
tfidf = TfidfVectorizer(stop_words = 'english')

def sentence_embeddings(model,tokenizer,sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=187)
    #we get the sentence embeddings
    with torch.no_grad():
        output = model(**inputs)
    # Extract the output of the [CLS] token (first token)
    sentence_embedding = output.last_hidden_state[:, 0, :]
    return sentence_embedding

def process_text(text,method):
    if method == "Embeddings":
        vector = sentence_embeddings(model,tokenizer,text).numpy().flatten()
        vector = vector.tolist()
        response = requests.post('http://annoy-db2:5006/reco', json={'vector': vector})
        if response.status_code == 200:
            indices = response.json()
            # Retrieve paths for the indices
            paths = df2.title.iloc[indices].tolist()
            result = "\n".join(paths)
            return result
        else:
            return "Error in API request"
    elif method == "Bag of Words":
        tfidf = TfidfVectorizer(stop_words = 'english')
        vector = tfidf.fit_transform([text]).toarray().flatten()
        vector = np.pad(vector, (0, 15134 - len(vector)), mode='constant')
        vector = vector.tolist()
        response = requests.post('http://annoy-db1:5005/reco', json={'vector': vector})
        if response.status_code == 200:
            indices = response.json()
            paths = df2.title.iloc[indices].tolist()
            result = "\n".join(paths)
            return result
        else:
            return "Error in API request"


def transform(img):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),normalize])
    return transform(img)
def plot_image(path):
    img = Image.open(path)
    plt.axis('off')
    plt.imshow(img)

def plot_images(image_paths):
    num_images = len(image_paths)
    num_rows = 1
    num_cols = num_images
    fig = plt.figure(figsize=(15, 16))  
    for i, path in enumerate(image_paths, start=1):
        plt.subplot(num_rows, num_cols, i)
        plot_image(path)
    #plt.show()
    return fig


def process_image(picture):
    model = torch.load('model.pth')
    image = Image.fromarray(np.uint8(picture))
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        vector = model(image)
        # Convert the numpy array to a JSON serializable format
        vector = vector.squeeze(0)
        vector_json = vector.tolist()
    # Now we send the vector to the API
    response = requests.post('http://annoy-db:5000/reco', json={'vector': vector_json})
    if response.status_code == 200:
        indices = response.json()
        # Retrieve paths for the indices
        paths = df1.path.iloc[indices]
        fig,axs = plt.subplots(1,len(paths),figsize=(10*len(paths),18))
        for i,path in enumerate(paths):
           img = Image.open(path)
           axs[i].imshow(img)
           axs[i].axis('off')
        return fig
    else:
        return "Error in API request"

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot",title ="My Recommendation System",description="Please select a movie to get  some recommendations")
demo = gr.Interface(
    fn=process_text,
    inputs=["text",gr.Radio(["Embeddings", "Bag of Words"])],
    outputs=gr.Textbox(label="Movie Recommendation", lines=5),
    title ="My Recommendation System",
    description = "Please provide the description of the kind of movie you want to watch"
)

gr.TabbedInterface(
    [iface, demo], ["Recommended System based images", "Recommended System based NLP"]
).launch(server_name="0.0.0.0") #the server will be accessible under this address externally