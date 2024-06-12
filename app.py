from flask import Flask, request, jsonify
import torch
from transformers import HubertForSequenceClassification, HubertConfig, Wav2Vec2FeatureExtractor
import torch
import soundfile as sf
from flask_cors import CORS
import gdown
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*", "methods": ["POST"], "headers": ["Content-Type"]}})

# Load your fine-tuned model

file_id = '1xm9Uf7_wn3VR2ivuftCW0jkz5bDC0YxF'
model_name= 'model_hubert_finetuned_nopeft.pth'
if not os.path.exists(model_name):
    print(f"Downloading {model_name} from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_name, quiet=False)
else:
    print(f"output already exists, skipping download.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_name = "model_hubert_finetuned_nopeft.pth"  # Replace with your model path or Hugging Face model hub path
config = HubertConfig.from_pretrained("superb/hubert-large-superb-er")
config.id2label = {0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'disgust', 5: 'surprised', 6: 'fear', 7: 'calm'}
config.label2id = {"neutral": 0, "happy": 1, "angry": 2, "sad": 3, "disgust": 4, "surprised": 5, "fear": 6, "calm": 7}
config.num_labels = 8  # Set it to the number of classes in your SER task

# Load the pre-trained model with the modified configuration
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er", config=config, ignore_mismatched_sizes=True)
model.to(device)
checkpoint =torch.load(model_name, map_location = device)
model.load_state_dict(checkpoint)


model.eval()

# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['file']
    print(data)
    audio_input, sampling_rate = sf.read(data)
    
    # Preprocess audio input
    inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to('cuda' if torch.cuda.is_available() else 'cpu') for key, value in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Display prediction
    predictions = config.id2label[predicted_class]
    # predictions = {
    #     Predicted_class: {config.id2label[predicted_class]},
    #     Class_probabilities: {probabilities}
    # }
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)




# import streamlit as st
# from transformers import HubertForSequenceClassification, HubertConfig, Wav2Vec2FeatureExtractor
# import torch
# import soundfile as sf

# # Load model and tokenizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_name = "model_hubert_finetuned_nopeft.pth"  # Replace with your model path or Hugging Face model hub path
# config = HubertConfig.from_pretrained("superb/hubert-large-superb-er")
# config.id2label = {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad', 4: 'dis', 5: 'sur', 6: 'fea', 7: 'cal'}
# config.label2id = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "dis": 4, "sur": 5, "fea": 6, "cal": 7}
# config.num_labels = 8  # Set it to the number of classes in your SER task

# # Load the pre-trained model with the modified configuration
# model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er", config=config, ignore_mismatched_sizes=True)
# model.to(device)
# checkpoint =torch.load(model_name, map_location = device)
# model.load_state_dict(checkpoint)


# model.eval()

# # Load feature extractor
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

# st.title("Speech Emotion Recognition Model")

# uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

# if uploaded_file is not None:
#     # Load audio file
#     audio_input, sampling_rate = sf.read(uploaded_file)
    
#     # Preprocess audio input
#     inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
#     inputs = {key: value.to('cuda' if torch.cuda.is_available() else 'cpu') for key, value in inputs.items()}
    
#     # Get prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probabilities = torch.softmax(logits, dim=-1)
#         predicted_class = torch.argmax(probabilities, dim=1).item()
    
#     # Display prediction
#     st.write(f"Predicted class: {config.id2label[predicted_class]}")
#     st.write(f"Class probabilities: {probabilities}")