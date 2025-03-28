import torch
import numpy as np
from muq import MuQ
from transformers import AutoTokenizer, RobertaModel
import gensim.downloader as api
import fasttext
from librosa import resample
from transformers import ClapModel, ClapProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def muq(wavs, sr):
    """
    Process multiple audio samples with MuQ model.
    
    Args:
        wavs: List of audio samples
        sr: Sampling rate of input audio samples
        
    Returns:
        torch.Tensor: Stacked embeddings for all valid samples
    """
    device = 'cuda'
    embeddings = []
    target_sr = 24000
    
    # Load MuQ model once for the whole batch
    model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    model = model.to(device).eval()
    
    for i, wav in enumerate(wavs):
        # Convert stereo to mono by averaging channels
        if len(wav.shape) == 2:
            wav = wav.mean(axis=1)
            
        # Resample if needed
        if sr != target_sr:
            wav = resample(wav, orig_sr=sr, target_sr=target_sr)
            
        # Skip if too short
        if wav.size < 1024:
            print(f"Signal {i} too short for MuQ, skipping...")
            continue
            
        # Convert to torch tensor and move to device
        wav_tensor = torch.tensor(wav).unsqueeze(0).to(device)
        
        # Run model inference
        try:
            with torch.no_grad():
                output = model(wav_tensor, output_hidden_states=True)
                
            # Remove batch dimension
            embedding = output.last_hidden_state.squeeze()
            
            # Handle different shapes
            if embedding.dim() > 1:
                # Average across time
                embedding = torch.mean(embedding, dim=0).flatten()
                
            # Move to CPU and append to results
            embeddings.append(embedding.cpu())

            # print(f"MuQ: Generated embedding with with shape {embedding.shape}")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No valid embeddings were generated")
        
    # Stack all embeddings into a single tensor
    embeddings = torch.stack(embeddings)

    return embeddings


def RoBERTa(text):
    """
    Generate word-level embeddings using RoBERTa by aggregating subword embeddings.

    Args:
        text (str): Input text.

    Returns:
        torch.Tensor: Word-level embeddings (one embedding per word).
    """
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = RobertaModel.from_pretrained("FacebookAI/roberta-base").to(device)

    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        padding=True,
        truncation=True
    ).to(device)

    # Get the token embeddings
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (num_tokens, hidden_size)

    # Get the tokenized input (subwords)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))

    # Map subwords back to words using the 'Ġ' prefix
    word_embeddings = []
    current_word_embedding = []
    for i, token in enumerate(tokens):
        if token.startswith("Ġ") or (i == 0 and not token.startswith("Ġ")):  # Start of a new word
            # Aggregate the embeddings for the previous word
            if current_word_embedding:
                word_embeddings.append(torch.mean(torch.stack(current_word_embedding), dim=0))
                current_word_embedding = []

        # Add the subword embedding to the current word
        current_word_embedding.append(token_embeddings[i])

    # Add the last word's embedding
    if current_word_embedding:
        word_embeddings.append(torch.mean(torch.stack(current_word_embedding), dim=0))

    # Stack the word embeddings into a single tensor
    word_embeddings = torch.stack(word_embeddings)

    return word_embeddings.detach().cpu()


def fastText(text):
    model = fasttext.load_model('fastText/cc.en.300.bin')
    embeddings = []
    for word in text.split():
        emb = model.get_word_vector(word)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    embeddings = torch.from_numpy(embeddings)
    return embeddings


def word2vec(text):
    # Download and load the pre-trained Word2Vec model
    w2v_model = api.load('word2vec-google-news-300')
    words = text.split()
    vecs = []
    indices = []
    for i, word in enumerate(words):
        # Retrieve the word vector
        try:
            word_vector = w2v_model[word]
            vecs.append(word_vector)
            indices.append(i)  # Append the index for valid words
        except KeyError:
            print("Word '{}' not found in vocabulary".format(word))
            continue
    vecs = np.stack(vecs)
    print("word2vec shape: ", vecs.shape)
    return torch.from_numpy(vecs), indices


def load_CLAP():
    # Load the CLAP model and processor
    model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    return model, processor

def CLAP_sound(wavs, model, processor, sr):
    target_sr = 48000
    if sr != target_sr:
        wavs = [resample(wav, orig_sr=sr, target_sr=target_sr) for wav in wavs]
        sr = target_sr
    inputs = processor(audios=wavs, return_tensors="pt", sampling_rate=sr).to(device)
    audio_embed = model.get_audio_features(**inputs)
    return audio_embed.detach().cpu()

def CLAP_text(text, model, processor):
    # text should be a list of words
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    text_embeddings = model.get_text_features(**inputs)
    return text_embeddings.detach().cpu()
