# Set OMP_NUM_THREADS to 1 to avoid KMeans memory leak on Windows
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
import torch
from itertools import combinations
import pickle
import random

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from scipy.spatial import distance
from scipy.stats import wasserstein_distance as scipy_wasserstein

from models import muq, RoBERTa, word2vec, fastText, CLAP_text, CLAP_sound, load_CLAP
from util import Parameter, ParameterGenerator, load_soundfiles, remove_silence, set_sampling_rate


SAMPLING_RATE = 44100
CROSSFADE_DURATION = 0.01  # 10ms crossfade
OUTPUT_PATH = Path("./output")
CACHE_PATH = Path("./cache")

set_sampling_rate(SAMPLING_RATE)


def preprocess_sounds(sounds, slice_fn, trim_silence=True):
    if trim_silence:
        sounds = remove_silence(sounds)
    processed_sounds = []
    for sound in sounds:
        slices = slice_fn(sound)
        for slice in slices:
            processed_sounds.append(slice)
    return processed_sounds


# what transformations will we apply to the feature spaces?
def create_pipeline(normalization_method, dim):
    pca = PCA(n_components=dim)
    return [normalization_method.fit_transform, pca.fit_transform]


def load_text_corpus(path):
    assert path.is_file()
    with open(path, "r") as f:
        return [word for line in f.readlines() for word in line.split()]
    

def generate_sound_cache_filename(corpus_name, slice_fn_name, encoder_name):
    """
    Generate a unique filename for sound embeddings based on corpus name, slice function, and encoder name.
    """
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return CACHE_PATH / f"sound_{corpus_name}_{slice_fn_name}_{encoder_name}.pkl"

def generate_text_cache_filename(text_file_name, encoder_name):
    """
    Generate a unique filename for text embeddings based on text file name and encoder name.
    """
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return CACHE_PATH / f"text_{text_file_name}_{encoder_name}.pkl"


def embed_sounds(sounds, encoder, cache_file=None):
    # Check if cache_file is provided and exists
    if cache_file is not None and cache_file.exists():
        print(f"Loading sound embeddings from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
            return embeddings
    
    # Batch process sounds
    embeddings = encoder(sounds, SAMPLING_RATE)
    
    if cache_file is not None:
        # Save embeddings to cache
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
    
    return embeddings


def embed_text(text, encoder, cache_file=None):
    # if cache_file is provided, load the embeddings from there
    if cache_file is not None and cache_file.exists():
        print(f"Loading text embeddings from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    embeddings = encoder(text)

    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings


def find_nearest_neighbors(S, points, distance_metric):
    """

    :param S: The space S of all sound embeddings
    :param points: A list of points that you want to find the nearest neighbors in S
    :param distance_metric: 'euclidean' or 'cosine'
    :return: list of (the nearest point, the idx of the nearest point)
    """
    neigh = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=distance_metric).fit(S)
    neighbor_indices = neigh.kneighbors(points, return_distance=False).flatten()
    return S[neighbor_indices], neighbor_indices


def apply_crossfade(sound1, sound2, crossfade_samples):
    """Crossfade between two sounds over the given number of samples."""
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    # Apply fades
    sound1[-crossfade_samples:] *= fade_out
    sound2[:crossfade_samples] *= fade_in
    
    # Concatenate with overlap
    return np.concatenate([sound1[:-crossfade_samples], sound1[-crossfade_samples:] + sound2[:crossfade_samples], sound2[crossfade_samples:]])


def save_output(sound_list, filename, normalize=False):
    if normalize:
        sound_list = [sound / np.max(np.abs(sound)) for sound in sound_list]
        filename = filename.parent / f"{filename.stem}_normalized{filename.suffix}"

    crossfade_samples = int(CROSSFADE_DURATION * SAMPLING_RATE)
    output = sound_list[0]

    for i in range(1, len(sound_list)):
        output = apply_crossfade(output, sound_list[i], crossfade_samples)
    
    if normalize:
        output = output / np.max(np.abs(output))  # normalize

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    counter = 1
    if filename.exists():
        while Path(f"{filename.stem}_{counter}{filename.suffix}").exists():
            counter += 1
        filename = filename.parent / f"{filename.stem}_{counter}{filename.suffix}"

    sf.write(filename, output, SAMPLING_RATE)

#################
# SLICE METHODS #
#################

def entire_sample(sound):
    return sound


def equal_slices(sound, grain_size):
    if sound.size < grain_size:
        return [sound]
    return librosa.util.frame(sound, frame_length=grain_size, hop_length=grain_size, axis=0)


def get_onsets(y):
    # Detect onsets
    inter_onset_time = 10  # min number of frames between onsets
    onset_samples = librosa.onset.onset_detect(y=y, sr=SAMPLING_RATE, units='samples', backtrack=True, wait=inter_onset_time)

    # Split audio at onsets
    segments = []
    for i in range(len(onset_samples) - 1):
        segment = y[onset_samples[i]:onset_samples[i + 1]]
        segments.append(segment)

    # Add final segment (after last onset to end)
    if onset_samples[-1] < len(y):
        segments.append(y[onset_samples[-1]:])

    return segments


####################
# EVLAUATION METRICS
####################


def evaluate_clustering(X, labels):
    eval = {}

    # Check if we have enough samples in each cluster for silhouette score
    unique_labels = np.unique(labels)
    sample_counts = [np.sum(labels == label) for label in unique_labels]
    
    if len(unique_labels) < 2:
        # Need at least 2 clusters
        print("Warning: Cannot calculate clustering metrics with fewer than 2 clusters")
    elif min(sample_counts) < 2:
        # Need at least 2 samples per cluster for silhouette score
        print(f"Warning: Cannot calculate silhouette score with fewer than 2 samples per cluster (min: {min(sample_counts)})")
        
        # These metrics can still be calculated with 1 sample per cluster
        try:
            eval["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
            eval["davies_bouldin_score"] = davies_bouldin_score(X, labels)
        except Exception as e:
            print(f"Error calculating cluster metrics: {e}")
    else:
        # Calculate all metrics
        try:
            silhouette_avg = silhouette_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            
            eval["silhouette_score"] = silhouette_avg
            eval["calinski_harabasz_score"] = ch_score
            eval["davies_bouldin_score"] = db_score
        except Exception as e:
            print(f"Error calculating cluster metrics: {e}")

    return eval


####################
# DISTANCE METRICS #
####################

def pairwise_distance(t_embs, s_embs, distance_metric):
    """
    s_embs[i] is the sound mapped from t_embs[i]
    :param t_embs:
    :param s_embs:
    :return: a distance
    """
    assert t_embs.shape[0] == s_embs.shape[0], "Text and sound embeddings must have the same number of samples"
    max_pairs = 1000
    distance = 0
    num_t_embs = t_embs.shape[0]

        # If we have fewer possible pairs than max_pairs, use all of them
    total_possible_pairs = (num_t_embs * (num_t_embs - 1)) // 2
    if total_possible_pairs <= max_pairs:
        pairs = list(combinations(range(num_t_embs), 2))
    else:
        # Sample random pairs without replacement
        all_indices = list(range(num_t_embs))
        pairs = []
        while len(pairs) < max_pairs:
            i, j = random.sample(all_indices, 2)
            if i > j:  # Ensure consistent ordering
                i, j = j, i
            if (i, j) not in pairs:  # Avoid duplicates
                pairs.append((i, j))
    
    for i, j in pairs:
        t1, t2 = t_embs[i], t_embs[j]
        s1, s2 = s_embs[i], s_embs[j]
        dt = distance_metric(t1, t2)
        ds = distance_metric(s1, s2)
        diff = abs(dt - ds)
        distance += diff
    distance /= len(pairs)  # normalize by the number of pairs
    return distance

def wasserstein_distance(t_embs, s_embs, distance_metric, num_samples=1000):
    """
    Compare distance distributions instead of individual pairs
    
    Args:
        t_embs: Text embeddings
        s_embs: Sound embeddings
        distance_metric: Function to calculate distance between embeddings
        num_samples: Number of pairs to sample for distribution comparison
        
    Returns:
        float: Wasserstein distance between text and sound distance distributions
    """
    num_t_embs = t_embs.shape[0]
    
    # Generate random pairs for sampling
    pairs = random.sample(list(combinations(range(num_t_embs), 2)), 
                         min(num_samples, num_t_embs*(num_t_embs-1)//2))
    
    # Calculate distance distributions
    text_distances = []
    sound_distances = []
    
    for i, j in pairs:
        t1, t2 = t_embs[i], t_embs[j]
        s1, s2 = s_embs[i], s_embs[j]
        text_distances.append(distance_metric(t1, t2))
        sound_distances.append(distance_metric(s1, s2))
    
    # Compare the distributions using Wasserstein distance
    return scipy_wasserstein(text_distances, sound_distances)


def CLAP_distance(texts, sound, distance_metric):
    # first embed them
    model, processor = load_CLAP()
    t_embs = CLAP_text(texts, model, processor)
    s_embs = CLAP_sound(sound, model, processor, SAMPLING_RATE)
    # then calculate the distance between the two sets of embeddings
    dist = pairwise_distance(t_embs, s_embs, distance_metric)
    return dist  


#####################
# MAPPING FUNCTIONS #
#####################


def identity(sound_embeddings, text_embeddings, distance_metric):
    W = np.eye(sound_embeddings.shape[1])
    mapped_text_embeddings = [W @ emb for emb in text_embeddings]
    _, nn_idx = find_nearest_neighbors(sound_embeddings, mapped_text_embeddings, distance_metric)
    return nn_idx


def cluster_map(sound_embeddings, text_embeddings, distance_metric, k):
    print(f"Applying clustering with k={k}...")
    sound_kmeans = KMeans(n_clusters=k, n_init=10).fit(sound_embeddings)
    text_kmeans = KMeans(n_clusters=k, n_init=10).fit(text_embeddings)

    sound_cluster_centers = sound_kmeans.cluster_centers_
    text_cluster_centers = text_kmeans.cluster_centers_

    sound_cluster_labels = sound_kmeans.labels_
    text_cluster_labels = text_kmeans.labels_

    # for each text cluster, find the nearest sound cluster
    nearest_cluster = {}  # maps a text cluster idx to the idx of the nearest sound cluster
    for i in range(k):
        center = text_cluster_centers[i]
        _, nearest_sound_cluster_idx = find_nearest_neighbors(sound_cluster_centers, [center], distance_metric)
        nearest_cluster[i] = nearest_sound_cluster_idx

    nearest_sounds = []
    # for each text embedding, find the nearest sound embedding that is in the nearest cluster
    for i, emb in enumerate(text_embeddings):
        text_cluster = text_kmeans.labels_[i]
        nearest_sound_cluster = nearest_cluster[text_cluster]  # idx of nearest sound cluster

        # get a list of the sound emebeddings in that cluster (this should be done in a separate loop)
        cluster_sound_idx = np.where(sound_kmeans.labels_ == nearest_sound_cluster)[0]  # list idx of sounds in that cluster
        cluster_sounds = sound_embeddings[cluster_sound_idx]

        # find the sound in this cluster that is nearest the text embedding
        nn, _ = find_nearest_neighbors(cluster_sounds, [emb], distance_metric)  # emb of nearest sound
        nearest_sounds.append(nn)

    # convert each sound embedding to the index it corresponds to in sound_corpus
    nearest_sound_idx = [np.where((sound_embeddings == sound))[0][0] for sound in nearest_sounds]
    return nearest_sound_idx, (sound_cluster_labels, text_cluster_labels)  # return the cluster labels for evaluation purposes


def icp_map(
    sound_embeddings,
    text_embeddings,
    distance_metric,
    max_iterations=50,
    batch_size=32,
    cycle_weight=0.1,
    learning_rate=0.01,
    return_validation_loss=False
):
    """
    Mini-Batch Cycle Iterative Closest Point (MBC-ICP) method for mapping between spaces.
    
    Args:
        sound_embeddings (np.ndarray): Sound embeddings (S space).
        text_embeddings (np.ndarray): Text embeddings (T space).
        distance_metric (str): Distance metric to use ("euclidean" or "cosine").
        max_iterations (int): Maximum number of iterations.
        batch_size (int): Mini-batch size for optimization.
        cycle_weight (float): Weight for cycle consistency loss.
        learning_rate (float): Learning rate for optimization.
        return_validation_loss (bool): Whether to return validation loss over iterations.
        
    Returns:
        np.ndarray: Transformed text embeddings after ICP.
        list (optional): Validation loss over iterations (if return_validation_loss=True).
    """
    print(f"Applying ICP mapping with {max_iterations} iterations...")
    
    # Initialize transformation matrices
    dim = sound_embeddings.shape[1]
    W_S = np.eye(dim)  # Sound to Text
    W_T = np.eye(dim)  # Text to Sound
    
    # Define distance function based on the metric
    if distance_metric == "euclidean":
        def dist_fn(x, y):
            return np.linalg.norm(x - y, axis=1)
    elif distance_metric == "cosine":
        def dist_fn(x, y):
            return np.array([distance.cosine(x[i], y[i]) for i in range(len(x))])
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    # Convert to numpy arrays if they aren't already
    if isinstance(sound_embeddings, torch.Tensor):
        sound_embeddings = sound_embeddings.detach().numpy()
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().numpy()
    
    # Track validation loss over iterations
    validation_losses = []

    # Main optimization loop
    for iter_num in range(max_iterations):
        # Randomly sample mini-batches
        if sound_embeddings.shape[0] > batch_size:
            sound_batch_indices = np.random.choice(sound_embeddings.shape[0], batch_size, replace=False)
            sound_batch = sound_embeddings[sound_batch_indices]
        else:
            sound_batch = sound_embeddings
            sound_batch_indices = np.arange(sound_embeddings.shape[0])
            
        if text_embeddings.shape[0] > batch_size:
            text_batch_indices = np.random.choice(text_embeddings.shape[0], batch_size, replace=False)
            text_batch = text_embeddings[text_batch_indices]
        else:
            text_batch = text_embeddings
            text_batch_indices = np.arange(text_embeddings.shape[0])
        
        # Step 1: Find nearest text embeddings for each sound embedding
        mapped_sound = sound_batch @ W_S
        
        # For each mapped sound, find the nearest text embedding
        nearest_text_indices = []
        for i in range(mapped_sound.shape[0]):
            distances = dist_fn(np.repeat(mapped_sound[i:i+1], text_batch.shape[0], axis=0), text_batch)
            nearest_idx = np.argmin(distances)
            nearest_text_indices.append(nearest_idx)
            
        nearest_text = text_batch[nearest_text_indices]
        
        # Step 2: Find nearest sound embeddings for each text embedding
        mapped_text = text_batch @ W_T
        
        # For each mapped text, find the nearest sound embedding
        nearest_sound_batch_indices = []
        for i in range(mapped_text.shape[0]):
            distances = dist_fn(np.repeat(mapped_text[i:i+1], sound_batch.shape[0], axis=0), sound_batch)
            nearest_idx = np.argmin(distances)
            nearest_sound_batch_indices.append(nearest_idx)
            
        nearest_sound = sound_batch[nearest_sound_batch_indices]
        
        # Step 3: Update transformation matrices
        # Loss 1: Sound->Text transformation accuracy
        gradient_W_S = 2 * (mapped_sound - nearest_text).T @ sound_batch
        
        # Loss 2: Text->Sound transformation accuracy
        gradient_W_T = 2 * (mapped_text - nearest_sound).T @ text_batch
        
        # Loss 3: Cycle consistency (S->T->S should be close to S)
        if cycle_weight > 0:
            cycled_sound = (sound_batch @ W_S) @ W_T
            cycle_loss_sound = cycled_sound - sound_batch
            gradient_cycle_W_S = 2 * cycle_loss_sound.T @ (sound_batch @ W_T.T)
            gradient_cycle_W_T = 2 * ((sound_batch @ W_S).T @ cycle_loss_sound)
            
            # Loss 4: Cycle consistency (T->S->T should be close to T)
            cycled_text = (text_batch @ W_T) @ W_S
            cycle_loss_text = cycled_text - text_batch
            gradient_cycle_W_T += 2 * cycle_loss_text.T @ (text_batch @ W_S.T)
            gradient_cycle_W_S += 2 * ((text_batch @ W_T).T @ cycle_loss_text)
            
            # Add cycle loss gradients
            gradient_W_S += cycle_weight * gradient_cycle_W_S
            gradient_W_T += cycle_weight * gradient_cycle_W_T
        
        # Update matrices
        W_S -= learning_rate * gradient_W_S
        W_T -= learning_rate * gradient_W_T
        
        # Optional: Orthogonalize the transformation matrices
        u_s, _, vh_s = np.linalg.svd(W_S, full_matrices=False)
        W_S = u_s @ vh_s
        
        u_t, _, vh_t = np.linalg.svd(W_T, full_matrices=False)
        W_T = u_t @ vh_t
        
        # Calculate validation loss every 10 iterations or at the last iteration
        if iter_num % 10 == 0 or iter_num == max_iterations - 1:
            mapped_text_all = text_embeddings @ W_T
            mapped_sound_all = sound_embeddings @ W_S
            
            # Sample for validation to avoid memory issues
            max_validation = min(1000, text_embeddings.shape[0], sound_embeddings.shape[0])
            val_indices = np.random.choice(min(text_embeddings.shape[0], sound_embeddings.shape[0]), max_validation, replace=False)
            
            validation_loss = np.mean(dist_fn(mapped_text_all[val_indices], sound_embeddings[val_indices])) + \
                              np.mean(dist_fn(mapped_sound_all[val_indices], text_embeddings[val_indices]))
            
            if cycle_weight > 0:
                cycle_loss = np.mean(dist_fn(mapped_text_all[val_indices] @ W_S, text_embeddings[val_indices])) + \
                             np.mean(dist_fn(mapped_sound_all[val_indices] @ W_T, sound_embeddings[val_indices]))
                validation_loss += cycle_weight * cycle_loss
            validation_losses.append(validation_loss)
            print(f"Iteration {iter_num}: Validation Loss = {validation_loss:.4f}")
    
    # Apply final mapping to all text embeddings
    mapped_text_embeddings = text_embeddings @ W_T

    _, neighbor_indices = find_nearest_neighbors(sound_embeddings, mapped_text_embeddings, distance_metric)

    if return_validation_loss:
        return neighbor_indices, validation_losses
    return neighbor_indices


def run(params, evaluator=None, cache=True, save_sound=False, normalize=False):
    sound_corpus_path = Path(params.sound_path)
    text_corpus_path = Path(params.text_path)

    if params.sound_encoder == "MuQ":
        sound_encoder = muq

    if params.text_encoder == "RoBERTa":
        text_encoder = RoBERTa
    elif params.text_encoder == "word2vec":
        text_encoder = word2vec
    elif params.text_encoder == "fastText":
        text_encoder = fastText

    print("Loading sound and text data...")
    sound_corpus = load_soundfiles(sound_corpus_path)
    text_corpus = load_text_corpus(text_corpus_path)

    # check to see if this will be a valid run
    if params.dim > len(sound_corpus) or params.dim > len(text_corpus):
        print(f"!!!: Input PCA dimension {params.dim} cannot be larger than length of sound corpus ({len(sound_corpus)}) or text corpus ({len(text_corpus)})")
        print("Ending this run.")
        return None

    # slice_fn: a function that determines how to separate sounds
    try:
        grain_size = int(params.sound_preprocessing / 1000.0 * SAMPLING_RATE)
        slice_fn = lambda y: equal_slices(y, grain_size)
        slice_fn_name = f"grain{params.sound_preprocessing}"
    except TypeError:
        if params.sound_preprocessing == "onsets":
            slice_fn = lambda y: get_onsets(y)
            slice_fn_name = "onsets"
        elif params.sound_preprocessing == "full":
            # slice_fn = lambda y: y
            slice_fn_name = "full"
        else:
            raise ValueError(f"Unknown preprocessing {params.sound_preprocessing}")
    if params.sound_preprocessing != 'full':
        sound_corpus = preprocess_sounds(sound_corpus, slice_fn, params.trim_silence)

    if params.sound_encoder == "MuQ":
        sound_corpus = [s for s in sound_corpus if s.size > 1024]  # MuQ requires sounds longer than 1024 samples

    sound_cache_file = None
    text_cache_file = None
    if cache:
        # Generate cache file paths
        sound_cache_file = generate_sound_cache_filename(
            corpus_name=sound_corpus_path.stem,
            slice_fn_name=slice_fn_name,
            encoder_name=params.sound_encoder,
        )
        text_cache_file = generate_text_cache_filename(
            text_file_name=text_corpus_path.name,
            encoder_name=params.text_encoder,
        )

    print("Embedding sounds...")
    sound_embeddings = embed_sounds(sound_corpus, sound_encoder, sound_cache_file)

    print("Embedding text...")
    text_embeddings = embed_text(" ".join(text_corpus), text_encoder, text_cache_file)
    if params.text_encoder == "word2vec":
        text_embeddings, valid_indices = text_embeddings
        text_corpus = [t for i, t in enumerate(text_corpus) if i in valid_indices]  # filter out invalid text samples that word2vec couldn't process

    # check again to see if this will be a valid run
    if params.dim > len(sound_embeddings) or params.dim > len(text_embeddings):
        print(
            f"!!!: Input PCA dimension {params.dim} cannot be larger than length of sound embeds ({len(sound_embeddings)}) or text embeds ({len(text_embeddings)})")
        print("Ending this run.")
        return None

    print("Transforming embeddings...")
    # what transformations will we apply to the feature space?
    if params.normalization == "standard":
        norm_method = StandardScaler()
    else:
        raise KeyError("Unknown normalization method")

    transform_pipeline = create_pipeline(norm_method, params.dim)
    for transform in transform_pipeline:
        sound_embeddings = transform(sound_embeddings)
        text_embeddings = transform(text_embeddings)

    mapping = params.mapping
    print(f"Mapping text to sound with method {mapping}...")
    # mapping options: identity, cluster_map, icp
    if mapping == "identity":
        neighbor_indices = identity(sound_embeddings, text_embeddings, params.distance_metric)
    elif mapping == "cluster":
        if params.k >= len(sound_embeddings) or params.k >= len(text_embeddings):
            print(f"!!!: Input k {params.k} cannot be larger than length of sound embeds ({len(sound_embeddings)}) or text embeds ({len(text_embeddings)})")
            print("Ending this run.")
            return None
        neighbor_indices, (sound_cluster_labels, text_cluster_labels) = cluster_map(sound_embeddings, text_embeddings, params.distance_metric, k=params.k)
    elif mapping == "icp":
        # Set ICP-specific parameters 
        # toy
        # icp_iterations = 50
        # batch_size = 16
        # cycle_weight = 0.5
        # learning_rate = 0.01
        # # anonymous_corpus
        # icp_iterations = 100
        # batch_size = 16
        # cycle_weight = 0.7
        # learning_rate = 0.001
        # best for TinySOL
        icp_iterations = 75
        batch_size = 16
        cycle_weight = 0.7
        learning_rate = 0.001
        
        
        neighbor_indices = icp_map(
            sound_embeddings, 
            text_embeddings, 
            params.distance_metric,
            max_iterations=icp_iterations,
            batch_size=batch_size,
            cycle_weight=cycle_weight,
            learning_rate=learning_rate
        )
    else:
        raise Exception("Invalid mapping provided")

    print("Evaluating mapping...")
    if params.distance_metric == "euclidean":
        distance_fn = lambda x, y: np.linalg.norm(x - y)
    elif params.distance_metric == "cosine":
        distance_fn = lambda x, y: distance.cosine(x, y)

    print("Fetching sounds...")
    output_sounds = [sound_corpus[i] for i in neighbor_indices]

    # Calculate pairwise distance
    pairwise_dist = pairwise_distance(text_embeddings, sound_embeddings[neighbor_indices], distance_fn)
    print(f"Pairwise Distance: {pairwise_dist}")
    
    # Calculate Wasserstein distance
    wasserstein_dist = wasserstein_distance(text_embeddings, sound_embeddings[neighbor_indices], distance_fn)
    print(f"Wasserstein Distance: {wasserstein_dist}")

    # Calculate CLAP pairwise distance
    clap_dist = CLAP_distance(text_corpus, output_sounds, distance_fn)
    print(f"CLAP Distance: {clap_dist}")

    if save_sound:
        print("Saving output...")
        save_path = OUTPUT_PATH / f"{params.filename()}.wav"
        save_output(output_sounds, save_path, normalize=normalize)

    print("Done.")

    # Save scores if evaluator is provided
    if evaluator:
        scores = {
            "pairwise_distance": pairwise_dist,
            "wasserstein_distance": wasserstein_dist,
            "CLAP_distance": clap_dist  # Add CLAP distance to scores
        }
        
        if params.mapping == "cluster":
            # First, evaluate individual spaces
            sound_eval = evaluate_clustering(sound_embeddings, sound_cluster_labels)
            text_eval = evaluate_clustering(text_embeddings, text_cluster_labels)
            
            # Add prefixes to distinguish between sound and text metrics
            sound_scores = {f"sound_{key}": value for key, value in sound_eval.items()}
            text_scores = {f"text_{key}": value for key, value in text_eval.items()}
            scores.update(sound_scores)
            scores.update(text_scores)
            
            # Now evaluate the combined space
            # 1. Create combined embeddings
            combined_embeddings = np.vstack((sound_embeddings, text_embeddings))
            
            # 2. Create combined labels 
            # - Keep original cluster assignments
            # - Offset text clusters to avoid overlap (e.g., if sound has clusters 0,1,2, text would be 3,4,5)
            num_sound_clusters = np.max(sound_cluster_labels)
            combined_labels = np.concatenate([
                sound_cluster_labels,
                text_cluster_labels + num_sound_clusters + 1
            ])
            
            # 3. Also create domain labels (0 for sound, 1 for text)
            domain_labels = np.concatenate([
                np.zeros(len(sound_embeddings)),
                np.ones(len(text_embeddings))
            ])
            
            # 4. Evaluate combined space with cluster labels
            combined_cluster_eval = evaluate_clustering(combined_embeddings, combined_labels)
            combined_scores = {f"combined_{key}": value for key, value in combined_cluster_eval.items()}
            scores.update(combined_scores)
            
            # 5. Evaluate separation between domains
            domain_eval = evaluate_clustering(combined_embeddings, domain_labels)
            domain_scores = {f"domain_{key}": value for key, value in domain_eval.items()}
            scores.update(domain_scores)
        
        evaluator.save_result(params, scores)

        return scores

if __name__ == "__main__":

    e = ParameterGenerator(
        sound_path="./corpora/sound/choir",
        text_path="./corpora/text/haiku.txt",
        sound_encoders=["MuQ"],
        text_encoders=["fastText"],
        # mappings=["identity", "cluster", "icp"],
        mappings=["cluster"],
        sound_preprocessings=[1000],
        normalizations=["standard"],
        dims=[5],
        distance_metrics=["euclidean"],
        mapping_evaluations=["pairwise"],
        ks=[5]  # Different values of k for clustering
    )

    parameter_list = e.create_params()
    for parameters in parameter_list:
        scores = run(parameters, cache=True, save_sound=True, normalize=True)
