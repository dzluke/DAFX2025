import librosa
import numpy as np
from itertools import product
from pathlib import Path

SAMPLING_RATE = None

def set_sampling_rate(sr):
    """
    Set the global sampling rate for the system. This is to avoid circular imports with main..py
    """
    global SAMPLING_RATE
    SAMPLING_RATE = sr

class Parameter:
    """
    A set of parameters for one run of the system
    """
    def __init__(self, sound_path, text_path, sound_encoder, text_encoder, mapping, 
                 sound_preprocessing, normalization, dim, distance_metric, k=0.0, 
                 trim_silence=True, icp_iterations=50, batch_size=32, cycle_weight=0.1, 
                 learning_rate=0.01):
        self.sound_path = sound_path
        self.text_path = text_path
        self.sound_encoder = sound_encoder
        self.text_encoder = text_encoder
        self.mapping = mapping
        self.sound_preprocessing = sound_preprocessing
        self.normalization = normalization
        self.dim = dim
        self.distance_metric = distance_metric
        self.k = k
        self.trim_silence = trim_silence
        
        # Add ICP-specific parameters
        if mapping == "icp":
            self.icp_iterations = icp_iterations
            self.batch_size = batch_size
            self.cycle_weight = cycle_weight
            self.learning_rate = learning_rate

    def to_string(self):
        """
        Returns a string representation of the Parameter object.
        """
        k_info = ""
        if self.mapping == 'cluster':
            k_info = f"K: {self.k if hasattr(self, 'k') else 'N/A'}, "
        
        icp_info = ""
        if self.mapping == 'icp':
            icp_info = (
                f"ICP Iterations: {self.icp_iterations if hasattr(self, 'icp_iterations') else 50}, "
                f"Batch Size: {self.batch_size if hasattr(self, 'batch_size') else 32}, "
                f"Cycle Weight: {self.cycle_weight if hasattr(self, 'cycle_weight') else 0.1}, "
                f"Learning Rate: {self.learning_rate if hasattr(self, 'learning_rate') else 0.01}, "
            )
        
        return (
            f"Sound Path: {self.sound_path}, "
            f"Text Path: {self.text_path}, "
            f"Sound Encoder: {self.sound_encoder}, "
            f"Text Encoder: {self.text_encoder}, "
            f"Mapping: {self.mapping}, "
            f"{k_info}"
            f"{icp_info}"
            f"Sound Preprocessing: {self.sound_preprocessing}, "
            f"Normalization: {self.normalization}, "
            f"Dimension: {self.dim}, "
            f"Distance Metric: {self.distance_metric}, "
            f"Trim Silence: {self.trim_silence}"
        )
    

    def filename(self):
        """
        Returns a unique filename string for the parameter set.
        """
        sound_preprocessing_str = (
            f"grain{self.sound_preprocessing}" if isinstance(self.sound_preprocessing, int) else self.sound_preprocessing
        )
        trim_silence_str = "trim_silence" if self.trim_silence else ""
        
        # Include k in the filename only for cluster mapping
        k_str = f"k{self.k}_" if self.mapping == 'cluster' and hasattr(self, 'k') else ""
        
        # Include ICP parameters if using ICP mapping
        icp_str = ""
        if self.mapping == 'icp' and hasattr(self, 'icp_iterations'):
            icp_str = f"icp{self.icp_iterations}_cw{self.cycle_weight}_lr{self.learning_rate}_"
        
        filename = (
            f"{Path(self.sound_path).stem}_"
            f"{Path(self.text_path).stem}_"
            f"{self.sound_encoder}_"
            f"{self.text_encoder}_"
            f"{self.mapping}_"
            f"{sound_preprocessing_str}_"
            f"{self.normalization}_"
            f"{self.distance_metric}_"
            f"dim{self.dim}_"
            f"{k_str}"
            f"{icp_str}"
            f"{trim_silence_str}"
        )
        return filename.replace(" ", "_")

class ParameterGenerator:
    """
    A set of all possible parameters to test for the system. It outputs a list of Parameter objects that exhaustively cover
    all possible input parameters.
    """

    def __init__(self, sound_path, text_path, sound_encoders, text_encoders, mappings, 
                 sound_preprocessings, normalizations, dims, distance_metrics, 
                 mapping_evaluations, ks=None, clustering_evaluations=None,
                 icp_iterations=None, batch_sizes=None, cycle_weights=None, learning_rates=None):
        self.sound_path = sound_path
        self.text_path = text_path
        self.sound_encoders = sound_encoders
        self.text_encoders = text_encoders
        self.mappings = mappings
        self.sound_preprocessings = sound_preprocessings
        self.normalizations = normalizations
        self.dims = dims
        self.distance_metrics = distance_metrics
        self.mapping_evaluations = mapping_evaluations
        if ks is not None:
            self.ks = ks
        if clustering_evaluations is not None:
            self.clustering_evaluations = clustering_evaluations
            
        # ICP parameters
        if icp_iterations is not None:
            self.icp_iterations = icp_iterations
        else:
            self.icp_iterations = [50]  # Default value
            
        if batch_sizes is not None:
            self.batch_sizes = batch_sizes
        else:
            self.batch_sizes = [32]  # Default value
            
        if cycle_weights is not None:
            self.cycle_weights = cycle_weights
        else:
            self.cycle_weights = [0.1]  # Default value
            
        if learning_rates is not None:
            self.learning_rates = learning_rates
        else:
            self.learning_rates = [0.01]  # Default value

    def create_params(self):
        param_combinations = product(
            self.sound_encoders,
            self.text_encoders,
            self.mappings,
            self.sound_preprocessings,
            self.normalizations,
            self.dims,
            self.distance_metrics
        )
        params = []
        for sound_encoder, text_encoder, mapping, sound_preprocessing, normalization, dims, distance_metric in param_combinations:
            if mapping == 'cluster':
                assert hasattr(self, 'ks'), "ks must be provided for clustering evaluations"
                for k in self.ks:
                    params.append(
                        Parameter(
                            self.sound_path,
                            self.text_path,
                            sound_encoder,
                            text_encoder,
                            mapping,
                            sound_preprocessing,
                            normalization,
                            dims,
                            distance_metric,
                            k=k
                        )
                    )
            elif mapping == 'icp':
                # Generate parameter combinations for ICP
                for icp_iter in self.icp_iterations:
                    for batch_size in self.batch_sizes:
                        for cycle_weight in self.cycle_weights:
                            for lr in self.learning_rates:
                                params.append(
                                    Parameter(
                                        self.sound_path,
                                        self.text_path,
                                        sound_encoder,
                                        text_encoder,
                                        mapping,
                                        sound_preprocessing,
                                        normalization,
                                        dims,
                                        distance_metric,
                                        icp_iterations=icp_iter,
                                        batch_size=batch_size,
                                        cycle_weight=cycle_weight,
                                        learning_rate=lr
                                    )
                                )
            else:
                params.append(
                    Parameter(
                        self.sound_path,
                        self.text_path,
                        sound_encoder,
                        text_encoder,
                        mapping,
                        sound_preprocessing,
                        normalization,
                        dims,
                        distance_metric
                    )
                )

        return params


def load_soundfiles(path):
    soundfiles = []
    for file in path.iterdir():
        if file.is_dir():
            soundfiles = load_soundfiles(file)
        else:
            if file.suffix in [".wav", ".aif", ".mp3", ".m4a"]:
                data, sr = librosa.load(file, sr=SAMPLING_RATE)
                assert sr == SAMPLING_RATE
                if data.ndim > 1:  # convert to mono
                    data = data.sum(axis=1)
                    data = data / np.abs(data).max()
                soundfiles.append(data)
    return soundfiles


def remove_silence(sounds, top_db=60):
    """
    Removes leading and trailing silence from a list of sound arrays.

    Args:
        sounds (list of np.ndarray): List of audio signals as numpy arrays.
        top_db (int): Threshold (in decibels) below reference to consider as silence.

    Returns:
        list of np.ndarray: List of trimmed audio signals.
    """
    trimmed_sounds = []
    for sound in sounds:
        # Check if the whole sound is below the top_db threshold
        if librosa.get_duration(y=sound) == 0 or np.max(np.abs(sound)) < librosa.db_to_amplitude(-top_db):
            continue

        # Trim silence from the beginning and end
        trimmed_sound, _ = librosa.effects.trim(sound, top_db=top_db)
        trimmed_sounds.append(trimmed_sound)
    return trimmed_sounds