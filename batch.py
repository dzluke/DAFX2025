from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import ParameterGenerator
from main import run
from eval import EvaluationFramework, analyze_experiments

def run_comprehensive_evaluation(check_cache=True):
    # Initialize the evaluation framework
    framework = EvaluationFramework()
    
    # Define corpora to test
    sound_corpora = [
        # "./corpora/sound/toy",
        "./corpora/sound/anonymous_corpus",
        "./corpora/sound/mothman",
        "./corpora/sound/TinySOL",
        "./corpora/sound/pres",
        "./corpora/sound/targets",
        "./corpora/sound/choir",
    ]
    
    text_corpora = [
        "./corpora/text/test.txt",
        "./corpora/text/repeat.txt",
        "./corpora/text/longer.txt"
    ]

    sound_encoders = ["MuQ"]
    # text_encoders = ["fastText", "word2vec", "RoBERTa"]  # Added RoBERTa for text encoding
    text_encoders = ["word2vec", "RoBERTa"]  # Added RoBERTa for text encoding
    mappings = ["identity", "cluster", "icp"]
    ks = [2, 5, 10, 20, 30]  # Different values of k for clustering
    sound_preprocessings = [250, "full", 1000]
    normalizations = ["standard"]
    dims = [50, 2, 5, 10, 20, 100]  # Different dimensions for embeddings
    distance_metrics = ["euclidean", "cosine"]
    mapping_evaluations = ["pairwise", 'wasserstein', 'CLAP']  #this actually does nothing currently

    # For each corpus combination, run evaluations
    for sound_corpus in sound_corpora:
        for text_corpus in text_corpora:
            print(f"Evaluating with sound corpus: {Path(sound_corpus).stem} and text corpus: {Path(text_corpus).stem}")
            
            # Create parameter combinations
            e = ParameterGenerator(
                sound_path=sound_corpus,
                text_path=text_corpus,
                sound_encoders=sound_encoders,
                text_encoders=text_encoders,
                mappings=mappings,
                sound_preprocessings=sound_preprocessings,
                normalizations=normalizations,
                dims=dims,
                distance_metrics=distance_metrics,
                mapping_evaluations=mapping_evaluations,
                ks=ks  # Include the k values
            )
            
            # Run each parameter combination
            parameter_list = e.create_params()

            # Get existing results to avoid re-running experiments
            existing_results = None
            if framework.results_file.exists():
                existing_results = framework.get_results_dataframe()
            
            for parameters in parameter_list:
                # Check if this experiment has been run before
                if check_cache and existing_results is not None and not existing_results.empty:
                    # Start with basic matching conditions
                    conditions = (
                        (existing_results['sound_corpus'] == Path(parameters.sound_path).stem) &
                        (existing_results['text_corpus'] == Path(parameters.text_path).stem) &
                        (existing_results['sound_encoder'] == parameters.sound_encoder) &
                        (existing_results['text_encoder'] == parameters.text_encoder) &
                        (existing_results['mapping_method'] == parameters.mapping) &
                        (existing_results['sound_preprocessing'] == parameters.sound_preprocessing) &
                        (existing_results['normalization'] == parameters.normalization) &
                        (existing_results['dim'] == parameters.dim) &
                        (existing_results['distance_metric'] == parameters.distance_metric) &
                        (existing_results['k'] == parameters.k) 
                    )
                    

                    # Check if any matching rows exist
                    if conditions.any():
                        print(f"Skipping previously run experiment: {parameters.to_string()}")
                        continue
                    else:
                        print(f"Running new experiment: {parameters.to_string()}")
                
                run(parameters, cache=True, evaluator=framework)
    
    # Generate a report of all evaluations
    report = framework.generate_report()
    print("Evaluation complete! Report generated.")
    
    # Analyze and visualize results
    analyze_experiments()
    
    return report


if __name__ == "__main__":
    run_comprehensive_evaluation()
