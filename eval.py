import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EvaluationFramework:
    """
    Framework for evaluating text-to-sound mapping with different parameters
    across multiple input corpora.
    """
    def __init__(self, results_dir="./evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "evaluation_results.csv"
        self.initialize_results_file()
    
    def initialize_results_file(self):
        """Initialize the results file if it doesn't exist."""
        if not self.results_file.exists():
            columns = [
                "sound_corpus", "text_corpus", "sound_encoder", "text_encoder",
                "mapping_method", "sound_preprocessing", "normalization",
                "dim", "distance_metric", "pairwise_distance", "wasserstein_distance",
                "CLAP_distance",  # Add CLAP distance column
                # Sound cluster metrics
                "sound_silhouette_score", "sound_calinski_harabasz_score", "sound_davies_bouldin_score",
                # Text cluster metrics
                "text_silhouette_score", "text_calinski_harabasz_score", "text_davies_bouldin_score",
                # Combined cluster metrics
                "combined_silhouette_score", "combined_calinski_harabasz_score", "combined_davies_bouldin_score",
                # Domain separation metrics
                "domain_silhouette_score", "domain_calinski_harabasz_score", "domain_davies_bouldin_score"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.results_file, index=False)
    
    def save_result(self, params, scores):
        """
        Save evaluation results for a parameter set.
        
        Args:
            params: Parameter object with configuration
            scores: Dict containing evaluation scores
        """
        # Read existing results
        df = pd.read_csv(self.results_file)
        
        # Create a new row with the parameters and scores
        sound_corpus = Path(params.sound_path).stem
        text_corpus = Path(params.text_path).stem
        
        # Create base row with parameters
        row = {
            "sound_corpus": sound_corpus,
            "text_corpus": text_corpus,
            "sound_encoder": params.sound_encoder,
            "text_encoder": params.text_encoder, 
            "mapping_method": params.mapping,
            "sound_preprocessing": params.sound_preprocessing,
            "normalization": params.normalization,
            "dim": params.dim,
            "distance_metric": params.distance_metric,
            "k": params.k if params.mapping == "cluster" else 0  # Set k to 0 for identity mapping
        }
        
        # Add all scores to the row
        row.update(scores)
        
        # # Check if a matching row already exists
        # conditions = (
        #     (df["sound_corpus"] == sound_corpus) &
        #     (df["text_corpus"] == text_corpus) &
        #     (df["sound_encoder"] == params.sound_encoder) &
        #     (df["text_encoder"] == params.text_encoder) &
        #     (df["mapping_method"] == params.mapping) &
        #     (df["sound_preprocessing"] == params.sound_preprocessing) &
        #     (df["normalization"] == params.normalization) &
        #     (df["dim"] == params.dim) &
        #     (df["distance_metric"] == params.distance_metric)
        # )
        
        # if params.mapping == "cluster" and hasattr(params, "k"):
        #     conditions &= (df["k"] == params.k)
        
        # matching_rows = df[conditions]
        
        # if not matching_rows.empty:
        #     # Update the first matching row
        #     index = matching_rows.index[0]
        #     for key, value in row.items():
        #         df.at[index, key] = value
        # else:
        #     # Add the new row
        #     df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

         # Add the new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Save the updated results
        df.to_csv(self.results_file, index=False)
    
    def find_best_parameters(self, metric="pairwise_distance", lower_is_better=True, 
                            sound_corpus=None, text_corpus=None):
        """
        Find the best parameter set based on a specific metric.
        
        Args:
            metric: The metric to optimize
            lower_is_better: Whether a lower score is better
            sound_corpus: Filter by sound corpus (optional)
            text_corpus: Filter by text corpus (optional)
            
        Returns:
            pd.DataFrame: The best parameter combinations
        """
        df = pd.read_csv(self.results_file)
        
        # Apply filters if provided
        if sound_corpus:
            df = df[df["sound_corpus"] == sound_corpus]
        if text_corpus:
            df = df[df["text_corpus"] == text_corpus]
        
        # Sort based on the metric
        ascending = lower_is_better
        df_sorted = df.sort_values(by=metric, ascending=ascending)
        
        # Return the top results
        return df_sorted.head(10)
    
    def compare_encoders(self, metric="pairwise_distance", lower_is_better=True):
        """
        Compare performance of different encoder combinations across all corpora.
        
        Args:
            metric: The metric to compare
            lower_is_better: Whether a lower score is better
            
        Returns:
            pd.DataFrame: Average performance by encoder combination
        """
        df = pd.read_csv(self.results_file)
        
        # Group by encoder combinations and compute average score
        grouped = df.groupby(["sound_encoder", "text_encoder"])[metric].agg(["mean", "std", "count"])
        
        # Sort based on mean score
        ascending = lower_is_better
        return grouped.sort_values(by="mean", ascending=ascending)
    
    def plot_parameter_impact(self, parameter, metric="pairwise_distance"):
        """
        Create boxplot showing impact of a parameter on the evaluation metric.
        
        Args:
            parameter: Parameter to analyze (e.g., "sound_encoder", "dim")
            metric: Metric to evaluate
        """
        df = pd.read_csv(self.results_file)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=parameter, y=metric, data=df)
        plt.title(f"Impact of {parameter} on {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.results_dir / f"impact_{parameter}_on_{metric}.png"
        plt.savefig(plot_file)
        plt.close()
    
    def generate_report(self):
        """
        Generate a comprehensive report of all evaluation results.
        
        Returns:
            dict: A dictionary containing the report data
        """
        df = pd.read_csv(self.results_file)
        
        # Check which metrics are available in the dataframe
        available_metrics = set(df.columns)
        
        report = {
            "total_runs": len(df),
            "unique_sound_corpora": df["sound_corpus"].nunique(),
            "unique_text_corpora": df["text_corpus"].nunique(),
        }
        
        # Add best metric scores if available
        for metric_name in ["pairwise_distance", "wasserstein_distance"]:
            if metric_name in available_metrics:
                report[f"best_{metric_name}"] = self.find_best_parameters(metric_name).to_dict(orient="records")[0]
        
        # Add best silhouette scores if available
        if "combined_silhouette_score" in available_metrics:
            report["best_combined_silhouette"] = self.find_best_parameters(
                "combined_silhouette_score", 
                lower_is_better=False
            ).to_dict(orient="records")[0]
        
        if "sound_silhouette_score" in available_metrics:
            report["best_sound_silhouette"] = self.find_best_parameters(
                "sound_silhouette_score", 
                lower_is_better=False
            ).to_dict(orient="records")[0]
        
        if "text_silhouette_score" in available_metrics:
            report["best_text_silhouette"] = self.find_best_parameters(
                "text_silhouette_score", 
                lower_is_better=False
            ).to_dict(orient="records")[0]
        
        # Summary by sound_corpus
        report["by_sound_corpus"] = {}
        for corpus in df["sound_corpus"].unique():
            corpus_df = df[df["sound_corpus"] == corpus]
            
            corpus_report = {
                "runs": len(corpus_df)
            }
            
            # Add average metrics if available
            for metric_name in ["pairwise_distance", "wasserstein_distance"]:
                if metric_name in available_metrics:
                    corpus_report[f"avg_{metric_name}"] = corpus_df[metric_name].mean()
                    corpus_report[f"best_{metric_name}"] = self.find_best_parameters(
                        metric_name, 
                        sound_corpus=corpus
                    ).to_dict(orient="records")[0]
            
            report["by_sound_corpus"][corpus] = corpus_report
        
        # Summary by text_corpus (similar structure)
        report["by_text_corpus"] = {}
        for corpus in df["text_corpus"].unique():
            corpus_df = df[df["text_corpus"] == corpus]
            
            corpus_report = {
                "runs": len(corpus_df)
            }
            
            # Add average pairwise distance if available
            if "pairwise_distance" in available_metrics:
                corpus_report["avg_pairwise_distance"] = corpus_df["pairwise_distance"].mean()
                corpus_report["best_pairwise"] = self.find_best_parameters(
                    "pairwise_distance", 
                    text_corpus=corpus
                ).to_dict(orient="records")[0]
            
            report["by_text_corpus"][corpus] = corpus_report
        
        # Save report to JSON
        report_path = self.results_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")
        
        return report

    def get_results_dataframe(self):
        """
        Returns the complete results dataframe.
        """
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame()

    def get_unique_values(self, column):
        """
        Returns the unique values in a specific column.
        
        Args:
            column: The column name to get unique values from
            
        Returns:
            list: The unique values in the column
        """
        df = self.get_results_dataframe()
        if column in df.columns:
            return df[column].unique().tolist()
        return []

def analyze_experiments(results_file="./evaluation_results/evaluation_results.csv", filter_params=None, compare_param=None):
    """
    Analyze experiments from the evaluation results file, with optional filtering by parameters and comparison of a specific parameter.

    Args:
        results_file (str): Path to the evaluation results CSV file.
        filter_params (dict, optional): Dictionary of parameters to filter by.
        compare_param (str, optional): Parameter to compare results for (e.g., 'dim', 'k', 'sound_preprocessing').
    """
    # Load the results file
    results_file = Path(results_file)
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    df = pd.read_csv(results_file)

    if df.empty:
        print("No results to analyze.")
        return
    
    # Apply filters if provided
    if filter_params:
        print("\n" + "="*50)
        print(f"FILTERED ANALYSIS - Using the following filters:")
        for param, values in filter_params.items():
            if isinstance(values, list):
                print(f"  * {param}: {values}")
            else:
                print(f"  * {param}: {values}")
        print("="*50)
        
        filtered_df = df.copy()
        filter_description = []
        
        for param, values in filter_params.items():
            if param in filtered_df.columns:
                # Handle both single values and lists of values
                if isinstance(values, list):
                    filtered_df = filtered_df[filtered_df[param].isin(values)]
                    filter_description.append(f"{param} in {values}")
                else:
                    filtered_df = filtered_df[filtered_df[param] == values]
                    filter_description.append(f"{param} = {values}")
        
        # Check if we have any results after filtering
        if filtered_df.empty:
            print(f"No results match the filter criteria: {', '.join(filter_description)}")
            return
            
        print(f"\nAnalyzing {len(filtered_df)} experiments matching the filter criteria")
        print(f"(Total experiments in dataset: {len(df)})")
        df = filtered_df
    else:
        print(f"\nAnalyzing all {len(df)} experiments (no filters applied)")

    # Additional summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 50)
    
    # Print unique values for key parameters
    print("Parameter distribution in analyzed experiments:")
    for param in ['sound_corpus', 'text_corpus', 'sound_encoder', 'text_encoder', 'dim', 'k']:
        if param in df.columns:
            unique_values = df[param].value_counts().to_dict()
            print(f"  {param}: {unique_values}")

    print("\n" + "=" * 50)
    print("ANALYSIS OF EVALUATION RESULTS")
    print("=" * 50 + "\n")

    # Analyze both pairwise, Wasserstein, and CLAP distance metrics
    metrics = []
    if "pairwise_distance" in df.columns:
        metrics.append("pairwise_distance")
    if "wasserstein_distance" in df.columns:
        metrics.append("wasserstein_distance")
    if "CLAP_distance" in df.columns:  # Include CLAP distance in analysis
        metrics.append("CLAP_distance")
    
    if not metrics:
        print("No distance metrics found in results.")
        return

    for metric in metrics:
        print(f"\nAVERAGE {metric.upper()} BY MAPPING METHOD")
        print("-" * 50)

        # Analyze by mapping method
        mapping_methods = df["mapping_method"].unique()
        for mapping in mapping_methods:
            mapping_rows = df[df["mapping_method"] == mapping]
            values = mapping_rows[metric].dropna()

            if not values.empty:
                avg_score = values.mean()
                print(f"Mapping: {mapping}")
                print(f"  Average {metric}: {avg_score:.4f}")
                print(f"  Number of experiments: {len(values)}")

                # If cluster mapping, analyze metrics by k
                if mapping == "cluster" and "k" in mapping_rows.columns:
                    print(f"\n  {metric} by k:")
                    for k in sorted(mapping_rows["k"].dropna().unique()):
                        k_rows = mapping_rows[mapping_rows["k"] == k]
                        values = k_rows[metric].dropna()
                        if not values.empty:
                            avg = values.mean()
                            print(f"    k={k}: {avg:.4f}  count={len(values)}")
                    print()
            else:
                print(f"Mapping: {mapping}")
                print(f"  No valid {metric} results found")
                print()

    if compare_param and compare_param in df.columns:
        print(f"\nCOMPARISON OF RESULTS BY '{compare_param}'")
        print("-" * 50)
        unique_values = sorted(df[compare_param].dropna().unique())
        for value in unique_values:
            subset = df[df[compare_param] == value]
            print(f"{compare_param} = {value}:")
            for metric in ["pairwise_distance", "wasserstein_distance", "CLAP_distance"]:
                if metric in subset.columns:
                    avg_metric = subset[metric].mean()
                    print(f"  Avg {metric}: {avg_metric:.4f} (n={len(subset)})")
            print()
    
    # Analyze clustering metrics if present
    clustering_metrics = [
        "combined_silhouette_score", "domain_silhouette_score",
        "sound_silhouette_score", "text_silhouette_score"
    ]
    
    # Check if we have clustering metrics in the results
    has_clustering_metrics = any(metric in df.columns for metric in clustering_metrics)
    
    if has_clustering_metrics and "cluster" in df["mapping_method"].unique():
        print("\nCLUSTERING METRICS ANALYSIS")
        print("-" * 50)
        
        cluster_df = df[df["mapping_method"] == "cluster"]
        
        # Analyze combined silhouette scores by k
        if "combined_silhouette_score" in cluster_df.columns:
            print("\nCOMBINED SILHOUETTE SCORE BY K")
            print("(Higher values indicate better-defined clusters across both domains)")
            print("-" * 50)
            
            for k in sorted(cluster_df["k"].dropna().unique()):
                k_rows = cluster_df[cluster_df["k"] == k]
                values = k_rows["combined_silhouette_score"].dropna()
                if not values.empty:
                    avg = values.mean()
                    print(f"k={k}: {avg:.4f}")
                    
                    # # Show by corpus if we have multiple
                    # if k_rows["sound_corpus"].nunique() > 1:
                    #     print("  By sound corpus:")
                    #     for corpus in k_rows["sound_corpus"].unique():
                    #         corpus_rows = k_rows[k_rows["sound_corpus"] == corpus]
                    #         corpus_avg = corpus_rows["combined_silhouette_score"].mean()
                    #         print(f"    {corpus}: {corpus_avg:.4f}")
            
        # Analyze domain silhouette scores
        # if "domain_silhouette_score" in cluster_df.columns:
        #     print("\nDOMAIN SILHOUETTE SCORE BY K")
        #     print("(Lower values are better - indicate better blending between domains)")
        #     print("-" * 50)
            
        #     for k in sorted(cluster_df["k"].dropna().unique()):
        #         k_rows = cluster_df[cluster_df["k"] == k]
        #         values = k_rows["domain_silhouette_score"].dropna()
        #         if not values.empty:
        #             avg = values.mean()
        #             print(f"k={k}: {avg:.4f}")
                    
        #             # Interpretation guide
        #             if avg < -0.1:
        #                 print("  Interpretation: Excellent domain blending (strong negative score)")
        #             elif avg < 0:
        #                 print("  Interpretation: Good domain blending (negative score)")
        #             elif avg < 0.1:
        #                 print("  Interpretation: Moderate domain blending (near zero)")
        #             else:
        #                 print("  Interpretation: Poor domain separation (positive score)")
                    
        #             # Show by corpus if we have multiple
        #             if k_rows["sound_corpus"].nunique() > 1:
        #                 print("  By sound corpus:")
        #                 for corpus in k_rows["sound_corpus"].unique():
        #                     corpus_rows = k_rows[k_rows["sound_corpus"] == corpus]
        #                     corpus_avg = corpus_rows["domain_silhouette_score"].mean()
        #                     print(f"    {corpus}: {corpus_avg:.4f}")
        
        # Compare sound and text clustering quality
        # if "sound_silhouette_score" in cluster_df.columns and "text_silhouette_score" in cluster_df.columns:
        #     print("\nSOUND VS TEXT CLUSTERING QUALITY")
        #     print("(Higher values indicate better-defined clusters in each domain)")
        #     print("-" * 50)
            
        #     sound_avg = cluster_df["sound_silhouette_score"].dropna().mean()
        #     text_avg = cluster_df["text_silhouette_score"].dropna().mean()
            
        #     print(f"Average sound silhouette score: {sound_avg:.4f}")
        #     print(f"Average text silhouette score: {text_avg:.4f}")
            
        #     if sound_avg > text_avg:
        #         diff = sound_avg - text_avg
        #         print(f"Sound clusters are better defined by {diff:.4f}")
        #     elif text_avg > sound_avg:
        #         diff = text_avg - sound_avg
        #         print(f"Text clusters are better defined by {diff:.4f}")
        #     else:
        #         print("Sound and text clusters are equally well-defined")
            
        #     print("\nBy k value:")
        #     for k in sorted(cluster_df["k"].dropna().unique()):
        #         k_rows = cluster_df[cluster_df["k"] == k]
        #         sound_k_avg = k_rows["sound_silhouette_score"].dropna().mean()
        #         text_k_avg = k_rows["text_silhouette_score"].dropna().mean()
        #         print(f"  k={k}:")
        #         print(f"    Sound: {sound_k_avg:.4f}, Text: {text_k_avg:.4f}")
    
    
    # Print best overall results for each metric
    # for metric in metrics:
    #     print(f"\nBest {metric} results:")
    #     best_idx = df[metric].idxmin()  # Lower is better for both pairwise and Wasserstein
    #     if pd.notna(best_idx):
    #         best_row = df.loc[best_idx]
    #         print(f"  Best {metric}: {best_row[metric]:.4f}")
    #         for param in ['sound_corpus', 'text_corpus', 'sound_encoder', 'text_encoder', 
    #                     'mapping_method', 'dim', 'k']:
    #             if param in best_row and pd.notna(best_row[param]):
    #                 print(f"    {param}: {best_row[param]}")

    # Analyze clustering metrics if present
    clustering_metrics = [
        "combined_silhouette_score", "combined_calinski_harabasz_score",
        "combined_davies_bouldin_score"
    ]
    
    # Check if we have clustering metrics in the results
    has_clustering_metrics = any(metric in df.columns for metric in clustering_metrics)
    
    if has_clustering_metrics and "cluster" in df["mapping_method"].unique():
        print("\nCLUSTERING METRICS ANALYSIS")
        print("-" * 50)
        
        cluster_df = df[df["mapping_method"] == "cluster"]
        
        if compare_param and compare_param in cluster_df.columns:
            print(f"\nCLUSTERING METRICS BY '{compare_param}'")
            print("-" * 50)
            
            for value in sorted(cluster_df[compare_param].dropna().unique()):
                subset = cluster_df[cluster_df[compare_param] == value]
                print(f"{compare_param} = {value}:")
                
                for metric in clustering_metrics:
                    if metric in subset.columns:
                        avg_metric = subset[metric].mean()
                        print(f"  Avg {metric}: {avg_metric:.4f} (n={len(subset)})")
                print()
        else:
            print(f"No valid '{compare_param}' found in clustering metrics analysis.")
    
    # Print best overall results for each metric
    # for metric in metrics:
    #     print(f"\nBest {metric} results:")
    #     best_idx = df[metric].idxmin()  # Lower is better for both pairwise and Wasserstein
    #     if pd.notna(best_idx):
    #         best_row = df.loc[best_idx]
    #         print(f"  Best {metric}: {best_row[metric]:.4f}")
    #         for param in ['sound_corpus', 'text_corpus', 'sound_encoder', 'text_encoder', 
    #                     'mapping_method', 'dim', 'k']:
    #             if param in best_row and pd.notna(best_row[param]):
    #                 print(f"    {param}: {best_row[param]}")

    # Analyze clustering metrics if present
    clustering_metrics = [
        "combined_silhouette_score", "combined_calinski_harabasz_score",
        "combined_davies_bouldin_score"
    ]
    
    # Check if we have clustering metrics in the results
    has_clustering_metrics = any(metric in df.columns for metric in clustering_metrics)
    
    if has_clustering_metrics:
        print("\nPEARSON'S CORRELATION ANALYSIS")
        print("-" * 50)
        
        for mapping_metric in metrics:  # Include CLAP_distance in correlation analysis
            for clustering_metric in clustering_metrics:
                if clustering_metric in df.columns:
                    # Drop rows with NaN values for the selected metrics
                    valid_rows = df[[mapping_metric, clustering_metric]].dropna()
                    if not valid_rows.empty:
                        correlation = valid_rows[mapping_metric].corr(valid_rows[clustering_metric])
                        print(f"Pearson's correlation between {mapping_metric} and {clustering_metric}: {correlation:.4f}")
                    else:
                        print(f"Not enough data to calculate correlation between {mapping_metric} and {clustering_metric}.")
            print('\n')

    print("\nAnalysis complete.")
    return df  # Return the filtered dataframe for further analysis if needed
    

def generate_plots(results_file="./evaluation_results/evaluation_results.csv", 
                   output_dir="./evaluation_results/plots", 
                   filter_params=None,
                   compare_param=None,
                   clustering_metrics=None,
                   mapping_metrics=None):
    """
    Generate plots from the evaluation results with optional filtering and comparison of a specific parameter.

    Args:
        results_file (str): Path to the evaluation results CSV file.
        output_dir (str): Directory to save the generated plots.
        filter_params (dict, optional): Dictionary of parameters to filter by.
        compare_param (str, optional): Parameter to compare results for (e.g., 'dim', 'k', 'sound_preprocessing').
        clustering_metrics (list or str, optional): Specific clustering metrics to analyze.
        mapping_metrics (list or str, optional): Specific mapping metrics to analyze.
    """
    results_file = Path(results_file)
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    df = pd.read_csv(results_file)

    if df.empty:
        print("No results to analyze.")
        return
    
    # Apply filters if provided
    if filter_params:
        print("\n" + "="*50)
        print(f"FILTERED PLOT GENERATION - Using the following filters:")
        for param, values in filter_params.items():
            if isinstance(values, list):
                print(f"  * {param}: {values}")
            else:
                print(f"  * {param}: {values}")
        print("="*50)
        
        filtered_df = df.copy()
        
        for param, values in filter_params.items():
            if param in filtered_df.columns:
                # Handle both single values and lists of values
                if isinstance(values, list):
                    filtered_df = filtered_df[filtered_df[param].isin(values)]
                else:
                    filtered_df = filtered_df[filtered_df[param] == values]
        
        # Check if we have any results after filtering
        if filtered_df.empty:
            print(f"No results match the filter criteria")
            return
            
        print(f"\nGenerating plots for {len(filtered_df)} experiments matching the filter criteria")
        df = filtered_df
    else:
        print(f"\nGenerating plots for all {len(df)} experiments (no filters applied)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default metrics if none provided
    if clustering_metrics is None:
        clustering_metrics = ['combined_silhouette_score']
    elif isinstance(clustering_metrics, str):
        clustering_metrics = [clustering_metrics]
        
    if mapping_metrics is None:
        mapping_metrics = ['pairwise_distance']
    elif isinstance(mapping_metrics, str):
        mapping_metrics = [mapping_metrics]
    
    # Filter to only metrics that exist in the dataframe
    clustering_metrics = [m for m in clustering_metrics if m in df.columns]
    mapping_metrics = [m for m in mapping_metrics if m in df.columns]
    
    if not clustering_metrics:
        print("No specified clustering metrics found in results.")
    if not mapping_metrics:
        print("No specified mapping metrics found in results.")
    
    if not clustering_metrics or not mapping_metrics:
        return
    
    print(f"Analyzing clustering metrics: {', '.join(clustering_metrics)}")
    print(f"Analyzing mapping metrics: {', '.join(mapping_metrics)}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_parameters = ["mapping_method", "sound_encoder", "text_encoder", "dim", "k"]
    
    # Generate plots for mapping evaluation scores vs. sound preprocessing
    if "sound_preprocessing" in df.columns:
        for metric in mapping_metrics:
            if metric in df.columns:
                plt.figure(figsize=(10, 6))
                # Sort data by the x-axis label
                sorted_df = df.sort_values(by="sound_preprocessing")
                sns.boxplot(x="sound_preprocessing", y=metric, data=sorted_df)
                plt.title(f"Effect of Sound Preprocessing on {metric} (lower is better)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_file = output_dir / f"sound_preprocessing_effect_on_{metric}.png"
                plt.savefig(plot_file)
                plt.close()
                print(f"Saved plot: {plot_file}")

    # Generate plots for mapping evaluation scores vs. dim
    if "dim" in df.columns:
        for metric in mapping_metrics:
            if metric in df.columns:
                plt.figure(figsize=(10, 6))
                # Sort data by the x-axis label
                sorted_df = df.sort_values(by="dim")
                sns.boxplot(x="dim", y=metric, data=sorted_df)
                plt.title(f"Effect of Dimensionality (dim) on {metric} (lower is better)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_file = output_dir / f"dim_effect_on_{metric}.png"
                plt.savefig(plot_file)
                plt.close()
                print(f"Saved plot: {plot_file}")

    if compare_param and compare_param in df.columns:
        print(f"\nGENERATING PLOTS FOR '{compare_param}'")
        print("-" * 50)
        for metric in mapping_metrics + clustering_metrics:
            if metric in df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=compare_param, y=metric, data=df)
                plt.title(f"Effect of {compare_param} on {metric}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_file = output_dir / f"{compare_param}_effect_on_{metric}.png"
                plt.savefig(plot_file)
                plt.close()
                print(f"Saved plot: {plot_file}")

    # Generate a plot showing the average of all three evaluation metrics for each k value (excluding k=50)
    if "k" in df.columns and "mapping_method" in df.columns:
        cluster_df = df[(df["mapping_method"] == "cluster") & (df["k"] != 50)]

        # Ensure the required metrics are present
        metrics = ["pairwise_distance", "wasserstein_distance", "CLAP_distance"]
        available_metrics = [metric for metric in metrics if metric in cluster_df.columns]

        if available_metrics:
            # Calculate the average of each metric for each k
            avg_metrics = cluster_df.groupby("k")[available_metrics].mean().reset_index()

            # Separate CLAP distance for secondary y-axis
            clap_avg = avg_metrics[["k", "CLAP_distance"]]
            other_metrics = avg_metrics.drop(columns=["CLAP_distance"]).melt(id_vars="k", var_name="Metric", value_name="Average")

            # Generate the plot
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot pairwise and Wasserstein distances on the primary y-axis
            sns.lineplot(data=other_metrics, x="k", y="Average", hue="Metric", marker="o", ax=ax1)
            ax1.set_title("Effect of k on Evaluation Metrics (Excluding k=50)")
            ax1.set_xlabel("k (Number of Clusters)")
            ax1.set_ylabel("Average Metric Value (Pairwise/Wasserstein)")
            ax1.legend(title="Metric", loc="upper left")

            # Plot CLAP distance on the secondary y-axis
            ax2 = ax1.twinx()
            sns.lineplot(data=clap_avg, x="k", y="CLAP_distance", color="red", marker="o", ax=ax2)
            ax2.set_ylabel("Average CLAP Distance", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            plt.tight_layout()

            # Save the plot
            plot_file = output_dir / "k_effect_on_all_metrics_with_clap_secondary_axis.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved plot: {plot_file}")
    
    print("\nPlot generation complete!")
    return df  # Return the filtered dataframe for further analysis if needed

if __name__ == "__main__":
    e = EvaluationFramework()
    # Example usage with filter:
    filter_params = None  # No filter
    compare = None
    filter_params = {'distance_metric':'euclidean'}
    compare = 'text_encoder'  # Compare by sound preprocessing method

    analyze_experiments(filter_params=filter_params, compare_param=compare)

    pwd = 'pairwise_distance'
    wsd = 'wasserstein_distance'
    cm = ['combined_silhouette_score', 'combined_calinski_harabasz_score', 'combined_davies_bouldin_score']
    
    # Generate plots with specific metrics
    generate_plots(
        filter_params=filter_params,
        clustering_metrics=None,
        mapping_metrics=[pwd],
        compare_param=compare
    )
    
    
    # e.generate_report()  # writes to evaluation_results/evaluation_report.json
