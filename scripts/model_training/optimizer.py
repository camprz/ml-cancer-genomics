import umap
import numpy as np
from sklearn.metrics import silhouette_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.cluster import HDBSCAN
import joblib
import random 

random.seed(2024) 

class UMAPHDBSCANOptimizer:
    def __init__(self, data):
        self.data = data
        self.results = []
        self.best_params = None
        self.best_score = None
    
    def optimize(self, params):
        n_neighbors = int(params['n_neighbors'])
        min_samples = int(params['min_samples'])
        min_cluster_size = int(params['min_cluster_size'])
        
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=3, random_state=2024)
        embedding = umap_model.fit_transform(self.data)
        
        clusterer = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embedding)
        
        if len(np.unique(labels)) <= 1:
            score = 0  # All points are considered noise
        else:
            score = silhouette_score(self.data, labels)
        
        # Record the parameters and their corresponding score
        self.results.append((n_neighbors, min_samples, min_cluster_size, score))
        
        return {'loss': -score, 'status': STATUS_OK}
    
    def run_optimization(self, max_evals=100):
        search_space = {
            'n_neighbors': hp.quniform('n_neighbors', 2, 50, 1),
            'min_samples': hp.quniform('min_samples', 2, 50, 1),
            'min_cluster_size': hp.quniform('min_cluster_size', 2, 50, 1)
        }
        
        trials = Trials()
        self.best_params = fmin(
            fn=self.optimize,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        self.best_score = -min(trial['result']['loss'] for trial in trials.trials)
        return self.best_params, self.best_score, self.results

    def save_models(self, pipeline_name):
        umap_model = umap.UMAP(n_neighbors=int(self.best_params["n_neighbors"]), min_dist=0.0, n_components=3, random_state=2024)
        embedding = umap_model.fit_transform(self.data)
        umap_filename = f'./models/trained_models/umap_{pipeline_name}.sav'
        joblib.dump(umap_model, umap_filename)

        clusterer = HDBSCAN(min_samples=int(self.best_params["min_samples"]), 
                                    min_cluster_size=int(self.best_params["min_cluster_size"]))
        labels = clusterer.fit_predict(embedding)
        hdbscan_filename = f'./models/trained_models/hdbscan_{pipeline_name}.sav'
        joblib.dump(clusterer, hdbscan_filename)
        print("Saved models in ./models/trained_models")
        
        return embedding, labels
        
# Example usage:
# optimizer = UMAPHDBSCANOptimizer(umap_df)
# best_params, best_score = optimizer.run_optimization(max_evals=100)
# pipeline_name = "example_pipeline"
# umap_filename, hdbscan_filename = optimizer.save_models(pipeline_name)