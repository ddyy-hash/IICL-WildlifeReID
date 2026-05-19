#!/usr/bin/env python3

import os
import sys
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import time
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] Faiss is not installed; falling back to brute-force search.")
    print("   Recommended install: pip install faiss-cpu or pip install faiss-gpu")

from config import Config

class FeatureDatabase:
    
    def __init__(self, feature_dim: int = 512, index_type: str = "IVF", use_gpu: bool = False):
        """
        
        Args:
        """
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        
        self.features = []
        self.metadata = []
        self.id_to_idx = {}
        
        self.index = None
        self.is_trained = False
        
        self.total_features = 0
        self.db_stats = {
            'insertions': 0,
            'updates': 0,
            'deletions': 0,
            'queries': 0,
            'avg_query_time': 0.0
        }
        
        self._init_index()
        
        print("[OK] Feature database initialized")
        print(f"   Feature dimension: {self.feature_dim}")
        print(f"   Index type: {self.index_type}")
        print(f"   GPU acceleration: {self.use_gpu}")
        print(f"   Faiss available: {FAISS_AVAILABLE}")
        
    def _init_index(self):
        global FAISS_AVAILABLE
        
        if not FAISS_AVAILABLE:
            print("[WARN] Faiss is unavailable; using brute-force search mode.")
            self.index = None
            return
        
        self.ivf_min_samples = 1000
        self.pending_ivf_switch = False
        self.target_nlist = 100
            
        try:
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatL2(self.feature_dim)
                
            elif self.index_type == "IVF":
                print("[INFO] IVF indexing will activate after the feature count reaches the threshold; using Flat for now.")
                self.index = faiss.IndexFlatL2(self.feature_dim)
                self.pending_ivf_switch = True
                
            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(self.feature_dim, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            if self.use_gpu:
                print("[INFO] Using GPU acceleration")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                
        except Exception as e:
            print(f"[WARN] Failed to initialize the Faiss index: {e}; falling back to brute-force search.")
            self.index = None
            FAISS_AVAILABLE = False
    
    def _maybe_switch_to_ivf(self):
        if not self.pending_ivf_switch or not FAISS_AVAILABLE:
            return
        
        if self.total_features >= self.ivf_min_samples:
            print(f"[INFO] Feature count reached {self.total_features}; switching to an IVF index...")
            try:
                import math
                nlist = min(
                    self.target_nlist,
                    max(4, int(math.sqrt(self.total_features))),
                    self.total_features // 39
                )
                
                quantizer = faiss.IndexFlatL2(self.feature_dim)
                new_index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist)
                new_index.nprobe = min(10, nlist)
                
                train_features = np.array(self.features).astype('float32')
                new_index.train(train_features)
                
                new_index.add(train_features)
                
                self.index = new_index
                self.is_trained = True
                self.pending_ivf_switch = False
                
                print(f"[OK] Switched to the IVF index successfully (nlist={nlist}, nprobe={new_index.nprobe})")
                
            except Exception as e:
                print(f"[WARN] Failed to switch to the IVF index: {e}; continuing with the Flat index.")
    
    def add_features(self, features: np.ndarray, metadata: List[Dict[str, Any]]) -> List[int]:
        """
        
        Args:
            
        Returns:
        """
        if len(features) == 0:
            return []
            
        n_features = features.shape[0]
        feature_ids = []
        
        for i in range(n_features):
            feature_id = self.total_features + i
            feature_ids.append(feature_id)
            
            self.features.append(features[i])
            self.metadata.append(metadata[i])
            self.id_to_idx[feature_id] = len(self.features) - 1
            
            self.db_stats['insertions'] += 1
        
        self.total_features += n_features
        
        self._maybe_switch_to_ivf()
        
        if self.index is not None:
            try:
                if not self.pending_ivf_switch:
                    features_f32 = features.astype('float32')
                    self.index.add(features_f32)
                else:
                    features_f32 = features.astype('float32')
                    self.index.add(features_f32)
                    
            except Exception as e:
                print(f"[WARN] Failed to update the Faiss index: {e}")
        
        print(f"[OK] Added {n_features} features to the database; total={self.total_features}")
        return feature_ids
    
    def update_features(self, feature_id: int, new_features: np.ndarray, new_metadata: Dict[str, Any]) -> bool:
        """
        
        Args:
            
        Returns:
        """
        if feature_id not in self.id_to_idx:
            print(f"[WARN] Feature ID {feature_id} does not exist")
            return False
        
        idx = self.id_to_idx[feature_id]
        
        self.features[idx] = new_features
        self.metadata[idx] = new_metadata
        
        self.db_stats['updates'] += 1
        
        
        print(f"[OK] Updated feature ID {feature_id}")
        return True
    
    def delete_features(self, feature_ids: List[int]) -> int:
        """
        
        Args:
            
        Returns:
        """
        deleted_count = 0
        
        for feature_id in feature_ids:
            if feature_id in self.id_to_idx:
                idx = self.id_to_idx[feature_id]
                
                self.metadata[idx]['deleted'] = True
                self.metadata[idx]['delete_timestamp'] = datetime.now().isoformat()
                
                deleted_count += 1
                self.db_stats['deletions'] += 1
        
        print(f"[OK] Soft-deleted {deleted_count} features")
        return deleted_count
    
    def search(self, query_features: np.ndarray, k: int = 5, threshold: float = 0.0) -> Tuple[List[List[int]], List[List[float]]]:
        """
        
        Args:
            
        Returns:
            (indices_list, distances_list)
        """
        start_time = time.time()
        n_queries = query_features.shape[0]
        
        if self.total_features == 0:
            print("[WARN] The database is empty")
            return [[] for _ in range(n_queries)], [[] for _ in range(n_queries)]
        
        k = min(k, self.total_features)
        
        if self.index is not None and FAISS_AVAILABLE:
            try:
                query_f32 = query_features.astype('float32')
                distances, indices = self.index.search(query_f32, k)
                
                results_ids = []
                results_distances = []
                
                for i in range(n_queries):
                    valid_mask = indices[i] != -1
                    batch_indices = indices[i][valid_mask]
                    batch_distances = distances[i][valid_mask]
                    
                    feature_ids = []
                    for idx in batch_indices:
                        if idx < len(self.features):
                            for fid, fidx in self.id_to_idx.items():
                                if fidx == idx and not self.metadata[fidx].get('deleted', False):
                                    feature_ids.append(fid)
                                    break
                    
                    results_ids.append(feature_ids)
                    results_distances.append(batch_distances.tolist())
                
                query_time = time.time() - start_time
                self._update_query_stats(query_time, n_queries)
                
                return results_ids, results_distances
                
            except Exception as e:
                print(f"[WARN] Faiss search failed: {e}; falling back to brute-force search.")
        
        return self._brute_force_search(query_features, k, threshold)
    
    def _brute_force_search(self, query_features: np.ndarray, k: int, threshold: float) -> Tuple[List[List[int]], List[List[float]]]:
        print("[INFO] Using brute-force search mode")
        
        if not self.features:
            return [[] for _ in range(query_features.shape[0])], [[] for _ in range(query_features.shape[0])]
        
        features_array = np.array(self.features)
        
        results_ids = []
        results_distances = []
        
        for query in query_features:
            distances = np.linalg.norm(features_array - query, axis=1)
            
            if len(distances) > k:
                indices = np.argpartition(distances, k)[:k]
                indices = indices[np.argsort(distances[indices])]
            else:
                indices = np.argsort(distances)
            
            feature_ids = []
            distances_list = []
            
            for idx in indices:
                if not self.metadata[idx].get('deleted', False):
                    for fid, fidx in self.id_to_idx.items():
                        if fidx == idx:
                            feature_ids.append(fid)
                            distances_list.append(float(distances[idx]))
                            break
                
                if len(feature_ids) >= k:
                    break
            
            results_ids.append(feature_ids)
            results_distances.append(distances_list)
        
        return results_ids, results_distances
    
    def batch_search(self, query_features: np.ndarray, k: int = 5, batch_size: int = 100) -> Tuple[List[List[int]], List[List[float]]]:
        """
        
        Args:
            
        Returns:
            (indices_list, distances_list)
        """
        n_queries = query_features.shape[0]
        all_indices = []
        all_distances = []
        
        for i in range(0, n_queries, batch_size):
            batch_queries = query_features[i:i+batch_size]
            batch_indices, batch_distances = self.search(batch_queries, k)
            
            all_indices.extend(batch_indices)
            all_distances.extend(batch_distances)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"[INFO] Batch-search progress: {i+batch_size}/{n_queries}")
        
        return all_indices, all_distances
    
    def get_feature_by_id(self, feature_id: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        
        Args:
            
        Returns:
        """
        if feature_id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[feature_id]
        
        if self.metadata[idx].get('deleted', False):
            return None
        
        return self.features[idx], self.metadata[idx]
    
    def get_features_by_ids(self, feature_ids: List[int]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        
        Args:
            
        Returns:
        """
        results = []
        
        for feature_id in feature_ids:
            result = self.get_feature_by_id(feature_id)
            if result is not None:
                results.append(result)
        
        return results
    
    def save_database(self, save_dir: str = "fea_data"):
        os.makedirs(save_dir, exist_ok=True)
        
        features_path = os.path.join(save_dir, "features_db.npy")
        metadata_path = os.path.join(save_dir, "metadata_db.pkl")
        mapping_path = os.path.join(save_dir, "id_mapping.pkl")
        stats_path = os.path.join(save_dir, "db_stats.pkl")
        
        if self.features:
            features_array = np.array(self.features)
            np.save(features_path, features_array)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.id_to_idx, f)
        
        with open(stats_path, 'wb') as f:
            pickle.dump(self.db_stats, f)
        
        if self.index is not None:
            index_path = os.path.join(save_dir, "faiss_index.bin")
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
        
        print(f"[OK] Database saved to {save_dir}")
        print(f"   Feature count: {self.total_features}")
        print(f"   Index type: {self.index_type}")
        
        return {
            'features': features_path,
            'metadata': metadata_path,
            'mapping': mapping_path,
            'stats': stats_path,
            'index': os.path.join(save_dir, "faiss_index.bin") if self.index is not None else None
        }
    
    def load_database(self, load_dir: str = "fea_data"):
        features_path = os.path.join(load_dir, "features_db.npy")
        metadata_path = os.path.join(load_dir, "metadata_db.pkl")
        mapping_path = os.path.join(load_dir, "id_mapping.pkl")
        stats_path = os.path.join(load_dir, "db_stats.pkl")
        index_path = os.path.join(load_dir, "faiss_index.bin")
        
        if os.path.exists(features_path):
            features_array = np.load(features_path)
            self.features = [features_array[i] for i in range(features_array.shape[0])]
            self.total_features = len(self.features)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.id_to_idx = pickle.load(f)
        
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.db_stats = pickle.load(f)
        
        if os.path.exists(index_path) and FAISS_AVAILABLE:
            try:
                cpu_index = faiss.read_index(index_path)
                if self.use_gpu:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                else:
                    self.index = cpu_index
                
                self.is_trained = True
                print("Faiss index loaded successfully.")
            except Exception as e:
                print(f"Faiss index loading failed: {e}")
                self._init_index()
        
        print(f"Database loaded successfully; feature count={self.total_features}")
        
        return self.total_features > 0
    
    def _update_query_stats(self, query_time: float, n_queries: int):
        self.db_stats['queries'] += n_queries
        
        total_time = self.db_stats['avg_query_time'] * (self.db_stats['queries'] - n_queries)
        total_time += query_time
        self.db_stats['avg_query_time'] = total_time / self.db_stats['queries']
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_features': self.total_features,
            'index_type': self.index_type,
            'faiss_available': FAISS_AVAILABLE,
            'use_gpu': self.use_gpu,
            'is_trained': self.is_trained,
            **self.db_stats
        }
        
        return stats
    
    def rebuild_index(self):
        if not FAISS_AVAILABLE or len(self.features) == 0:
            return
        
        print(f"Rebuilding the Faiss index (features={len(self.features)})")
        
        self._init_index()
        
        if self.index_type == "IVF" and not self.is_trained and len(self.features) >= 100:
            features_array = np.array(self.features).astype('float32')
            self.index.train(features_array)
            self.is_trained = True
        
        if self.index_type != "IVF" or self.is_trained:
            features_array = np.array(self.features).astype('float32')
            self.index.add(features_array)
        
        print("Index rebuild complete.")


_global_feature_db = None

def get_feature_database(feature_dim: int = 512, index_type: str = "IVF", use_gpu: bool = False) -> FeatureDatabase:
    """
    
    Args:
        
    Returns:
    """
    global _global_feature_db
    
    if _global_feature_db is None:
        _global_feature_db = FeatureDatabase(feature_dim, index_type, use_gpu)
    
    return _global_feature_db


def initialize_feature_database_from_config(config_path: str = 'config/illumination_config.yaml') -> FeatureDatabase:
    """
    
    Args:
        
    Returns:
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    training_cfg = config.get('training', {})
    metric_learning_cfg = training_cfg.get('metric_learning', {})
    
    feature_dim = 512
    
    db = get_feature_database(
        feature_dim=feature_dim,
        index_type="IVF",
        use_gpu=False
    )
    
    db.load_database("fea_data")
    
    return db


if __name__ == "__main__":
    print("Testing the feature database")
    
    db = FeatureDatabase(feature_dim=128, index_type="Flat")
    
    n_test = 1000
    test_features = np.random.randn(n_test, 128).astype('float32')
    test_metadata = [{'id': i, 'label': f'dog_{i}', 'timestamp': datetime.now().isoformat()} for i in range(n_test)]
    
    feature_ids = db.add_features(test_features, test_metadata)
    print(f"Added {len(feature_ids)} test features")
    
    query_features = np.random.randn(5, 128).astype('float32')
    indices, distances = db.search(query_features, k=5)
    print(f"Search complete; returned {len(indices)} result sets")
    
    stats = db.get_stats()
    print(f"Database statistics: {stats}")
    
    db.save_database("test_db")
    print("Database save complete")
    
    db2 = FeatureDatabase(feature_dim=128, index_type="Flat")
    db2.load_database("test_db")
    print(f"Database load complete; feature count={db2.total_features}")
    
    print("Feature-database smoke test complete")
