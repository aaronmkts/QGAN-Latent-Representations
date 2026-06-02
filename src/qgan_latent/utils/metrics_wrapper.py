from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from .metrics import NDB  
from skimage.metrics import structural_similarity, peak_signal_noise_ratio 
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

class MetricsManager:
    def __init__(self, cfg, real_images: np.ndarray | jnp.ndarray):
        self.metrics: dict[str, BaseMetric] = {}
        
        active_list = cfg.get('active_metrics', [])

        if 'ndb_jsd' in active_list:
            self.metrics['ndb_jsd'] = NDB_JSD_Metric(
                real_images=real_images,
                number_of_bins=cfg.get('ndb_jsd', {}).get('number_of_bins', 100),
                significance_level=cfg.get('ndb_jsd', {}).get('significance_level', 0.05),
                whitening=cfg.get('ndb_jsd', {}).get('whitening', True),
                max_dims=cfg.get('ndb_jsd', {}).get('max_dims', None)
            )
            
        if 'ssim' in active_list:
            self.metrics['ssim'] = SSIMMetric(real_images=real_images)
            
        if 'psnr' in active_list:
            self.metrics['psnr'] = PSNRMetric(real_images=real_images)
            
        if 'cos_sim' in active_list:
            self.metrics['cos_sim'] = CosineSimilarityMetric(real_images=real_images)

    def update(self, fake_images: np.ndarray | jnp.ndarray):
        for metric in self.metrics.values():
            metric.update_fake_images(fake_images)

    def compute(self) -> dict[str, float]:
        results = {}
        if not self.metrics:
            return results
        
        # OPTIMIZATION: Execute metric computations in parallel threads
        # This helps if any metric is blocking (though SSIM/PSNR are now multi-process via joblib)
        with ThreadPoolExecutor(max_workers=len(self.metrics)) as executor:
            future_to_metric = {
                executor.submit(metric.compute): name 
                for name, metric in self.metrics.items()
            }
            
            for future in as_completed(future_to_metric):
                name = future_to_metric[future]
                try:
                    res = future.result()
                    
                    if isinstance(res, dict):
                        if 'NDB' in res:
                            # NDB is returned as count, we might want to normalize it
                            n_bins = self.metrics[name].ndb_evaluator.number_of_bins
                            results[f"val/NDB_K"] = float(res['NDB'] / n_bins)
                        if 'JS' in res:
                            results[f"val/JS"] = float(res['JS'])
                    else:
                        results[f"val/{name.upper()}"] = float(res)
                        
                except Exception as exc:
                    print(f'{name} generated an exception: {exc}')

        return results

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()


class BaseMetric(ABC):
    def __init__(self, real_images: np.ndarray | jnp.ndarray = None):
        """
        Base class for GAN metrics.
        Args:
            real_images: The static validation/test set images. 
                         If provided here, they are stored permanently.
        """
        self.real_images = None
        self.fake_images = None
        
        # Initialize real images immediately if provided
        if real_images is not None:
            self.set_real_images(real_images)

    def set_real_images(self, images: np.ndarray | jnp.ndarray):
        """
        Set the real images (ground truth). This usually happens once before training.
        Converts to numpy and ensures shape is (N, 28, 28) for image metrics.
        """
        imgs = np.array(images)
        # Squeeze channel dim if present (N, 28, 28, 1) -> (N, 28, 28)
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs.squeeze(-1)
        self.real_images = imgs

    def update_fake_images(self, images: np.ndarray | jnp.ndarray):
        """
        Update the fake images for the current epoch.
        """
        imgs = np.array(images)
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs.squeeze(-1)
        self.fake_images = imgs

    def reset(self):
        """
        Clears the fake images buffer. Real images are kept.
        """
        self.fake_images = None

    def _get_flat_features(self, images):
        """Helper to flatten images for NDB/CosSim."""
        return images.reshape(images.shape[0], -1)

    @abstractmethod
    def compute(self) -> float | dict:
        pass


class NDB_JSD_Metric(BaseMetric):
    def __init__(self, real_images=None, number_of_bins=100, significance_level=0.05, 
                 z_threshold=None, whitening=False, max_dims=None, cache_folder=None):
        super().__init__(real_images)
        self.ndb_evaluator = NDB(
            training_data=None, # Will be set via construct_bins
            number_of_bins=number_of_bins,
            significance_level=significance_level,
            z_threshold=z_threshold,
            whitening=whitening,
            max_dims=max_dims,
            cache_folder=cache_folder
        )
        self.bins_initialized = False

    def compute(self):
        if self.real_images is None or self.fake_images is None:
            return {'NDB': 0, 'JS': 0}

        flat_real = self._get_flat_features(self.real_images)
        flat_fake = self._get_flat_features(self.fake_images)

        # Initialize bins using real data (only needs to be done once technically, 
        # but re-doing ensures safety if real_images changed)
        if not self.bins_initialized:
            self.ndb_evaluator.construct_bins(flat_real, bins_file=None)
            self.bins_initialized = True
        
        results = self.ndb_evaluator.evaluate(flat_fake)
        return results


class SSIMMetric(BaseMetric):
    def compute(self) -> float:
        if self.real_images is None or self.fake_images is None:
            return 0.0
        
        n = min(len(self.real_images), len(self.fake_images))
        real = self.real_images[:n]
        fake = self.fake_images[:n]

        # OPTIMIZATION: Parallelize the loop using joblib
        # n_jobs=-1 uses all available cores
        ssim_values = Parallel(n_jobs=-1)(
            delayed(structural_similarity)(real[i], fake[i], data_range=1.0)
            for i in range(n)
        )
            
        return float(np.mean(ssim_values))


class PSNRMetric(BaseMetric):
    def compute(self) -> float:
        if self.real_images is None or self.fake_images is None:
            return 0.0
        
        n = min(len(self.real_images), len(self.fake_images))
        real = self.real_images[:n]
        fake = self.fake_images[:n]

        # OPTIMIZATION: Parallelize the loop using joblib
        psnr_values = Parallel(n_jobs=-1)(
            delayed(peak_signal_noise_ratio)(real[i], fake[i], data_range=1.0)
            for i in range(n)
        )
            
        return float(np.mean(psnr_values))

class CosineSimilarityMetric(BaseMetric):
    def compute(self) -> float:
        if self.real_images is None or self.fake_images is None:
            return 0.0

        v1 = self._get_flat_features(self.real_images)
        v2 = self._get_flat_features(self.fake_images)
        
        norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
        
        # Avoid division by zero
        norm1[norm1 == 0] = 1e-8
        norm2[norm2 == 0] = 1e-8
        
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        # Cosine similarity matrix (N_real, N_fake)
        sim_matrix = np.dot(v1_norm, v2_norm.T)
        
        # Rescale to [0, 1]
        res = 0.5 + 0.5 * sim_matrix
        
        return float(np.mean(res))
    
