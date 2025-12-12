from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional
import io

import torch
from PIL import Image, ImageStat

from config import Settings
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline
from schemas import TrellisResult, TrellisRequest, TrellisParams
import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np

class TrellisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.trellis_model_id
        )
        self.pipeline.cuda()
        logger.success("Trellis pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        trellis_request: TrellisRequest,
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        image_rgb = trellis_request.image.convert("RGB")
        logger.info(f"Generating Trellis {trellis_request.seed=} and image size {trellis_request.image.size}")

        params = self.default_params.overrided(trellis_request.params)

        start = time.time()
        buffer = None
        try:
            outputs = self.pipeline.run(
                image_rgb,
                seed=trellis_request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                    
                },
                preprocess_image=False,
                formats=["gaussian"],
                num_oversamples=params.num_oversamples,
            )

            generation_time = time.time() - start
            gaussian = outputs["gaussian"][0]


            temp_ply = "temp_before_refine.ply"
            gaussian.save_ply(temp_ply)
            
            # Load point cloud with Open3D
            pcd = o3d.io.read_point_cloud(temp_ply)
            num_points_before = len(pcd.points)
            
            # Apply statistical outlier removal if we have enough points
            logger.warning(f"Refining PLY file with Open3D: {temp_ply}")
            
            # Statistical outlier removal - parameters from config
            # nb_neighbors: number of neighbors to consider (higher = smoother)
            # std_ratio: threshold (higher = keeps more points, preserves geometry)
            pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            
            num_points_after = len(pcd_filtered.points)
            removed_points = num_points_before - num_points_after
            
            logger.warning(f"Outlier removal: {removed_points} points removed ({removed_points/num_points_before*100:.1f}%)")
            
            # Convert inlier_indices to numpy array for filtering
            inlier_mask = np.array(inlier_indices)
            
            # Load original PLY with plyfile to preserve all Gaussian Splatting properties
            plydata = PlyData.read(temp_ply)
            vertex = plydata['vertex']
            
            # Filter vertices using inlier mask
            filtered_vertex = vertex[inlier_mask]
            
            # Create new PLY with filtered data
            refined_ply = temp_ply.replace('.ply', '_refined.ply')
            new_vertex = PlyElement.describe(filtered_vertex, 'vertex')
            PlyData([new_vertex], text=False).write(refined_ply)
            
            logger.warning(f"Refined PLY saved with {num_points_after} points (preserved all Gaussian Splatting properties)")
            # Apply inlier mask directly to gaussian internal tensors to drop outliers
            # without reloading from PLY or changing coordinate frames.
            try:
                # Ensure mask is boolean tensor on correct device
                mask_torch = torch.tensor(inlier_mask, dtype=torch.bool, device=gaussian.device)

                # Slice all gaussian internal tensors consistently
                if hasattr(gaussian, '_xyz') and gaussian._xyz is not None:
                    gaussian._xyz = gaussian._xyz[mask_torch]
                if hasattr(gaussian, '_features_dc') and gaussian._features_dc is not None:
                    gaussian._features_dc = gaussian._features_dc[mask_torch]
                if hasattr(gaussian, '_features_rest') and gaussian._features_rest is not None:
                    gaussian._features_rest = gaussian._features_rest[mask_torch]
                if hasattr(gaussian, '_scaling') and gaussian._scaling is not None:
                    gaussian._scaling = gaussian._scaling[mask_torch]
                if hasattr(gaussian, '_rotation') and gaussian._rotation is not None:
                    gaussian._rotation = gaussian._rotation[mask_torch]
                if hasattr(gaussian, '_opacity') and gaussian._opacity is not None:
                    gaussian._opacity = gaussian._opacity[mask_torch]

                logger.warning("Applied inlier mask to gaussian tensors without reload")
            except Exception as e:
                logger.warning(f"Failed to apply inlier mask to gaussian tensors: {e}")

            # Clean up temporary files
            if os.path.exists(temp_ply):
                os.remove(temp_ply)
            if os.path.exists(refined_ply):
                os.remove(refined_ply)

            # Save ply to buffer
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None # bytes
            )

            logger.success(f"Trellis finished generation in {generation_time:.2f}s.")
            return result
        finally:
            if buffer:
                buffer.close()

