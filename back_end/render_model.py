import torch
from utils import load_ckpt, visualize_depth
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import imageio
import os
import argparse
from tqdm import tqdm

from models.rendering import *
from models.nerf import *

import metrics

from datasets import dataset_dict

torch.backends.cudnn.benchmark = True


# Change to your settings...
############################
N_vocab = 1500
encode_appearance = True
N_a = 48
encode_transient = True
N_tau = 16
beta_min = 0.1 # doesn't have effect in testing
# ckpt_path = 'ckpts/brandenburg_scale8_nerfw/epoch=19.ckpt'
# ckpt_path = 'ckpts/sacre_scale8_nerfw/epoch=30.ckpt'
# ckpt_path = 'ckpts/brandenburg_gate_2/epoch=8.ckpt'
N_emb_xyz = 10
N_emb_dir = 4
N_samples = 256
N_importance = 256
use_disp = False
chunk = 1024*32
#############################


class ModelRender():
    def __init__(self, ckpt_path:str, test_appearance_idx:int, dataset_path:str, output_path:str):
        self.ckpt_path = ckpt_path
        self.test_appearance_idx = test_appearance_idx
        self.dataset_path = dataset_path
        self.output_path = output_path

        embedding_xyz = PosEmbedding(N_emb_xyz-1, N_emb_xyz)
        embedding_dir = PosEmbedding(N_emb_dir-1, N_emb_dir)
        embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
        if encode_appearance:
            embedding_a = torch.nn.Embedding(N_vocab, N_a).cuda()
            load_ckpt(embedding_a, ckpt_path, model_name='embedding_a')
            embeddings['a'] = embedding_a
        if encode_transient:
            embedding_t = torch.nn.Embedding(N_vocab, N_tau).cuda()
            load_ckpt(embedding_t, ckpt_path, model_name='embedding_t')
            embeddings['t'] = embedding_t
        self.embeddings = embeddings
            

        nerf_coarse = NeRF('coarse',
                        in_channels_xyz=6*N_emb_xyz+3,
                        in_channels_dir=6*N_emb_dir+3).cuda()
        nerf_fine = NeRF('fine',
                        in_channels_xyz=6*N_emb_xyz+3,
                        in_channels_dir=6*N_emb_dir+3,
                        encode_appearance=encode_appearance,
                        in_channels_a=N_a,
                        encode_transient=encode_transient,
                        in_channels_t=N_tau,
                        beta_min=beta_min).cuda()

        load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
        load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

        self.models = {'coarse': nerf_coarse, 'fine': nerf_fine}

        dataset = dataset_dict['phototourism'] \
                (dataset_path,
                split='test_train',
                img_downscale=2, use_cache=True)

        dir_name = self.output_path
        os.makedirs(dir_name, exist_ok=True)
        self.dir_name = dir_name

        # define testing camera intrinsics (hard-coded, feel free to change)
        dataset.test_img_w, dataset.test_img_h = 320, 240
        dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
        dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                [0, dataset.test_focal, dataset.test_img_h/2],
                                [0,                  0,                    1]])

        # select appearance embedding, hard-coded for each scene
        dataset.test_appearance_idx = test_appearance_idx # 85572957_6053497857.jpg
        N_frames = 30*4
        dx = np.concatenate((
            np.linspace(0, 0.03, N_frames - 30), np.linspace(0.03, 0.05, 30)
        ))
        dy = np.concatenate((
            np.linspace(0, 0.7, N_frames - 30), np.linspace(0.7, 0.6, 30)
        ))
        dz = np.concatenate((
            np.linspace(0, 1.8, N_frames - 30), np.linspace(1.8, 1.9, 30)
        ))
        # define poses
        dataset.poses_test = np.tile(dataset.poses_dict[1123], (N_frames, 1, 1))
        for i in range(N_frames):
            dataset.poses_test[i, 0, 3] += dx[i]
            dataset.poses_test[i, 1, 3] += dy[i]
            dataset.poses_test[i, 2, 3] += dz[i]
        
        self.dataset = dataset
        self.dx = dx

    @torch.no_grad()
    def f(self, rays, ts, **kwargs):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, chunk):
            kwargs_ = {}
            if 'a_embedded' in kwargs:
                kwargs_['a_embedded'] = kwargs['a_embedded'][i:i+chunk]
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+chunk],
                            ts[i:i+chunk],
                            N_samples,
                            use_disp,
                            0,
                            0,
                            N_importance,
                            chunk,
                            self.dataset.white_back,
                            test_time=True,
                            **kwargs_)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    @torch.no_grad()
    def batched_inference(self, models, embeddings,
                        rays, ts, N_samples, N_importance, use_disp,
                        chunk,
                        white_back,
                        **kwargs):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                render_rays(models,
                            embeddings,
                            rays[i:i+chunk],
                            ts[i:i+chunk] if ts is not None else None,
                            N_samples,
                            use_disp,
                            0,
                            0,
                            N_importance,
                            chunk,
                            white_back,
                            test_time=True,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def render_loop(self):
        dataset = self.dataset
        dx = self.dx
        imgs, psnrs = [], []
        # render
        for i in tqdm(range(min(len(dataset), len(dx)))):
            sample = dataset[i]
            rays = sample['rays']
            ts = sample['ts']
            results = self.batched_inference(self.models, self.embeddings, rays.cuda(), ts.cuda(),
                                        256, 256, False,
                                        16384,
                                        dataset.white_back)
            w, h = sample['img_wh']
            
            img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
            
            img_pred_ = (img_pred*255).astype(np.uint8)
            imgs += [img_pred_]
            imageio.imwrite(os.path.join(self.dir_name, f'{i:03d}.png'), img_pred_)
            
            if 'rgbs' in sample:
                rgbs = sample['rgbs']
                img_gt = rgbs.view(h, w, 3)
                psnr = metrics.psnr(img_gt, img_pred).item()
                print(f"psnr: {psnr}")
                psnrs += [psnr]
                mean_psnr = np.mean(psnrs)
                print(f'Mean PSNR : {mean_psnr:.2f}')

            imageio.mimsave(os.path.join(self.dir_name, f'render_result.gif'),
                                imgs, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--appearence_idx", type=int)
    args = parser.parse_args()

    renderer = ModelRender(args.ckpt_path, args.appearence_idx, args.data_path, args.output_path)
    renderer.render_loop()
