# Towards Latent Masked Image Modeling for Self-Supervised Visual Representation Learning (ECCV 2024)
<a href=""><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a> &nbsp;
<a href='https://yibingwei-1.github.io/projects/lmim/lmim.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

## Pre-trained Model
<table>
  <tr>
    <th colspan="1">arch.</th>
    <th colspan="1">epochs</th>
    <th colspan="1">data</th>
    <th colspan="3">checkpoint</th>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>800</td>
    <td>ImageNet-1K</td>
    <td><a href="">download</a></td>
  </tr>
  </tr>
</table>

## Pre-train Latent MIM
### Requirements
* Python 3.8 (or newer)
* PyTorch 2.1
* torchvision
* Other dependencies: hydra-core, numpy, scipy, submitit, wandb, timm

```bash
PYTHONPATH=. python launcher.py -m --config-name=lmim \
  worker=main_lmim \
  encoder=vit_base decoder_depth=3 avg_sim_coeff=0.1 loss=infonce_patches patch_gap=4 \
  epochs=800 warmup_epochs=40 blr=1.5e-4 min_lr_frac=0.25 weight_decay=0.05 \
  batch_size=256 accum_iter=2 env.ngpu=8 \
  dataset=imagenet resume=True \
  output_dir=./checkpoints \
  data_path=/path/to/imagenet \
  env.slurm=True env.distributed=True
```

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bib
@inproceedings{linz2024asva,
    title={Audio-Synchronized Visual Animation},
    author={Lin Zhang and Shentong Mo and Yijing Zhang and Pedro Morgado},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2024}
}
```