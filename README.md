<p align="center">
<!--   <h1 align="center"><img height="100" src="https://github.com/imlixinyang/director3d-page/raw/master/assets/icon.ico"></h1> -->
  <h2 align="center"> <b> SynergyAmodal 😷⇒☺️ </b>: Deocclude Anything with Text Control</h2>
  <p align="center">
<!--         <a href="https://arxiv.org/pdf/2406.17601"><img src='https://img.shields.io/badge/arXiv-Director3D-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://imlixinyang.github.io/director3d-page'><img src='https://img.shields.io/badge/Project_Page-Director3D-green' alt='Project Page'></a>
        <a href='https://colab.research.google.com/drive/1LtnxgBU7k4gyymOWuonpOxjatdJ7AI8z?usp=sharing'><img src='https://img.shields.io/badge/Colab_Demo-Director3D-yellow?logo=googlecolab' alt='Project Page'></a> -->
</p>

![teaser_](https://github.com/user-attachments/assets/6f94e6b0-f26e-426b-aa84-7429f0f854b5)

**⭐ Key components of SynergyAmodal**:

- A full completion diffusion model achieving zero-shot generalization and textual controllability.

**🔥 News**:

- 🥰 Check out our new gradio demo by simply running ```python app.py```.

- 😊 Our paper is accepted by ACMMM 2025.

## Qualitative comparison: SynergyAmodal vs pretrained SD2 inpainting

Local comparison run on **2026-10-30**: the released **SynergyAmodal EMA model**
versus its pretrained **Stable Diffusion 2 inpainting** backbone. This comparison
does not use the experimental PixArt model.

There are **8 examples × 2 seeds × 2 text-CFG settings × 2 models = 64 generated
images**. The first two examples reproduce the boat/player inputs from the
PixArt training previews; the remaining six are deterministically selected
Synergy16K cases. Every result is included below, without selecting the best seed.

Columns: **scene → visible mask → dataset target → SD2 CFG 1 → Synergy CFG 1 →
SD2 CFG 7.5 → Synergy CFG 7.5**. Click an image to inspect the full-size sheet.

### Seed 45

![Seed 45: boat, basketball player, tent, vegetables](../all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison/comparison-seed-45-page-1.png)

![Seed 45: soccer player, bus, horse, camper](../all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison/comparison-seed-45-page-2.png)

### Seed 46

![Seed 46: boat, basketball player, tent, vegetables](../all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison/comparison-seed-46-page-1.png)

![Seed 46: soccer player, bus, horse, camper](../all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison/comparison-seed-46-page-2.png)

### Comparison protocol

- Both models receive the same observed image, visible mask, caption, **actual
  initial noise tensor**, and 25-step DDIM schedule (`999, 959, …, 39`) at 512px.
  The inpainting mask is the complement of the visible-object mask.
- Text CFG is matched at 1 and 7.5; Synergy's image CFG stays at 1. The models use
  their native image conditioning: Synergy adds scene RGB and uses white outside
  the visible object; SD2 uses its native zero-masked normalized RGB. This is a
  comparison of the two models as designed, not a weights-only ablation.
- These are **raw RGB outputs from one global pass**. There is no mask-decoder
  cleanup, visible-pixel replacement, or final compositing. The demo's optional
  second local refinement pass is not used. Targets are displayed only and are
  never passed to either model.
- The six additional crops are framed from the **visible mask only**, with a
  50% margin and minimum source crop size of 512px. Selection uses annotated
  hidden-area fractions of 15–75%. Dataset targets are pseudo-ground truth, and
  these cases may overlap the released model's training set; this is not a
  held-out benchmark.
- SD2 weights come from the documented
  [community mirror](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting),
  pinned to `5f74973cbb64c8568780732c17f43eb269d63a0d`, because the original
  Stability AI endpoint is deprecated. Downloaded weight SHA256s were verified.
  After legacy parameter-name conversion, all **248 VAE tensors and 372 text
  encoder tensors match SynergyAmodal's stored frozen weights exactly**.

**What these examples show:** SD2 inpainting already preserves substantial
visible appearance, including the basketball player's blue uniform, but often
fills the surrounding scene, enlarges objects, or invents extra structure.
SynergyAmodal more consistently isolates and completes the requested object on
white. It is not perfect: tent interiors and hidden player/horse details vary
with seed and guidance. These examples support the distinction between generic
inpainting capability and task-specific amodal completion; they do not establish
that initialization alone explains the experimental PixArt model's failures.

[Output directory and run notes](../all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison/README.md) ·
[Per-case records and checkpoint provenance](../all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison/run.json) ·
[Reproduction script](scripts/compare_pretrained.py)

From the parent workspace, using the existing environment and a free GPU:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
  /home/minht/miniconda3/envs/pix2ges/bin/python \
  SynergyAmodal/scripts/compare_pretrained.py \
  --output all_data/outputs/synergy_amodal/outputs/qualitative/pretrained-inpainting-comparison-new
```

The figures are local artifact links through `all_data`; they require this
workspace's artifact directory and are not uploaded to GitHub.

## 🔧 Installation
- create a new conda enviroment
```
conda create -n synergyamodal python=3.10
conda activate synergyamodal
```

- install pytorch (or use your own if it is compatible with ```xformers```)
```
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

- install ```xformers``` for momory-efficient attention
```
conda install xformers -c xformers
```

- install ```pip``` packages
```
pip install -r requirements.txt
```

- clone this repo:
```
git clone https://github.com/imlixinyang/SynergyAmodal.git
cd SynergyAmodal
```

## Model
Download the pre-trained model by:
```
wget https://huggingface.co/cloudyfall/DeoccAnything/resolve/main/vae_ckpt_dir/epoch%3D5-step%3D100000.ckpt?download=true -O vae.ckpt
wget https://huggingface.co/cloudyfall/DeoccAnything/resolve/main/ldm_ckpt_dir/epoch%3D8-step%3D58000.ckpt?download=true -O ldm.ckpt
```

Download the pre-trained weights of ZIM from [the official ZIM repository](https://github.com/naver-ai/ZIM). We thank the ZIM team for this excellent work.

## Dataset
Download and extract our dataset with:
```
wget https://huggingface.co/datasets/cloudyfall/SynergyAmodal16K/resolve/main/dataset.tar.gz
tar -xzvf dataset.tar.gz
```

The original images are from [the official EntitySeg repository](https://github.com/adobe-research/EntitySeg-Dataset). Please follow its licences, which only allows for non-commercial research use.

## Citation

```
@article{li2025synergyamodal,
  title={SynergyAmodal: Deocclude Anything with Text Control},
  author={Li, Xinyang and Yi, Chengjie and Lai, Jiawei and Lin, Mingbao and Qu, Yansong and Zhang, Shengchuan and Cao, Liujuan},
  journal={arXiv preprint arXiv:2504.19506},
  year={2025}
}
```


## License

Our code and annotations are released under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) Licensed, for academic research use only. 

If you have any questions, please contact me via [imlixinyang@gmail.com](mailto:imlixinyang@gmail.com). 
