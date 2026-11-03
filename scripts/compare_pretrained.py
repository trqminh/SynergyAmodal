"""Matched qualitative comparison: released SynergyAmodal vs SD2 inpainting.

Single global RGB pass, no mask decoder/post-compositing, no training.
Uses the existing verified release loader in diagnostics/run_synergy_qualitative.py.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'pixart-alpha-amodal'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from safetensors import safe_open
from safetensors.torch import save_file
from diagnostics.run_synergy_qualitative import ReleasedModel, sha256, save_tensor

REVISION = '5f74973cbb64c8568780732c17f43eb269d63a0d'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def tensor_image(im):
    return torch.from_numpy(np.array(im, copy=True)).permute(2,0,1).float()/255


def prepare_cases(data, output):
    from pixart_amodal.synergy16k_data import Synergy16K
    dataset = Synergy16K(data, ROOT/'SynergyAmodal', seed=42, length_multiplier=1)
    cases = []
    # Exact two fixed inputs from the ongoing PixArt run's training previews.
    for index in [0,8073]:
        item = dataset[(0,index)]
        cases.append(dict(id=f'preview-{index}',annotation_index=index,caption=item['caption'],
            image=item['image'],visible=item['modal_mask'],target=item['amodal_image'],
            framing='same deterministic augmented frame as PixArt training preview',
            image_id=item['image_id'],instance_id=item['instance_id']))
    annotations=json.loads((data/'amodal_annotations.json').read_text())
    order=list(range(len(annotations)));random.Random(42).shuffle(order)
    seen={annotations[i]['filename'] for i in [0,8073]}
    for index in order:
        ann=annotations[index]
        if 'amodal_rgb' not in ann or ann['filename'] in seen:continue
        x,y,w,h=ann['bbox']
        if min(w,h)<96 or any(ann[k]['size'] != [h,w] for k in ['modal_mask','amodal_mask']):continue
        vm=mask_utils.decode(ann['modal_mask']).astype(bool)
        am=mask_utils.decode(ann['amodal_mask']).astype(bool)
        if not vm.any() or not am.any():continue
        hidden=float((am & ~vm).sum()/am.sum())
        if not .15 <= hidden <= .75:continue
        with Image.open(data/'images'/ann['filename']) as im:scene=im.convert('RGB').copy()
        with Image.open(data/'amodal_rgbs'/ann['amodal_rgb']) as im:target_crop=im.convert('RGB').copy()
        if target_crop.size!=(w,h):continue
        full_visible=Image.new('L',scene.size,0)
        full_visible.paste(Image.fromarray(vm.astype(np.uint8)*255),(x,y))
        full_target=Image.new('RGB',scene.size,'white')
        target_crop=Image.fromarray(np.where(am[...,None],np.array(target_crop),255).astype(np.uint8))
        full_target.paste(target_crop,(x,y))
        # Frame using only the visible object; never use amodal extent to crop.
        vx0,vy0,vx1,vy1=full_visible.getbbox()
        side=max(512,int(np.ceil(1.5*max(vx1-vx0,vy1-vy0))))
        left=int((vx0+vx1-side)//2);top=int((vy0+vy1-side)//2)
        rectangle=(left,top,left+side,top+side)
        def crop(im,background,resampling):
            canvas=Image.new(im.mode,(side,side),background)
            canvas.paste(im,(-left,-top))
            return canvas.resize((512,512),resampling)
        rgb=crop(scene,'black',Image.Resampling.BILINEAR)
        visible=crop(full_visible,0,Image.Resampling.NEAREST)
        target=crop(full_target,'white',Image.Resampling.BILINEAR)
        cases.append(dict(id=f'dataset-{index}',annotation_index=index,caption=ann['captions'][0],
            image=tensor_image(rgb),visible=torch.from_numpy(np.array(visible,copy=True))[None].float()/255,
            target=tensor_image(target),framing='visible-mask bounding square, 50% margin, min 512 source pixels',
            crop_xyxy=list(rectangle),hidden_fraction_selection=hidden,
            image_id=ann['image_id'],instance_id=ann['instance_id']))
        seen.add(ann['filename'])
        if len(cases)==8:break
    if len(cases)!=8:raise ValueError('Expected eight cases')
    for case in cases:
        folder=output/'inputs'/case['id'];folder.mkdir(parents=True)
        save_tensor(case['image'],folder/'scene.png')
        save_tensor(case['visible'],folder/'visible_mask.png')
        save_tensor(1-case['visible'],folder/'inpaint_mask.png')
        save_tensor(case['target'],folder/'target.png')
        save_tensor(case['image']*case['visible']+1-case['visible'],folder/'visible_object.png')
        info={k:v for k,v in case.items() if not torch.is_tensor(v)}
        info['input_sha256']={p.name:sha256(p) for p in folder.glob('*.png')}
        write_json(folder/'case.json',info)
    return cases


def check_frozen_weights(base, weights):
    """Verify the release really retained the original frozen VAE and CLIP."""
    checkpoint=torch.load(weights/'ldm.ckpt',map_location='cpu',mmap=True,weights_only=False)['state_dict']
    audit={}
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel
    for component,cls,prefix in [('vae',AutoencoderKL,'vae.'),
                                  ('text_encoder',CLIPTextModel,'text_encoder.')]:
        maximum=0.;count=0
        # Diffusers converts legacy VAE attention names (key -> to_k, etc.).
        original=cls.from_pretrained(str(base/component),torch_dtype=torch.float32,
            local_files_only=True,use_safetensors=True)
        for key,source in original.state_dict().items():
            if prefix+key not in checkpoint:raise KeyError(prefix+key)
            stored=checkpoint[prefix+key]
            error=float((source.float()-stored.float()).abs().max())
            maximum=max(maximum,error);count+=1
        del original
        if maximum>1e-5:raise ValueError(f'Frozen {component} differs from base: {maximum}')
        audit[component]={'tensors_compared':count,'max_abs_difference':maximum}
    return audit


def render(output,cases,seeds):
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,16);small=ImageFont.truetype(fontpath,14)
    headers=['Observed scene','Visible mask','Dataset target','SD2 inpaint CFG 1','Synergy CFG 1',
             'SD2 inpaint CFG 7.5','Synergy CFG 7.5']
    for seed in seeds:
        for group in range(2):
            selected=cases[group*4:group*4+4]
            sheet=Image.new('RGB',(7*256,64+4*304),'white');draw=ImageDraw.Draw(sheet)
            draw.text((10,7),f'SynergyAmodal vs pretrained SD2 inpainting | seed {seed} | DDIM25 | raw RGB, no compositing',font=font,fill='black')
            for col,label in enumerate(headers):draw.text((col*256+6,37),label,font=small,fill='black')
            for row,case in enumerate(selected):
                y=64+row*304
                draw.text((8,y+5),f"{case['id']} | {case['caption']}",font=font,fill='black')
                inputs=output/'inputs'/case['id'];pred=output/f'seed-{seed}'/case['id']
                paths=[inputs/'scene.png',inputs/'visible_mask.png',inputs/'target.png']
                paths += [pred/f'{method}-cfg{cfg:g}.png' for cfg in [1.,7.5] for method in ['sd2','synergy']]
                for col,path in enumerate(paths):
                    with Image.open(path) as im:
                        sheet.paste(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS),(col*256,y+34))
            sheet.save(output/f'comparison-seed-{seed}-page-{group+1}.png')


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError('Use a new output directory to preserve earlier results')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True)
    cases=prepare_cases(args.data,args.output)
    provenance=json.loads((args.base.parent.parent.parent/'sd2-inpainting-provenance.json').read_text())
    checked={}
    for name,record in provenance['files'].items():
        if record['lfs_sha256']:
            actual=sha256(args.base/name)
            if actual!=record['lfs_sha256']:raise ValueError(f'Base-model hash mismatch: {name}')
            checked[name]=actual
    frozen=check_frozen_weights(args.base,args.weights)
    manifest={'status':'running','seeds':args.seeds,'resolution':512,'steps':25,
        'text_cfg':[1.,7.5],'synergy_image_cfg':1.,'target_used_for_generation':False,
        'mask_decoder_used':False,'pixel_compositing':False,'stage':'one global RGB pass',
        'base_checkpoint':provenance,'base_weights_sha256_verified':checked,
        'synergy_weights':str(args.weights),'synergy_checkpoint_verification':json.loads(
            (ROOT/'diagnostics/synergy_checkpoint_verification.json').read_text()),
        'frozen_component_comparison':frozen,'comparison_script_sha256':sha256(Path(__file__)),
        'cases':[{k:v for k,v in c.items() if not torch.is_tensor(v)} for c in cases],
        'notes':['Released Synergy EMA versus pretrained 9-channel SD2 inpainting.',
            'SD2 mirror is community-maintained; the original Stability AI endpoint is deprecated.',
            'Same actual initial noise, caption, image, visible mask, DDIM grid and text CFG.',
            'Synergy also receives scene RGB and uses white outside visible RGB; SD2 uses its native zero-masked normalized RGB.',
            'Dataset examples may overlap released-model training; this is not a held-out benchmark.',
            'Selection of six additional examples uses hidden-area annotations; framing uses visible mask only.'],
        'records':[]}
    write_json(args.output/'run.json',manifest)
    model=ReleasedModel(ROOT/'diagnostics/vendor/synergy_sd2_config',args.weights,'cuda:0')
    # RGB-only comparison: do not invoke the separately trained mask decoder.
    del model.mask_vae
    base=StableDiffusionInpaintPipeline.from_pretrained(str(args.base),torch_dtype=torch.float32,
        vae=model.vae,text_encoder=model.text_encoder,tokenizer=model.tokenizer,
        safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True)
    base.to('cuda:0');base.set_progress_bar_config(disable=True)
    base.scheduler=DDIMScheduler.from_config(base.scheduler.config,steps_offset=39,timestep_spacing='leading')
    base.scheduler.set_timesteps(25,device='cuda:0')
    assert base.unet.config.in_channels==9 and model.unet.config.in_channels==13
    assert torch.equal(base.scheduler.alphas_cumprod,model.scheduler.alphas_cumprod)
    manifest['timesteps']=base.scheduler.timesteps.cpu().tolist()
    manifest['load_audit']=model.load_audit
    import diffusers,transformers
    manifest['versions']={'torch':torch.__version__,'diffusers':diffusers.__version__,'transformers':transformers.__version__}
    for seed in args.seeds:
        for case in cases:
            directory=args.output/f'seed-{seed}'/case['id'];directory.mkdir(parents=True)
            scene=case['image'][None].cuda();visible=case['visible'][None].cuda()
            noise=torch.randn(1,4,64,64,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed))
            save_file({'initial_noise':noise.cpu()},directory/'initial_noise.safetensors')
            def encode(image,subseed):
                g=torch.Generator(device='cuda:0').manual_seed(subseed)
                return model.vae.encode(image).latent_dist.sample(generator=g)*model.vae.config.scaling_factor
            scene_z=encode(scene*2-1,seed+100000)
            white_z=encode((scene*visible+1-visible)*2-1,seed+200000)
            base_z=encode((scene*2-1)*visible,seed+200000)
            mask=F.interpolate(1-visible,size=(64,64),mode='nearest')
            condition=torch.cat([mask,white_z,scene_z],dim=1)
            caption=[case['caption']];text=model.encode_text(caption)
            for cfg in [1.,7.5]:
                start=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=base(prompt=caption,image=scene,mask_image=1-visible,
                        masked_image_latents=base_z,latents=noise.clone(),strength=1.,
                        num_inference_steps=25,guidance_scale=cfg,eta=0.,output_type='latent').images
                base_rgb=(model.vae.decode(result.float()/model.vae.config.scaling_factor).sample/2+.5).clamp(0,1)
                save_tensor(base_rgb[0],directory/f'sd2-cfg{cfg:g}.png')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model.inference_ddim(noise.clone(),condition,text,num_inference_steps=25,cfg_image=1.,cfg_text=cfg)
                assert torch.equal(base.scheduler.timesteps,model.scheduler.timesteps)
                synergy_rgb=(model.vae.decode(result.float()/model.vae.config.scaling_factor).sample/2+.5).clamp(0,1)
                save_tensor(synergy_rgb[0],directory/f'synergy-cfg{cfg:g}.png')
                record={'case':case['id'],'seed':seed,'text_cfg':cfg,'seconds':time.monotonic()-start,
                    'noise_sha256':sha256(directory/'initial_noise.safetensors'),
                    'sd2_visible_rgb_mae':float((base_rgb-scene).abs()[visible.expand_as(scene).bool()].mean()),
                    'synergy_visible_rgb_mae':float((synergy_rgb-scene).abs()[visible.expand_as(scene).bool()].mean()),
                    'sd2_sha256':sha256(directory/f'sd2-cfg{cfg:g}.png'),
                    'synergy_sha256':sha256(directory/f'synergy-cfg{cfg:g}.png')}
                manifest['records'].append(record)
                write_json(args.output/'run.json',manifest)
                print(json.dumps(record),flush=True)
    render(args.output,cases,args.seeds)
    manifest['status']='complete'
    write_json(args.output/'run.json',manifest)
    print(f'Completed {len(manifest["records"])*2} generated images: {args.output}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    artifact=ROOT/'all_data/outputs/synergy_amodal'
    p.add_argument('--output',type=Path,default=artifact/'outputs/qualitative/pretrained-inpainting-comparison')
    p.add_argument('--data',type=Path,default=ROOT/'all_data/datasets/SynergyAmodal16K')
    p.add_argument('--weights',type=Path,default=Path('/data/minht/TheGrid/AmodalGen/all_data/prior_models/synergy_amodal'))
    p.add_argument('--base',type=Path,default=artifact/'cache/huggingface/models--sd2-community--stable-diffusion-2-inpainting/snapshots'/REVISION)
    p.add_argument('--seeds',type=int,nargs='+',default=[45,46])
    run(p.parse_args())


if __name__=='__main__':main()
