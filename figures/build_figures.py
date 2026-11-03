"""Build editable, code-faithful SynergyAmodal figures in the PixArt figure style.

Illustration tiles are copied from the existing reference diagram. Pillow reads
the visible-mask pixels to build its exact inverted 64-square vector mask.
No model execution is performed.
"""
import copy
import base64
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

DEST = Path(__file__).resolve().parent
WORKSPACE = DEST.parents[1]
REFERENCE = WORKSPACE / 'pixart-alpha-amodal/figures/training.drawio'
GREEN, BLUE, DARK = '#E2F0D9', '#DEEBF7', '#AFCBE5'
GRAY, INK, ORANGE, TEAL = '#F1F1F1', '#171717', '#F07835', '#00A6CE'


class Page:
    def __init__(self, name):
        self.name = name
        self.model = ET.Element('mxGraphModel', dx='1920', dy='1440', grid='1', gridSize='10',
            guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1',
            pageWidth='1920', pageHeight='1440', math='0', shadow='0', background='#FFFFFF')
        self.root = ET.SubElement(self.model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.number = 0

    def node(self, x, y, w, h, text='', style=''):
        self.number += 1
        cell = ET.SubElement(self.root, 'mxCell', id=f'cell-{self.number}', parent='1', vertex='1',
            value=text, style='html=0;whiteSpace=wrap;fontFamily=Arial;fontSize=22;fontColor='+INK+
            ';align=center;verticalAlign=middle;spacing=3;'+style)
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as': 'geometry'})
        return cell

    def text(self, x, y, w, h, text, size=22, bold=False, color=INK, align='center', font='Arial'):
        return self.node(x, y, w, h, text, f'text;fillColor=none;strokeColor=none;fontSize={size};'
            f'fontStyle={int(bold)};fontColor={color};align={align};fontFamily={font};')

    def box(self, x, y, w, h, text='', fill=GRAY, size=22, stroke=INK):
        return self.node(x, y, w, h, text, f'rounded=1;arcSize=12;fillColor={fill};'
            f'strokeColor={stroke};strokeWidth=1.7;fontSize={size};')

    def image(self, data, x, y, w, h=None):
        self.node(x, y, w, h or w, style='shape=image;imageAspect=0;aspect=fixed;strokeColor=none;image='+data+';')

    def condition_mask(self, x, y):
        self.node(x-1, y-1, 102, 102, style='fillColor=#FFFFFF;strokeColor=#B8B8B8;strokeWidth=1;')
        self.image(ASSETS['inverted_mask'], x, y, 100)

    def snow(self, x, y, size=30):
        self.text(x, y, size, size, '❄', size, color=TEAL, font='DejaVu Sans')

    def flame(self, x, y, size=27):
        self.image(ASSETS['fire'], x, y, size, size*46/36)

    def arrow(self, points, color=INK, dashed=False, end='block'):
        self.number += 1
        cell = ET.SubElement(self.root, 'mxCell', id=f'edge-{self.number}', parent='1', edge='1',
            style=f'edgeStyle=none;html=0;rounded=0;strokeColor={color};strokeWidth=1.8;'
                  f'endArrow={end};endFill=1;dashed={int(dashed)};dashPattern=8 5;')
        geometry = ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
        for key, (x, y) in [('sourcePoint', points[0]), ('targetPoint', points[-1])]:
            ET.SubElement(geometry, 'mxPoint', x=str(x), y=str(y), attrib={'as': key})
        if len(points) > 2:
            intermediate = ET.SubElement(geometry, 'Array', attrib={'as': 'points'})
            for x, y in points[1:-1]:
                ET.SubElement(intermediate, 'mxPoint', x=str(x), y=str(y))

    def vae(self, x, y, label='E', trainable=False, decoder=False):
        self.node(x+7.5, y-7.5, 110, 125, '', 'shape=trapezoid;rotation='+('270' if decoder else '90')+
            ';fillColor='+GREEN+';strokeColor='+INK+';strokeWidth=1.7;')
        self.text(x+12, y+33, 90, 50, label, 34, font='Times New Roman')
        if trainable:
            self.flame(x+87, y+4, 24)
        else:
            self.snow(x+84, y+4, 28)

    def circle(self, x, y, text='C', size=50):
        self.node(x, y, size, size, text, f'ellipse;fillColor={GRAY};strokeColor={INK};strokeWidth=1.7;fontSize=29;')

    def header(self, subtitle, training=False):
        self.text(25, 10, 265, 65, self.name, 48, align='left', font='Times New Roman')
        self.text(300, 23, 1120, 45, subtitle, 29, bold=True, align='left')
        self.snow(1480, 26, 32)
        self.text(1520, 20, 170, 42, 'Frozen' if training else 'Fixed weights', 22, align='left')
        if training:
            self.flame(1700, 23)
            self.text(1740, 20, 150, 42, 'Trainable', 22, align='left')
        else:
            self.text(1700, 20, 185, 42, 'EMA models', 22, align='right')


def get_assets():
    root = ET.parse(REFERENCE).getroot()
    cells = {c.get('id'): c for c in root.iter('mxCell')}
    def data(cell):
        return cell.get('style').split('image=', 1)[1].split(';', 1)[0]
    assets = {'fire': data(cells['cell-5']), 'observed': data(cells['cell-8'])}
    # Select tiles by their location in the reference rather than transient IDs.
    for c in root.iter('mxCell'):
        g = c.find('mxGeometry')
        if g is None or 'image=data:image/png,' not in c.get('style', ''):
            continue
        xy = (float(g.get('x', 0)), float(g.get('y', 0)))
        if xy == (35, 515): assets['target'] = data(c)
        if xy == (35, 935): assets['mask'] = data(c)
        if xy == (460, 400): assets['noise'] = data(c)
        if xy == (1740, 738): assets['prediction'] = data(c)
    assert len(assets) == 6, assets.keys()
    # Match F.interpolate(..., size=(64, 64))'s default nearest sampling:
    # output pixel (x, y) reads floor(x*W/64), floor(y*H/64).
    # Vectorize the actual complement, preserving the source pixel values.
    with Image.open(io.BytesIO(base64.b64decode(assets['mask'].split(',', 1)[1]))) as mask:
        mask = mask.convert('L')
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" shape-rendering="crispEdges">',
               '<rect width="64" height="64" fill="white"/>']
        for y in range(64):
            row = [255-mask.getpixel((x*mask.width//64, y*mask.height//64)) for x in range(64)]
            x = 0
            while x < 64:
                end = x+1
                while end < 64 and row[end] == row[x]:
                    end += 1
                if row[x] != 255:
                    gray = f'{row[x]:02x}' * 3
                    svg.append(f'<rect x="{x}" y="{y}" width="{end-x}" height="1" fill="#{gray}"/>')
                x = end
        svg.append('</svg>')
    assets['inverted_mask'] = 'data:image/svg+xml,' + base64.b64encode(''.join(svg).encode()).decode()
    return assets


def unet(p, x, y, w, h, trainable=True):
    p.box(x, y, w, h, fill=GRAY)
    p.text(x+18, y+12, w-75, 43, 'Conditional SD2 inpainting U-Net', 27, True, align='left')
    if trainable: p.flame(x+w-48, y+15)
    else: p.snow(x+w-48, y+17)
    # U silhouette is schematic; labels represent block groups, not layer counts.
    xs = [x+35, x+155, x+275, x+395, x+515]
    ys = [y+90, y+120, y+155, y+120, y+90]
    for a, b, label in zip(xs, ys, ['Down', 'Down', 'Mid', 'Up', 'Up']):
        p.box(a, b, 80, 90, label, DARK, 20)
    for i in range(4):
        p.arrow([(xs[i]+80, ys[i]+45), (xs[i]+100, ys[i]+45), (xs[i]+100, ys[i+1]+45), (xs[i+1], ys[i+1]+45)])
    for left, right, level in [(0, 4, y+65), (1, 3, y+100)]:
        p.arrow([(xs[left]+40, ys[left]), (xs[left]+40, level), (xs[right]+40, level), (xs[right]+40, ys[right])], dashed=True)
    p.text(x+20, y+h-49, w-40, 38, '13 input channels → 4-channel noise prediction', 21)


def training():
    p = Page('Training')
    p.header('SynergyAmodal · RGB completion + amodal mask', True)
    p.text(30, 92, 700, 40, 'A  ·  Train the completion diffusion model', 25, True, align='left')
    p.box(45, 158, 210, 75, 'Caption\n50% → empty text', '#FFFFFF', 21)
    p.arrow([(255, 195), (310, 195)])
    p.box(310, 153, 250, 85, 'CLIP text encoder', BLUE, 23)
    p.snow(522, 157, 28)
    p.arrow([(560, 195), (1170, 195), (1170, 450)])
    p.text(840, 205, 320, 40, 'Text cross-attention', 21)
    p.box(1435, 115, 435, 153, fill='#FAFAFA', stroke='#B8B8B8')
    p.text(1450, 127, 405, 33, 'Independent condition dropout', 22, True)
    p.text(1450, 170, 405, 83, '50%: empty caption\n10%: zero all 9 image-condition channels\nTarget latent and diffusion noise retained', 20)

    # Complete target -> noised latent (supervision only).
    p.text(15, 280, 180, 35, 'Complete RGB', 22)
    p.image(ASSETS['target'], 45, 325, 110)
    p.arrow([(155, 380), (220, 380)])
    p.vae(220, 325)
    p.arrow([(345, 380), (390, 380)])
    p.box(390, 345, 80, 70, 'z₀', GREEN, 30)
    p.arrow([(470, 380), (510, 380)])
    p.circle(510, 360, '+', 40)
    p.image(ASSETS['noise'], 495, 260, 70)
    p.arrow([(530, 330), (530, 360)])
    p.text(585, 280, 330, 60, 'ε = N(0, I) + 0.1 εoffset\nChannel-wise offset noise', 20)
    p.arrow([(550, 380), (635, 380)])
    p.box(635, 345, 95, 70, 'zₜ', BLUE, 30)
    p.text(590, 420, 185, 35, '4 × 64 × 64', 20)
    p.text(325, 464, 440, 40, 'zₜ = √ᾱₜ z₀ + √(1 − ᾱₜ) ε', 23, font='Times New Roman')
    p.arrow([(730, 380), (860, 380), (860, 570)])

    # Clean condition branches; spatial mask is resized, not VAE encoded.
    p.text(15, 514, 180, 35, 'Occluded RGB I', 22)
    p.image(ASSETS['observed'], 45, 555, 110)
    p.arrow([(155, 610), (220, 610)])
    p.vae(220, 555)
    p.arrow([(345, 610), (635, 610)])
    p.box(635, 575, 95, 70, 'zᴵ', GREEN, 29)
    p.text(580, 650, 210, 35, '4 × 64 × 64', 20)
    p.arrow([(730, 610), (835, 610)])
    p.box(35, 735, 315, 95, 'Visible-object RGB Iᵥ\nI ⊙ Mᵥ + (1 − Mᵥ)', '#FFFFFF', 23)
    p.text(35, 833, 315, 40, 'White outside the visible object', 18, color='#686868')
    p.arrow([(350, 782), (420, 782)])
    p.vae(420, 727)
    p.arrow([(545, 782), (635, 782)])
    p.box(635, 747, 95, 70, 'zᵥ', GREEN, 29)
    p.text(580, 820, 210, 35, '4 × 64 × 64', 20)
    p.arrow([(730, 782), (860, 782), (860, 620)])
    p.text(10, 887, 190, 35, 'Visible mask Mᵥ', 21)
    p.image(ASSETS['mask'], 45, 930, 100)
    p.arrow([(145, 980), (235, 980)])
    p.box(235, 945, 190, 70, 'Invert + resize\n64 × 64', GREEN, 21)
    p.arrow([(425, 980), (485, 980)])
    p.text(450, 887, 170, 35, 'Condition mask', 21)
    p.condition_mask(485, 930)
    p.arrow([(585, 980), (635, 980)])
    p.box(635, 945, 95, 70, 'm', GREEN, 29)
    p.text(585, 1018, 200, 35, '1 × 64 × 64', 20)
    p.arrow([(730, 980), (810, 980), (810, 640), (845, 615)])
    p.circle(835, 570)
    p.text(777, 520, 160, 38, 'Concat', 23)
    p.text(990, 860, 380, 35, 'Concat order: [zₜ, m, zᵥ, zᴵ]', 20)
    p.arrow([(885, 595), (950, 595)])
    p.text(883, 546, 67, 35, '13 ch', 18)
    unet(p, 950, 450, 650, 320)
    p.box(1015, 320, 95, 55, 't', '#FFFFFF', 28)
    p.text(920, 277, 290, 36, 'Uniform diffusion timestep', 21)
    p.arrow([(1062, 375), (1062, 450)])
    p.text(960, 790, 630, 70, 'Pretrained 9-channel input + 4 scene channels\nNew input weights start at zero; train the full U-Net', 21)
    p.arrow([(1600, 595), (1705, 595)])
    p.image(ASSETS['prediction'], 1705, 548, 95)
    p.text(1660, 495, 190, 40, 'Predicted ε̂', 23)
    p.text(1665, 658, 180, 35, '4 × 64 × 64', 20)
    p.box(1390, 900, 405, 90, 'Ldiff = E[w(t) · MSE(ε̂, ε)]\nw(t) = min(SNR(t), 5) / SNR(t)', '#FFF7ED', 23, ORANGE)
    p.arrow([(1752, 643), (1815, 643), (1815, 875), (1752, 875), (1752, 900)], ORANGE)
    p.arrow([(565, 295), (575, 295), (575, 275), (1875, 275), (1875, 945), (1795, 945)], ORANGE, True)
    p.text(1630, 300, 230, 36, 'Same sampled ε', 19, color=ORANGE)
    p.arrow([(1390, 945), (980, 945), (980, 770)], ORANGE, True)
    p.text(995, 961, 380, 58, 'Backpropagate diffusion loss\nUpdate U-Net only', 22, color=ORANGE)
    p.text(210, 1045, 1600, 45, 'All RGB encodings: normalize to [−1, 1] → sample frozen VAE posterior → × 0.18215. Only z₀ is noised.', 21)

    # Independently trained mask decoder; no diffusion U-Net in this path.
    p.box(25, 1110, 1865, 275, fill='#FAFAFA', stroke='#B8B8B8')
    p.text(45, 1120, 1240, 40, 'B  ·  Train the amodal mask decoder separately', 25, True, align='left')
    p.image(ASSETS['target'], 60, 1185, 100)
    p.text(35, 1300, 150, 55, 'Complete RGB\n512 × 512', 19)
    p.arrow([(160, 1235), (235, 1235)])
    p.vae(235, 1180)
    p.arrow([(360, 1235), (410, 1235)])
    p.box(410, 1195, 225, 80, 'Sample latent\nScale → unscale', GREEN, 22)
    p.arrow([(635, 1235), (700, 1235)])
    p.vae(700, 1180, 'Dₘ', True, True)
    p.text(660, 1300, 220, 60, 'Train VAE decoder\n1-channel head, zero init', 19)
    p.arrow([(825, 1235), (900, 1235)])
    p.box(900, 1195, 200, 80, 'Mask logits\n1 × 512 × 512', BLUE, 22)
    p.arrow([(1100, 1235), (1180, 1235)], ORANGE)
    p.box(1180, 1195, 280, 80, 'Lmask = BCE + Dice', '#FFF7ED', 24, ORANGE)
    p.box(1535, 1195, 310, 80, 'Ground-truth amodal mask', '#FFFFFF', 22)
    p.arrow([(1535, 1235), (1460, 1235)], ORANGE)
    p.arrow([(1320, 1275), (1320, 1365), (650, 1365), (650, 1285), (700, 1285)], ORANGE, True)
    p.text(925, 1300, 380, 45, 'Update mask decoder only', 21, color=ORANGE)
    p.text(30, 1395, 1855, 33, 'Separate EMA shadows (β = 0.999) for U-Net and mask decoder · photos reused from the reference for illustration; noise is schematic', 19, color='#686868')
    return p


def inference():
    p = Page('Inference')
    p.header('SynergyAmodal · text-controlled amodal completion')
    p.box(30, 103, 1860, 115, fill='#FAFAFA', stroke='#B8B8B8')
    p.text(45, 113, 285, 90, 'Image + object clicks\nZIM → visible mask Mᵥ', 22)
    p.arrow([(330, 160), (390, 160)])
    p.box(390, 128, 410, 65, 'Optional erosion → resize + pad to 512²', '#FFFFFF', 21)
    p.arrow([(800, 160), (860, 160)])
    p.text(870, 115, 980, 88, 'One completion pass is expanded below.\nThe app runs it globally, then repeats it on an object crop with the same models.', 23)

    p.text(25, 245, 700, 40, 'Clean spatial conditions · reused at every step', 24, True, align='left')
    p.text(20, 305, 180, 35, 'Occluded RGB I', 22)
    p.image(ASSETS['observed'], 55, 350, 110)
    p.arrow([(165, 405), (235, 405)])
    p.vae(235, 350)
    p.arrow([(360, 405), (450, 405)])
    p.box(450, 370, 100, 70, 'zᴵ', GREEN, 29)
    p.box(35, 555, 280, 90, 'Visible RGB Iᵥ\nI ⊙ Mᵥ + (1 − Mᵥ)', '#FFFFFF', 22)
    p.arrow([(315, 600), (350, 600)])
    p.vae(350, 545)
    p.arrow([(475, 600), (510, 600)])
    p.box(510, 565, 90, 70, 'zᵥ', GREEN, 29)
    p.text(15, 720, 190, 35, 'Visible mask Mᵥ', 21)
    p.image(ASSETS['mask'], 55, 765, 100)
    p.arrow([(155, 815), (240, 815)])
    p.box(240, 780, 170, 70, 'Invert + resize\n64 × 64', GREEN, 21)
    p.arrow([(410, 815), (470, 815)])
    p.text(435, 730, 170, 30, 'Condition mask', 21)
    p.condition_mask(470, 765)
    p.arrow([(550, 405), (660, 405), (660, 575)])
    p.arrow([(600, 600), (635, 600)])
    p.arrow([(570, 815), (660, 815), (660, 625)])
    p.circle(635, 575)
    p.text(370, 655, 245, 60, 'cᴵ = [m, zᵥ, zᴵ]\n9 × 64 × 64', 20)
    p.arrow([(685, 600), (775, 600)])
    p.text(185, 890, 530, 65, 'RGB: sample VAE posterior × 0.18215\nEach RGB latent has 4 × 64 × 64 values', 21)

    # Text and timestep inputs above the denoising enclosure.
    p.box(820, 270, 245, 70, 'User text / empty text', '#FFFFFF', 21)
    p.arrow([(1065, 305), (1115, 305)])
    p.box(1115, 270, 280, 70, 'CLIP text encoder', BLUE, 23)
    p.snow(1358, 274, 27)
    p.text(1190, 365, 200, 60, 'Text cross-attention', 20)
    p.box(775, 435, 650, 440, fill=GRAY)
    p.text(795, 450, 575, 40, 'DDIM denoising · 25 steps per pass', 25, True, align='left')
    p.snow(1380, 454, 28)
    p.box(820, 555, 155, 90, '[zₜ, cᴵ]\n13 channels', BLUE, 23)
    p.arrow([(975, 600), (1035, 600)])
    p.box(1035, 550, 330, 100, 'SynergyAmodal U-Net\nEMA checkpoint', DARK, 25)
    p.arrow([(1395, 305), (1440, 305), (1440, 520), (1255, 520), (1255, 550)])
    p.arrow([(775, 600), (820, 600)])
    p.text(790, 497, 215, 35, 'zₜ · 4 × 64 × 64', 20)
    p.arrow([(895, 530), (895, 555)])
    p.box(1035, 700, 330, 70, 'Image / text guidance → ε̂', BLUE, 22)
    p.arrow([(1200, 650), (1200, 700)])
    p.box(820, 700, 155, 70, 'DDIM step\nzₜ → zₜ₋₁', GREEN, 21)
    p.arrow([(1035, 735), (975, 735)])
    p.arrow([(820, 735), (800, 735), (800, 530), (895, 530)], dashed=True)
    p.text(995, 790, 395, 55, 'Repeat over t\nKeep image and text conditions fixed', 20)
    p.image(ASSETS['noise'], 805, 355, 65)
    p.text(880, 360, 265, 55, 'Start zT ∼ N(0, I)\n(global pass)', 20)
    p.arrow([(837, 420), (837, 435)])

    # Both decoders receive the same unscaled completed latent.
    p.arrow([(897, 770), (897, 860), (1435, 860), (1435, 735), (1470, 735)])
    p.box(1470, 695, 160, 80, 'Final z₀\n÷ 0.18215', GREEN, 23)
    p.arrow([(1550, 695), (1550, 675), (1455, 675), (1455, 410), (1490, 410)])
    p.vae(1490, 355, 'Dᵣ', decoder=True)
    p.text(1445, 305, 230, 40, 'Original RGB decoder', 20)
    p.arrow([(1615, 410), (1715, 410)])
    p.box(1715, 360, 150, 100, 'Amodal RGB\n3 × 512²', BLUE, 22)
    p.arrow([(1455, 600), (1490, 600)])
    p.vae(1490, 545, 'Dₘ', decoder=True)
    p.text(1475, 490, 215, 45, 'Mask decoder · EMA', 19)
    p.arrow([(1615, 570), (1715, 570)])
    p.box(1715, 530, 150, 100, 'Mask logits\nSigmoid\nRound', BLUE, 21)
    p.arrow([(1790, 460), (1880, 460), (1880, 815), (1810, 815)])
    p.arrow([(1790, 630), (1790, 775)])
    p.box(1700, 775, 150, 80, 'RGB + alpha\nRGBA', GREEN, 23)
    p.text(1460, 870, 410, 65, 'Decode the same latent twice;\nthe mask is the predicted alpha channel', 20)

    p.box(30, 980, 1860, 140, fill='#FAFAFA', stroke='#B8B8B8')
    p.text(50, 995, 340, 42, 'Separate guidance controls', 23, True, align='left')
    p.text(410, 993, 1430, 45, 'ε̂ = ε∅ + simage (εimage − ε∅) + stext (εimage,text − εimage)', 28, font='Times New Roman')
    p.text(50, 1053, 1780, 45, 'Three branches: image + text  /  image + empty text  /  zero image condition + empty text. If both scales = 1, use one conditional prediction.', 21)

    p.text(30, 1145, 1500, 40, 'Coarse-to-fine app flow · reuse the completion pass and both decoders', 25, True, align='left')
    for x, w, text in [
        (35, 275, '1  Global pass\nResize + pad\nStart from Gaussian noise'),
        (365, 365, '2  Crop around predicted mask\nRestore input-image coordinates\nCrop original RGB + visible mask'),
        (785, 390, '3  Local pass at 512 × 512\nRe-encode crop conditions\nUse the same caption + models'),
        (1230, 290, '4  Restore crop\nPut RGB + alpha back\nThreshold alpha at 0.5'),
        (1575, 300, 'Final RGBA\nCompleted object\nTransparent background')]:
        p.box(x, 1210, w, 105, text, GREEN if x == 1575 else BLUE, 21)
    for a, b in [(310, 365), (730, 785), (1175, 1230), (1520, 1575)]:
        p.arrow([(a, 1262), (b, 1262)])
    p.text(30, 1340, 1840, 64, 'Local strength < 1: noise an encoded coarse RGB crop and run the remaining DDIM steps. At strength = 1 (app default), start from fresh noise.\nInference uses EMA U-Net + EMA mask decoder. The original RGB VAE and text encoder stay fixed. No weights are updated.', 21)
    return p


def write_drawio(path, pages):
    root = ET.Element('mxfile', host='app.diagrams.net', agent='SynergyAmodal figure builder')
    for p in pages:
        diagram = ET.SubElement(root, 'diagram', id=p.name.lower(), name=p.name)
        diagram.append(copy.deepcopy(p.model))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    ASSETS = get_assets()
    pages = [training(), inference()]
    for p in pages:
        write_drawio(DEST / (p.name.lower()+'.drawio'), [p])
    write_drawio(DEST / 'synergy_amodal.drawio', pages)
    sources = ['trainer_deocclusion_pseudo.py', 'trainer_deocclusion_pseudo_decoder.py',
               'app.py', 'utils.py', 'configs/inpainter_ours_pseudo.yaml', 'configs/rgba_ours_pseudo.yaml']
    records = {name: hashlib.sha256((DEST.parent / name).read_bytes()).hexdigest() for name in sources}
    records['../../pixart-alpha-amodal/figures/training.drawio'] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    (DEST / 'provenance.json').write_text(json.dumps({'source_sha256': records,
        'illustrations': 'Existing PixArt reference sample 7058153; illustrative only, not a SynergyAmodal result.',
        'condition_mask': 'Exact complement of the embedded visible mask, nearest-resized to 64 by 64 with floor-coordinate sampling and embedded as SVG pixel runs.',
        'noise_tiles': 'Existing schematic reference tiles; not generated model predictions.'}, indent=2)+'\n')
    print('Built training.drawio, inference.drawio, synergy_amodal.drawio')
