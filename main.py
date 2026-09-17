import yaml
from Colorizer import Colorizer, ColorSpreader
import os

cfg_path = './config.yaml'


with open(cfg_path) as stream:
    cfg = yaml.safe_load(stream)

out_root = cfg['out_dir']

if len(os.listdir(out_root)) > 0:
    a = None
    while a not in ('y', 'n'):
        a = input(
            f"Directory {out_root} is not empty. Some files may be overwritten and this can't be undone. Continue? y/n: \n").lower()

        if a == 'y':
            break
        elif a == 'n':
            print("Quitting.")
            quit()

for src_name, src in cfg['sources'].items():
    out_dir = f'{out_root}/{src_name}'
    color_spreader = ColorSpreader(**src['color'])
    src_dir = src['src_dir']
    gradient_thickness = src['gradient_thickness']
    grad_effect = src['grad_effect']
    n = src['n_generations']
    line_art_effect = src['line_art_effect']

    colorizer = Colorizer(color_spreader, n, gradient_thickness, grad_effect, line_art_effect)
    colorizer(src_dir, out_dir)