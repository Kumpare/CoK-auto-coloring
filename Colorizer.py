import numpy as np
import cv2 as cv
import os
from dataclasses import dataclass
from collections.abc import Generator

@dataclass
class ColorSpreader:

    hue_min: float
    hue_max: float
    light_min: float
    light_max: float
    saturation_min: float
    saturation_max: float

    @property
    def _mins(self) -> np.ndarray:
        return np.array([self.hue_min, self.light_min, self.saturation_min])

    @property
    def _maxs(self) -> np.ndarray:
        return np.array([self.hue_max, self.light_max, self.saturation_max])

    def __post_init__(self):
        assert self.hue_min <= self.hue_max
        assert self.light_min <= self.light_max
        assert self.saturation_min <= self.saturation_max

        assert self.hue_min >= 0
        assert self.hue_max <= 360

        assert self.light_min >= 0
        assert self.light_max <= 100
        assert self.saturation_min >= 0
        assert self.saturation_max <= 100


        self.hue_min = int(self.hue_min)
        self.hue_max = int(self.hue_max)
        self.light_min = self.light_min/100
        self.light_max = self.light_max/100
        self.saturation_min = self.saturation_min/100
        self.saturation_max = self.saturation_max/100

    def __call__(self, n: int) -> np.ndarray:
        do_hues = int(self.hue_min != self.hue_max and n > 1)
        do_lights = int(self.light_min != self.light_max and n > 1)
        do_sats = int(self.saturation_min != self.saturation_max and n > 1)
        n_dims = do_hues + do_lights + do_sats
        n_sides = int(n**(1/n_dims)) if n_dims > 0 else 1
        n_outputs = n_sides**n_dims

        r = n - n_outputs if n_dims > 0 else 0

        x = np.zeros((n_outputs + r, 3), dtype='float32')

        hues = np.linspace(self.hue_min, self.hue_max, n_sides) if do_hues == 1 and n_sides > 1 else np.array([(self.hue_min + self.hue_max)/2])
        lights = np.linspace(self.light_min, self.light_max, n_sides) if do_lights == 1 and n_sides > 1 else np.array([(self.light_min + self.light_max)/2])
        sats = np.linspace(self.saturation_min, self.saturation_max, n_sides) if do_sats == 1 and n_sides > 1 else np.array([(self.saturation_min + self.saturation_max)/2])

        x[:n_outputs, :] = np.array(np.meshgrid(hues, lights, sats)).T.reshape(-1, 3)
        if r > 0:
            if int(r**(1/n_dims)) > 1:
                hue_step = (self.hue_max - self.hue_min)/4
                light_step = (self.light_max - self.light_min)/4
                sat_step = (self.saturation_max - self.saturation_min)/4
                other = ColorSpreader(hue_min=self.hue_min + hue_step, hue_max=self.hue_max - hue_step,
                                      light_min=(self.light_min + light_step)*100, light_max=(self.light_max - light_step)*100,
                                      saturation_min=(self.saturation_min + sat_step)*100, saturation_max=(self.saturation_max - sat_step)*100)
                x[n_outputs:, :] = other(r)
            elif r == 1:
                x[-1, :] = (self._mins + self._maxs)/2
            else: # 1 < r < 8
                x[n_outputs:, :] = np.linspace(self._mins, self._maxs, r + 2)[1:-1]

        return x

class Colorizer:

    ACCEPTED_FILE_FORMATS = ['.png', '.jpg', '.jpeg']

    def __init__(self, random_color: ColorSpreader, n_generations: int=20, gradient_thickness: int | None = 31, grad_effect: float = 0.1,
                 line_art_effect: float = 0.2):
        self.random_color = random_color
        self.n = n_generations
        self.gradient_thickness = gradient_thickness
        self.grad_effect = grad_effect
        self.line_art_effect = line_art_effect
        assert self.gradient_thickness is None or (self.gradient_thickness % 2 == 1 and 3 <= self.gradient_thickness <= 31)


    def __call__(self, src_dir_or_path: str | os.PathLike, out_dir: str | os.PathLike):
        for j, (img, img_name) in enumerate(self._generate_images(src_dir_or_path)):
            img_name = img_name.split('.')[0]
            result_dir = f'{out_dir}/{img_name}'
            self._make_out_dir(result_dir)

            img_f32 = img.astype('float32') / 255
            grad_magn = self._get_gradients(img_f32[:,:,:-1])
            img_hsv = cv.cvtColor(img_f32, cv.COLOR_BGR2HLS)
            line_art_mask = self._get_line_art_mask(img)
            result = img.copy()
            colors = self.random_color(self.n)
            for i, col in enumerate(colors):
                grad_color = self._get_grad_color(col)
                result[:,:,:-1] = self._colorize(img_hsv.copy(), line_art_mask, grad_magn, col, grad_color)
                cv.imwrite(f'{result_dir}/{img_name}_{i+1}.png', result)

    def _make_out_dir(self, out_dir: str | os.PathLike):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    def _generate_images(self, src_dir: str | os.PathLike) -> Generator[np.ndarray, str]:

        if os.path.isfile(src_dir):
            img = self._read_image(src_dir)
            if img is not None:
                yield img, src_dir.split('/')[-1]

        elif os.path.isdir(src_dir):
            for file_name in os.listdir(src_dir):
                for ff in self.ACCEPTED_FILE_FORMATS:
                    if file_name.endswith(ff):
                        break
                else:
                    continue

                fp = f'{src_dir}/{file_name}'
                img = self._read_image(fp)
                if img is not None:
                    yield img, file_name

        else:
            raise ValueError(f"{src_dir} is not a valid path to a valid image or directory")

    def _read_image(self, fp: str | os.PathLike) -> np.ndarray | None:
        img = cv.imread(fp, cv.IMREAD_UNCHANGED)
        if img.shape[-1] == 4:
            return img

        print(f"Image at {fp} is not a 4-channel image and will be skipped")
        return None

    def _get_gradients(self, x: np.ndarray) -> np.ndarray:
        if self.gradient_thickness is None:
            return np.zeros_like(x)

        gaussian_size = self.gradient_thickness
        x_grayscale = x.mean(axis=-1, keepdims=True)
        gaussian = cv.GaussianBlur(x_grayscale, [gaussian_size, gaussian_size], sigmaX=1, sigmaY=1)

        large_dx = cv.getDerivKernels(1, 0, gaussian_size)
        large_dx = np.outer(large_dx[0], large_dx[1])
        large_dy = cv.getDerivKernels(0, 1, gaussian_size)
        large_dy = np.outer(large_dy[0], large_dy[1])
        large_scaling = np.abs(large_dx).sum()/2

        dx = cv.filter2D(gaussian, -1, large_dx)/large_scaling
        dy = cv.filter2D(gaussian, -1, large_dy)/large_scaling

        magn = np.clip(np.sqrt(dx**2 + dy**2), 0, 1)

        return np.stack([magn for _ in range(3)], axis=-1)

    def _colorize(self, input_img: np.ndarray, outline_mask: np.ndarray, grad: np.ndarray,
                 main_color: np.ndarray, grad_color: np.ndarray) -> np.ndarray:

        img = (1 - grad) * main_color + grad * grad_color
        # img = img*small_grad + (1 - small_grad)*img_colored
        img = outline_mask * input_img + (1 - outline_mask) * img
        img[:,:,1] = (1 - outline_mask[:,:,0])*img[:,:,1] + outline_mask[:,:,0]*main_color[1]*self.line_art_effect
        img = cv.cvtColor(img.astype('float32'), cv.COLOR_HLS2BGR)
        img *= 255
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def _get_line_art_mask(self, img: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(img[:,:,:-1], cv.COLOR_BGR2GRAY)
        return np.expand_dims(np.clip(1 - (gray.astype('float32')-75)/(255-75), 0, 1), -1)

    def _get_grad_color(self, color: np.ndarray) -> np.ndarray:
        grad_color = color.copy()
        grad_color[1] = np.clip((1 + self.grad_effect)*grad_color[1], 0, 1)
        return grad_color

if __name__ == "__main__":

    rand_color = ColorSpreader(0, 130, 0.2, 0.5, 0.2, 0.7)
    colorizer = Colorizer(rand_color, n_generations=50, grad_effect=-0.25)

    img_path = f'originals/wood_planks.png'
    out_dir = 'D:/Test'

    colorizer(img_path, out_dir)