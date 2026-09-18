This is a short python script I wrote to automatically colorize original assets of Canvas of Kings. Technically works on any .png image with RGBA-channels, but results may vary.

Intended use:

  1. Acquire a .png image with RGBA-channels or make one yourself by using GIMP, for example.
  2. Modify config.yaml to your liking, instructions in the file. Useful site for picking colors in the HSL space: https://colorizer.org/, 
  3. Run main.py 

Currently tested on python version 3.14 and requires the following packages:
  - numpy
  - opencv-python
  - pyyaml

To-improve:
  - Some optimization
  - progress bar when images are being generated
  - ColorSpreader currently spreads the colors in a cube-shaped space. It would be more efficient to utilize HSL-space's double-cone-like shape.
  - Handle gradients better to allow further reaching gradients and different color values for gradients only
