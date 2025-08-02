<div align="center">
  <h1>GPIC Converter</h1>

  [![License](https://img.shields.io/github/license/bromander/GPIC?style=for-the-badge)](https://github.com/bromander/GPIC/blob/master/LICENSE)
</div>

## Getting Started
#### Releases Page

You can download the latest and most up-to-date .exe version from the [Releases page](https://github.com/bromander/GPIC/releases/tag/release)

#### Cloning the Repository

To get the source code, run:
```
git clone https://github.com/bromander/GPIC.git
```

## How It Works
Converting a .png (for example) to .gpic can be broken down into seven steps:
1. Extract the pixel matrix from the original image
2. Convert RGB values to greyscale using the formula: `0.299 * R + 0.587 * G + 0.114 * B`
3. Apply the [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) (DCT) to each block of the greyscale data
4. Divide the matrix into blocks (by default 8×8 pixels)
5. Apply [quantization](https://en.wikipedia.org/wiki/Quantization_(image_processing)) to each block
6. Reconstruct the pixel matrix by applying the inverse [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) (DCT) to each block
7. Apply [quantization](https://en.wikipedia.org/wiki/Quantization_(image_processing)) again to each reconstructed block

#### GPIC Library
Active development is underway of a library to work with the GPIC file format
