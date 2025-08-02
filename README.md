<div align="center">
  <h1>GPIC Converter</h1>
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
3. Apply the [Discrete Cosine Transform](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BA%D1%80%D0%B5%D1%82%D0%BD%D0%BE%D0%B5_%D0%BA%D0%BE%D1%81%D0%B8%D0%BD%D1%83%D1%81%D0%BD%D0%BE%D0%B5_%D0%BF%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5) (DCT) to each block of the greyscale data
4. Divide the matrix into blocks (by default 8×8 pixels)
5. Apply [quantization](https://ru.wikipedia.org/wiki/%D0%9A%D0%B2%D0%B0%D0%BD%D1%82%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_(%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B0_%D1%81%D0%B8%D0%B3%D0%BD%D0%B0%D0%BB%D0%BE%D0%B2)) to each block
6. Reconstruct the pixel matrix by applying the inverse [Discrete Cosine Transform](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BA%D1%80%D0%B5%D1%82%D0%BD%D0%BE%D0%B5_%D0%BA%D0%BE%D1%81%D0%B8%D0%BD%D1%83%D1%81%D0%BD%D0%BE%D0%B5_%D0%BF%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5) (DCT) to each block
7. Apply [quantization](https://ru.wikipedia.org/wiki/%D0%9A%D0%B2%D0%B0%D0%BD%D1%82%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_(%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B0_%D1%81%D0%B8%D0%B3%D0%BD%D0%B0%D0%BB%D0%BE%D0%B2)) again to each reconstructed block

#### GPIC Library
Active development is underway of a library to work with the GPIC file format
