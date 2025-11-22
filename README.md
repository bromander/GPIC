<div align="center">
  <h1>GPIC Converter<br>
    &
    GPIC file-loader</h1>
  
  [![License](https://img.shields.io/github/license/bromander/GPIC?style=for-the-badge)](https://github.com/bromander/GPIC/blob/master/LICENSE)
</div>

# Content

- [Getting Started](#getting-started)
    - [Releases Page](#releases-page)
    - [Cloning the Repository](#cloning-the-repository)
- [How It Works](#how-it-works)
    - [Action Sequence](#action-sequence)
    - [Structure](#structure)
    - [Verion Control](#verion-control)

## Getting Started
### Releases Page

You can download .exe version from the [Releases page](https://github.com/bromander/GPIC/releases/tag/release)

### Cloning the Repository

To get the source code and latest version run:
```
git clone https://github.com/bromander/GPIC.git
```

## How It Works
### Action Sequence
Converting a .png (for example) to .gpic can be broken down into seven steps:
1. Extract the pixel matrix from the original image
2. Convert RGB values to greyscale using the formula: `0.299 * R + 0.587 * G + 0.114 * B`
3. Apply the [Discrete Cosine Transform method](https://en.wikipedia.org/wiki/Discrete_cosine_transform) (DCT) to each block of the greyscale data
4. Divide the matrix into blocks (by default 8×8 pixels)
5. Apply [quantization method](https://en.wikipedia.org/wiki/Quantization_(image_processing)) to each block

### Structure
the file consists of 5 chunks:
1. CCAP – file signature\
   – \x89 G P C \n
   
3. CDAT – Image Information.\
   – a 12-byte chunk that currently stores the width and height of the image, as well as how many bytes the pixel array occupies.\
   It is indicated by the byte “O”.
   
4. CPIX – An array of pixels.\
   – a chunk that stores an array with data about each pixel.\
   It is indicated by the byte “A”.
   
5. CTXT – Image type.\
   – a 5-byte chunk that stores information about what kind of image it is. The image type consists of 4 characters.\
   It is indicated by the byte “T”.
   
6. CEND – A chunk indicating the end of the file.

### Verion Control
There are 4 version formats:
1. "0" - No compress (V1.5+)
2. "1" - LZMA only (V1.5+)
3. "2" - DCT & Quantization & LZMA 
4. "3" - DCT & Quantization & BROTLI (V1.6+)


<img width="165" height="298" alt="GPIC_logo" src="https://github.com/user-attachments/assets/5e58c174-ab11-4d81-baaa-660e4d318749" /> <img width="165" height="298" alt="GPIC_logo" src="https://github.com/user-attachments/assets/5e58c174-ab11-4d81-baaa-660e4d318749" /> <img width="165" height="298" alt="GPIC_logo" src="https://github.com/user-attachments/assets/5e58c174-ab11-4d81-baaa-660e4d318749" /> <img width="165" height="298" alt="GPIC_logo" src="https://github.com/user-attachments/assets/5e58c174-ab11-4d81-baaa-660e4d318749" />
