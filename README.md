# Pansharpening by convolutional neural networks in the full resolution framework


**Main Paper**: [Pansharpening by convolutional neural networks in the full resolution framework](https://ieeexplore.ieee.org/document/9745494) ([ArXiv](https://arxiv.org/abs/2111.08334)) is 
a deep learning method for Pansharpening based on unsupervised and full-resolution framework training.

## Cite Us

- Z-PNN
```
@article{Ciotola2022,  
         author={Ciotola, Matteo and Vitale, Sergio and Mazza, Antonio and Poggi, Giovanni and Scarpa, Giuseppe},  
         journal={IEEE Transactions on Geoscience and Remote Sensing},   
         title={Pansharpening by convolutional neural networks in the full resolution framework},   
         year={2022},  
         volume={},  
         number={},  
         pages={1-1},  
         doi={10.1109/TGRS.2022.3163887}
}

```

- Fast Z-PNN
```
@article{Ciotola2023,
         author = {Ciotola, Matteo and Scarpa, Giuseppe},
         title = {Fast Full-Resolution Target-Adaptive CNN-Based Pansharpening Framework},
         journal = {Remote Sensing},
         volume = {15},
         year = {2023},
         number = {2},
         article-number = {319},
         url = {https://www.mdpi.com/2072-4292/15/2/319},
         issn = {2072-4292},
         doi = {10.3390/rs15020319}
}

```

- Metrics
```
@article{Scarpa2022,
         author = {Scarpa, Giuseppe and Ciotola, Matteo},
         title = {Full-Resolution Quality Assessment for Pansharpening},
         journal = {Remote Sensing},
         volume = {14},
         year = {2022},
         number = {8},
         article-number = {1808},
         url = {https://www.mdpi.com/2072-4292/14/8/1808},
         issn = {2072-4292},
         doi = {10.3390/rs14081808}
}

```

## Authors
 - Matteo Ciotola (matteo.ciotola@unina.it);
 - Giuseppe Scarpa  (giscarpa@unina.it).
 
 
## License
Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/fast-z-pnn/LICENSE.txt)
(included in this package) 

## Prerequisites
All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

- Python 3.9 
- PyTorch 1.8.1 or 1.10.0
-  Cuda 10.1 or 11.3 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

- Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads) 
- Create a folder in which save the algorithm
- Download the algorithm and unzip it into the folder or, alternatively, from CLI:

```
git clone https://github.com/matciotola/fast-z-pnn
```

- Create the virtual environment with the `z_pnn_environment.yml`

```
conda env create -n z_pnn_env -f z_pnn_environment.yml
```

- Activate the Conda Environment

```
conda activate z_pnn_env
```

- Test it 

```
python main.py -i example/WV3_example.mat -o ./Output_Example -s WV3 -m Fast-Z-PNN --coregistration --show_results 
```


## Usage

### Before to start
To test this algorithm it is needed to create a `.mat` file. It must contain:
- `I_MS_LR`: Original Multi-Spectral Stack in channel-last configuration (Dimensions: H x W x B);
- `I_PAN`: Original Panchromatic band, without the third dimension (Dimensions: H x W).

It is possible to convert the GeoTIff images into the required format with the scripts provided in [`tiff_mat_conversion.py`](https://github.com/matciotola/Z-PNN/blob/master/tiff_mat_conversion.py): 

```
python tiff_mat_conversion.py -m Tiff2Mat -ms /path/to/ms.tif -pan /path/to/ms.tif  -o path/to/file.mat
```

Please refer to `--help` for more details. 

### Testing
The easiest command to use the algorithm on full resolution data:

```
python main.py -i path/to/file.mat -s sensor_name -m method
```
Several options are possible. Please refer to the parser help for more details:

```
python main.py -h
```
