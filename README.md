# Spectral-Illumination-Correction-Achieving-Relative-Color-Constancy-Under-the-Spectral-Domain

Source codes and datasets for the `Spectral Illumination Correction: Achieving Relative Color Constancy Under the Spectral Domain` paper.

## Environment

* Python 2.7
* Tensorflow 1.4.0


## Datasets

106 nonuniform illuminated images of multiple color charts in a lightbox taken by an iPhone6P and under different illumination color. The two color charts in every image are identical.

28 uniform illuminated images of a color chart taken by the same iPhone6P and under different illumination colors.


## Calibration

`python camera_calibration.py` to calculate the optimal `a` value to be used.


## Validation

`python paper_snic.py` to validate the SNIC algorithm for nonuniform illumination correction.

`python paper_fsim.py` to validate the fSIM algorithm for illumination matching.


## Publication

Please cite our paper:

[1] Y. Zhao, C. Elliott, H. Zhou, and K. Rafferty, “Spectral Illumination Correction: Achieving Relative Color Constancy Under the Spectral Domain,” in 2018 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT), 2018, pp. 690–695.


Thanks a lot!