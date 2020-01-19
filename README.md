# Spectral Illumination Correction Achieving Relative Color Constancy Under the Spectral Domain (ISSPIT 2018)

Source codes and datasets for the `Spectral Illumination Correction: Achieving Relative Color Constancy Under the Spectral Domain` paper.

## Environment

* Python 2.7
* Tensorflow 1.4.0


## Datasets

This repo contains two unique and challenging datasets for spectral illumination correction and relative color constancy:

<p align="center">
<img src="https://github.com/zyfccc/Spectral-Illumination-Correction-Achieving-Relative-Color-Constancy-Under-the-Spectral-Domain/blob/master/d2e1.png" height="220">
<img src="https://github.com/zyfccc/Spectral-Illumination-Correction-Achieving-Relative-Color-Constancy-Under-the-Spectral-Domain/blob/master/d2e2.png"  height="220">
</p>

106 nonuniform illuminated images of multiple color charts in a lightbox taken by an iPhone6P and under different illumination color. The two color charts in every image are identical.


<p align="center">
<img src="https://github.com/zyfccc/Spectral-Illumination-Correction-Achieving-Relative-Color-Constancy-Under-the-Spectral-Domain/blob/master/d1e1.png" height="220">
<img src="https://github.com/zyfccc/Spectral-Illumination-Correction-Achieving-Relative-Color-Constancy-Under-the-Spectral-Domain/blob/master/d1e2.png"  height="220">
</p>

28 uniform illuminated images of a color chart taken by the same iPhone6P and under different illumination colors.


## Calibration

Run `python camera_calibration.py` to calculate the optimal `a` value to be used.


## Validation

Run `python paper_snic.py` to validate the SNIC algorithm for nonuniform illumination correction.

Run `python paper_fsim.py` to validate the fSIM algorithm for illumination matching.


## Visualization

To visualize the corrected images, run `python visualization.py`.


## Publication

Please cite our paper if you find this work helpful:

https://ieeexplore.ieee.org/document/8642637

https://pureadmin.qub.ac.uk/ws/portalfiles/portal/163783181/sic_converted.pdf

[1] Y. Zhao, C. Elliott, H. Zhou, and K. Rafferty, “Spectral Illumination Correction: Achieving Relative Color Constancy Under the Spectral Domain,” in 2018 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT), 2018, pp. 690–695.

```
@inproceedings{zhao2018spectral,
  title={Spectral Illumination Correction: Achieving Relative Color Constancy Under the Spectral Domain},
  author={Zhao, Yunfeng and Elliott, Chris and Zhou, Huiyu and Rafferty, Karen},
  booktitle={2018 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)},
  pages={690--695},
  year={2018},
  organization={IEEE}
}
```

Thanks a lot!

Yunfeng


## Acknowledgment

This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie-Sklodowska-Curie grant agreement No 720325, FoodSmartphone.
