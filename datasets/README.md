# Data Preparing

1. Download the OCTA dataset from the [Baidu Netdisk](https://pan.baidu.com/s/1U1wVj07lKPwZAVkBY0s5Mw) (access code: wveo) and decompress it to the root directory.
2. Description of the OCTA dataset:
* The dataset in folder **Data33** was collected at the First Affiliated Hospital of Xi’an Jiaotong University, consisting of 110 normal eyes from 110 subjects. For each eye, both OCTA image (3×3 mm2 optic disc-centered, 320×320 pixels) and fundus image (FOV of 45°, optic disc-centered, 2576×1934 pixels) were captured. The dataset in folder **Data66** was collected at Zhongshan Ophthalmic Center of Sun Yat-sen University, consisting of 908 normal eyes from 476 subjects. For each eye, both OCTA image (6×6 mm2 optic disc-centered, 320×320 pixels) and fundus image (FOV of 45°, optic disc-centered, 1634×1634 pixels) were captured. 
* The original data is in the Origin folder, which contains folders data_A to data_D for OCTA images of four different depths. The corresponding gray and RGB labels for each subject are in folders Label_GRAY and Label_RGB, respectively.
3. Preprocessing:
* First, run augment.py to generate augmented data for training. Second, run fusion.py to generate pseudo-color images fused with four depth images. Third, run DataMaker_Cross_valid.py to split the dataset into several subsets contain training set and testing set.

```bash
.
└── datasets
    ├── Data66
    │    ├── Fundus
    │    │    └── *.jpg
    │    └── OCTA
    │         ├── Origin     
    │         │     ├── data_A
    │         │     │     └── *.jpg
    │         │     ├── data_B
    │         │     ├── data_C
    │         │     ├── data_D
    │         │     ├── Label_GRAY
    │         │     │       └── *.tif
    │         │     └── Label_RGB
    │         │             └── *.tif
    │         ├── annotation2label.py
    │         ├── augment.py
    │         ├── DataMaker_Cross_valid.py
    │         └── fusion.py
    └── Data33
         ├── Fundus
         │    └── ...
         └── OCTA
              └── ...
```
