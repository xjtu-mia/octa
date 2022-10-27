# AV-casNet
AV-casNet: Fully Automatic Arteriole-Venule Segmentation and Differentiation in OCT Angiography

## 1. Prepare OCTA datasets 
You can download the datasets from the [Baidu Netdisk](https://pan.baidu.com/s/1sQboL5_6vvk9hAuANQpbiA?pwd=n4i4) (access code: n4i4) and decompress it to the root directory. Please go to ["./datasets/README.md"](datasets/README.md) for details. 

## 2. Environment

- Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.
- You should also install MATLAB 2020b or later MATLAB version for running post-processing and evaluation code.

## 3. Run the whole workflow

- the whole workflow includes five main steps: training and testing a CNN model, building the graph data based on CNN predictions, and training and testing a GAT model. If you want to run all steps automatically, you can type the following:
```bash
python terminal.py --train_cnn=1 --test_cnn=1 --build_graph=1 --train_gat=1 --test_gat=1
```
- where =1 means execute the process and =0 means not to execute.
- The binaried segmentation results will be saved in the folder GAT_final. If you want to see the probability map and superpixel results, you can go to the result folders in ["./Core/UNet/"](Core/UNet/Result/) and ["./Core/GAT/"](Core/GAT/result_9/) for details.
- If you want to run some steps independently, you can simply set the parameter value of the corresponding step to 1 and the parameter value of the other steps to 0. Take training and testing a CNN model alone as an example:
```bash
python terminal.py --train_cnn=1 --test_cnn=1 --build_graph=0 --train_gat=0 --test_gat=0
```

## 4. Post-processing and evaluation

- This part of the code is implemented by MATLAB 2020b. When you have completed all the above processes, you can perform ["./post_processing.mlx"](post_processing.mlx) in MATLAB to get the final visual segmentation results and quantitative evaluation results.

## Citation

```bibtex
@misc{cao2021swinunet,
      title={AV-casNet: Fully Automatic Arteriole-Venule Segmentation and Differentiation in OCT Angiography}, 
      author={Xiayu Xu, Peiwei Yang, Hualin Wang, Zhanfeng Xiao, Gang Xing, Xiulan Zhang, Wei Wang, Feng Xu, Jiong Zhang, Jianqin Lei},
      year={2022},
}
```
