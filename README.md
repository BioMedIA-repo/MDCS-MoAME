# MDCS_MoAME

This is the official implementation of the paper "MDCS-MoAME: Multi-directional Composite Scanning with Mixture of Attention and Mamba Experts for Cancer Survival Prediction"  

## 1.Prepare the environment.

- Set up the environment according to environment.yal.

  ```bash
  pip install -r requirements.txt
  ```

- Add the files in ./mamba/mamba_ssm/modules to your environment (e.g., /env_name/lib/python3.10/site-packages/mamba_ssm/modules).

## 2. Process WSIs data

- Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/).

- Run ./preprocess/create_regions_fp.py. We divide WSIs into 4096 × 4096 regions at a magnification of 10x.

- Run ./preprocess/MDCS_wsi.py.  We apply our proposed MDCS strategy to regions.

- Run ./preprocess/extract_features_fp.py. We will further subdivide the regions into 512 × 512 patches and use ResNet-50 to extract features from the regions and patches. The final structure of datasets should be as following:

  ```bash
  DATA_ROOT_DIR/
      └──pt_files/
          ├── slide_1.pt
          ├── slide_2.pt
          └── ...
  ```

​	DATA_ROOT_DIR is the base directory of cancer type (e.g. the directory to TCGA_BLCA), which should be passed to the model with the argument `--data_root_dir` as shown in blca.sh.

## 3.Prepare genomic profiles data

- Unzip ./csv/**.zip to obtain genomic data for BLCA, BRCA, GBMLGG, LUAD, and UCEC datasets.

## 4.Running Experiments

Now you can run our method using the following command:

```bash
# BLCA dataset
bash blca.sh
# BRCA dataset
bash brca.sh
# GBMLGG dataset
bash gbmlgg.sh
# LUAD dataset
bash luad.sh
# UCEC dataset
bash ucec.sh
```
## Citations
If the code is helpful for your research, please consider citing:
```angular2
@inproceedings{qu2026mdcs,
  author={Linjie Qu, Jin Xiao, Xiangrong Liu, Changming Sun, Hui Cui, Yuqi Fang, Ran Su, Qiangguo Jin, leyi wei },
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title={MDCS-MoAME: Multi-directional Composite Scanning with Mixture of Attention and Mamba Experts for Cancer Survival Prediction}, 
  year={2026},
  pages={1-10}
}
```
## Social media
<p align="center"><img width="600" alt="image" src="https://github.com/BioMedIA-repo/.github/blob/052046a248d3831a599e11c85ff94cdd658c5abc/pic/wechat.png" height=""></p> 
Welcome to follow our [Wechat official account: iBioMedInfo] and [Xiaohongshu official account: iBioMedInfo], we will share recent studies on biomedical image and bioinformation analysis there. 

## Global Collaboration & Questions

Global Collaboration: We're on a mission to biomedical research, aiming for artificial intelligence and its applications to biomedical image and bioinformation analysis, promoting the development of the medical community. Collaborate with us to increase competitiveness.

Questions: General questions, please contact '23020241154348@xmu.stu.edu.cn'