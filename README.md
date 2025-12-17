# BioVisual
Hierarchical Bio-Inspired Network for Enhanced Contour Detection in Complex Scenes
****
# Code Guidance
1. Install the libs in requirement.txt:
   
   <code>cd /path/to/your/project
   pip install -r /full/path/to/requirements.txt</code>
   
2. Download and decompress the Datasets as the next Section mentioned:
3. 
   <code>tar -xzvf HED-BSDS.tar.gz</code>

4. Modify hyperparameters in cfgs.yaml.
5. train.py:
   
   Modify Line 21-22 according to /path/to/your/project;
   
   modify Line 36 to select different traing dataset;
   
   start to train by <code>python train.py</code>.
7. test.py:
   
   Modify Line 22-24, 29 according to /path/to/your/project;
   
   uncomment the dataset which you want to use in Line 42-46;
   
   start to test by <code>python test.py</code>.
   ****
# Datasets

We use the links in RCF Repository (really thanks for that).
The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:

    <code>wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz</code>
BIPED Dataset is Here

    https://drive.google.com/drive/folders/1lZuvJxL4dvhVGgiITmZsjUJPBBrFI_bM

SLSD is Here

    https://github.com/lllltdaf2/Sealand-segmentation-data 

****

# Tools

The matlab code used for evaluation in our experiments can be downloaded in [matlab code for evaluation](https://drive.google.com/file/d/16_aqTaeSiKPwCRMwdnvFXH7b7qYL_pKB/view?usp=sharing).

The PR-curve code can be downloaded in [plot-pr-curves](https://github.com/MCG-NKU/plot-edge-pr-curves)
****
# Examples of Results
markdown
![NYUD](https://github.com/k3chencoco/BioVisual/main/examples/img_5021.png)
![NYUD](https://github.com/k3chencoco/BioVisual/main/examples/img_5021_res.png)
****
# Reference

When building our code, we referenced the repositories as follow:

[DirectSAM-RS](https://github.com/StevenMsy/DirectSAM-RS)

[PVP-UNet](https://github.com/k3chencoco/PVP-UNet)

[XYW-Net](https://github.com/PXinTao/XYW-Net)

[PidiNet](https://github.com/zhuoinoulu/pidinet)
