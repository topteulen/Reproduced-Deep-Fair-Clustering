# Reproduced: Deep Fair Clustering

Peizhao Li, Han Zhao, and Hongfu Liu. "[Deep Fair Clustering for Visual Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Deep_Fair_Clustering_for_Visual_Learning_CVPR_2020_paper.html)", CVPR 2020.

### Prerequisites
Anaconda: https://www.anaconda.com/distribution/

### Getting Started
First, open the Anaconda prompt, clone the entire repository and move to this directory:
```bash
git clone https://github.com/topteulen/UVA-FACT.git
```

Then create and activate the environment with the following commands:
```bash
conda env create -f environment.yml
conda activate DFC
```

To view the notebook that runs the results run the following command:
```bash
jupyter notebook Results.ipynb
```

If you want to do your own training on the datasets you can use:
```bash
 main.py [-h] [--bs BS] [--k K] [--lr LR] [--iters ITERS]
               [--test_interval TEST_INTERVAL] [--adv_mult ADV_MULT]
               [--coeff_fair COEFF_FAIR] [--coeff_par COEFF_PAR] [--gpu GPU]
               [--seed SEED] [--corrupted CORRUPTED]
               [--corrupted_set CORRUPTED_SET] [--dataset DATASET]
               [--divergence DIVERGENCE]
```
Some of the more notable arguments are divergence which allows the user to set 
one of the following 3 divergence functions: KS, JS, CS. Furthermore, we have the 
--dataset which allows the user to choose between: Rmnist and usps. Finally,
the --corrupted_set allows which part of the dataset is to be corrupted and 
the --corrupted the amount by which this is done.

If something is unclear with the arguments you can always 
```bash
run python main.py --help
```

### Results 
The following results should be reproducable when running the notebook:

Results reproduced:
![reproducability](/save/reproducability.png)

Results corrupted sensitive attribute:
![corruption](/save/corruption.png)

Results pretrained cluster centers:
![cluster](/save/cluster.png)

Results divergence functions:
![divergence](/save/divergence.png)



## Reference
    @InProceedings{Li_2020_CVPR,
    author = {Li, Peizhao and Zhao, Han and Liu, Hongfu},
    title = {Deep Fair Clustering for Visual Learning},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }
