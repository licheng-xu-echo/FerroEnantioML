# FerroEnantioML
This repository contains the code for the paper "Synergy of Machine Learning, Computational Mechanism and Domain Expertise Enables Pd-Catalyzed C–H Asymmetric Alkenylation of Ferrocenophanes". The paper is under review.

## Requirements
The code was developed and tested using the following software packages and versions:

For python environment:
- Python 3.10.12
- RDKit 2024.03.2
- scikit-learn 1.3.0
- numpy 1.26.3
- mordred 1.2.0
- morfeus-ml 0.7.2
- xgboost 1.7.6
- lightgbm 4.2.0
- openbabel-wheel 3.1.1.21
- molop 0.1.33.2.2 (for TS initial guess generation)
- rxnb 0.3 (for semi-empirical level TS geometry optimization)

For semi-empirical level geometry optimizations and descriptor calculations:
- Gaussian 16
- xTB 6.6.1

To install MolOP, which is used to generate initial guess of TS geometries, please execute the following command:
```base
unzip MolOP-main.zip
cd MolOP-main
pip install -e .
```
To install rxnb, which is used to optimize TS geometries, please execute the following command:
```base
git clone https://github.com/licheng-xu-echo/RXNBarrier.git
cd RXNBarrier
pip install .
```


For result reproduction, please install the required packages (TODO):
```bash
pip install -r requirements.txt
```

## Demo & instructions for use
Here we provide several notebooks to demonstrate how to:
1. generate initial guess of TS geometries ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/generate_init_TS.ipynb))
2. perform semi-empirical level geometry optimizations and descriptor calculations ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/TS_opt_with_g16_xtb.ipynb))
3. screen the optimal descriptor-algorithm combinations ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/desc_model_screen.ipynb))
4. perform ML model training and experiment recommendation ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/recommend.ipynb))
5. perform out-of-sample prediction ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/out_of_sample_test.ipynb))
6. perform out-of-range prediction ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/valid_high_ee_extrapolation.ipynb))

## Citation
This paper is currently under review.