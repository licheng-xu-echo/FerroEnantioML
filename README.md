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
- pandas 2.3.3
- tqdm 4.67.1
- optuna 4.6.0
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


For result reproduction, please install the required packages:
```bash
pip install -r requirements.txt
```

## Demo & instructions for use
Here we provide several notebooks to demonstrate how to:
1. generate initial guess of TS geometries ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/generate_init_TS.ipynb))
2. perform semi-empirical level geometry optimizations and descriptor calculations ([demo](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/notebook/TS_opt_with_g16_xtb.ipynb))
3. performing machine learning model screening and hyperparameter optimization on specific descriptors ([other_desc_screen.py](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/other_desc_screen.py), [ligand_ferrocene_desc_screen.py](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/ligand_ferrocene_desc_screen.py))
```bash
cd scripts
python other_desc_screen.py --desc_name SPOC
python ligand_ferrocene_desc_screen.py --desc_name ACSF
# the results will be saved in the folder "results"
```
4. read the results from the hyperparameter-optimized models / descriptor screening, and plot the heatmaps ([read_screen_results.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/read_screen_results.ipynb))

5. perform feature selection ([feat_sel.py](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/feat_sel.py))
```bash
cd scripts
python feat_sel.py
# the results will be saved in the folder "results"
```
6. read the feature selection results, optimize the top‑k parameter for similar reactions, and perform reaction recommendation prediction upon completion of parameter optimization ([dist_method_topk_opt_and_recommend.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/dist_method_topk_opt_and_recommend.ipynb))
7. conduct baseline tests that comprise: (a) direct prediction using the training set's median and mean values, and (b) models utilizing one-hot encoded features as descriptors. ([run_baseline.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/run_baseline.ipynb))
8. perform out-of-sample prediction for ligands and solvents; then, remove training samples with ee > 90% before conducting reaction recommendation ([oos_prediction.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/oos_prediction.ipynb))
9. perform fully nested validation for the final TS + GB workflow, using inner cross-validation for hyperparameter / nearest-neighbor selection and outer leave-one-out cross-validation for performance estimation ([nested_screen.py](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/nested_screen.py))
```bash
cd scripts
python nested_screen.py \
  --screen ferr_lig \
  --task-ids 1 \
  --inner-folds 3 \
  --overwrite
# the results will be saved in the folder "results/nested"
```
10. summarize and visualize the GB + TS nested-validation results, including regression scatter plots for reviewer-response analyses ([gb_ts_nested_validation.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/gb_ts_nested_validation.ipynb))
11. analyze the feature importance of the final optimized model by averaging feature importances from the base and delta Gradient Boosting models across LOO splits ([final_model_feature_importance.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/final_model_feature_importance.ipynb))
12. compare xTB- and DFT-optimized transition-state geometries, including all-atom RMSD, descriptor-relevant steric-atom RMSD, and xTB-vs-DFT steric descriptor correlations ([xtb_dft_structure_steric_comparison.ipynb](https://github.com/licheng-xu-echo/FerroEnantioML/blob/main/scripts/xtb_dft_structure_steric_comparison.ipynb))

## Citation
This paper is currently under review.
