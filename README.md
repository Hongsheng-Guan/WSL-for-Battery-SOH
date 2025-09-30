# WSL for battery SOH estimation
This code is for our paper: Enabling Generalizable State of Health Estimation of Lithium-Ion Batteries with Extremely Minimal Labels via Weakly Supervised Learning.
> **⚠️ IMPORTANT NOTICE**  
> This repository contains code submitted for peer review.  
> **DO NOT DISTRIBUTE OR USE THIS CODE** until the paper is officially accepted.  
> Unauthorized use may compromise the review process.  

## Data description
`./data` contains processed data for Datasets 1-8, which can be used directly for model training and validation.  
The raw data for our in-house developed Datasets 1 and 2 are publicly available at [https://doi.org/10.5281/zenodo.15582113](https://doi.org/10.5281/zenodo.15582113). Datasets 3-7 are publicly available data from other laboratories: [Dataset 3](https://doi.org/10.35097/1947),[Dataset 4](https://doi.org/10.5281/zenodo.6379165), [Dataset 5](https://doi.org/10.57760/sciencedb.07456), [Dataset 6](https://www.batteryarchive.org/study_summaries.html), [Dataset 7](https://github.com/TengMichael/battery-charging-data-of-on-road-electric-vehicles), [Dataset 8](http://ivstskl.changan.com.cn/?p=2697).  
`./results` holds the **source data, model parameters and training losses** in our paper.

## Quick Start
We provide a detailed demo of our code running.
- `soh_individual_dataset.py`: The pipeline of "SOH estimation within each individual dataset".
- `soh_cross_laboratory_datasets.py`: The pipeline of "Validation by six laboratory datasets".
- `soh_pretraining_ev_data.py` and `soh_estimation_on_field_data.py`: The pipeline of "Validation by real-world driving data of electric vehicles".
- `comparison_methods_with_limited_labels.py`: The pipeline of "Comparison to supervised and semi-supervised methods in scenarios with minimal labels".
- `comparison_methods_with_enough_labels.py`: The pipeline of "Comparison to supervised methods in scenarios with sufficient single-condition labels".
- `comparison_methods_benchmark_tl.py`: The pipeline of "Comparison to transfer learning in scenarios with sufficient labels for source dataset".
- `knowledge_learned_by_the_DNN.py`: The pipeline of "Knowledge learned by the DNN within the proposed WSL framework".
- `pretraining_samples_effect.py`: The pipeline of "Impact of pre-training data volume".
- `fine_tuning_samples_effect.py`: The pipeline of "Impact of fine-tuning data volume".
- `random_six_samples_effect.py`: The pipeline of "Impact of stochastic fine-tuning of sample selection".
- `voltage_window_effect.py`: The pipeline of "Sensitivity to input voltage window".
- `ft_module_effect.py`: The pipeline of "Impact of fine-tuning strategy".
- `plot_figs.ipynb`: Plotting based on source data(`./results`)
