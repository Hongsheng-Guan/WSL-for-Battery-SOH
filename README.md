# WSL for battery SOH estimation
This code is for our paper: Enabling Generalizable Lithium-Ion Battery State of Health Estimation with Minimal Labels via Weakly Supervised Pre-Training.
> **⚠️ IMPORTANT NOTICE**  
> This repository contains code submitted for peer review.  
> **DO NOT DISTRIBUTE OR USE THIS CODE** until the paper is officially accepted.  
> Unauthorized use may compromise the review process.  

## Data description
`./data` contains processed data for Datasets 1-7, which can be used directly for model training and validation.  
The raw data for our in-house developed Datasets 1 and 2 are publicly available at [https://doi.org/10.5281/zenodo.15582113](https://doi.org/10.5281/zenodo.15582113). Datasets 3-7 are publicly available data from other laboratories: [Dataset 3](https://doi.org/10.35097/1947),[Dataset 4](https://doi.org/10.5281/zenodo.6379165), [Dataset 5](https://doi.org/10.57760/sciencedb.07456), [Dataset 6](https://www.batteryarchive.org/study_summaries.html), [Dataset 7](https://github.com/TengMichael/battery-charging-data-of-on-road-electric-vehicles).  
`./results` holds the **source data, model parameters and training losses** in our paper.

## Quick Start
### Prerequisites
Python 3.8+, PyTorch 1.10+， numpy 1.22+, pandas 2.0+, scikit-learn 1.5+

### Demo
We provide a detailed demo of our code running.
- `soh_individual_dataset.py`: The pipeline of "SOH estimation within each individual dataset".
- `soh_cross_laboratory_datasets.py`: The pipeline of "Validation from the six laboratory datasets".
- `soh_pretraining_ev_data.py`: The pipeline of "Validation from real-world driving data of electric vehicles".
- `comparison_methods_with_limited_labels.py`: The pipeline of "Comparison with supervised/semi-supervised methods in scenarios of extremely limited labels".
- `comparison_methods_with_enough_labels.py`: The pipeline of "Comparison with supervised methods in scenarios of sufficient single-condition labels" and "Comparison with supervised methods in scenarios of sufficient full-condition labels".
- `comparison_methods_benchmark_tl.py`: The pipeline of "Comparison with transfer learning in scenarios of sufficient labels for the source dataset".
- `pretraining_samples_effect.py` and `fine_tuning_samples_effect.py`: The pipeline of "Effect of the pre-training and fine-tuning data sizes".
- `plot_figs.ipynb`: Plotting based on source data(`./results`)
