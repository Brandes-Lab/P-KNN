# P-KNN: Joint Calibration of Pathogenicity Prediction Tools
**Pathogenicity-K-Nearest-Neighbor (P-KNN)** is a command-line tool for genome-wide, non-parametric calibration of multiple variant pathogenicity prediction scores. It transforms raw prediction scores from all tools into clinical interpretable metrics:
- Posterior probabilities of a variant being pathogenic or benign
- Log likelihood ratio (LLR) evidence strength, compatible with the [ACMG/AMP Bayesian framework](https://www.sciencedirect.com/science/article/pii/S1098360021017718?via%3Dihub) for clinical variant interpretation

**P-KNN** represents each variant as a point in a multidimensional space, with each dimension corresponding to a prediction tool’s score. Using a labeled dataset of pathogenic and benign variants, it applies a local K-nearest neighbor (KNN) framework combined with bootstrap estimation to conservatively estimate pathogenicity based on the proportion of pathogenic neighbors.

![Calibration Concept](https://github.com/Brandes-Lab/P-KNN/blob/main/Calibration_concept.jpg)


## Installation
```bash
git clone https://github.com/Brandes-Lab/P-KNN.git
cd P-KNN
pip install .[all]  # Choose 'cpu' or 'gpu' to install the specific version, or 'all' to install both CPU and GPU versions.
```
### options
- **cpu**: Installs the CPU-multiprocessing version of the package.
- **gpu**: Installs the GPU-enabled version.
- **all**: Installs both CPU and GPU versions

## Requirements
P-KNN is written in **Python 3** and requires the following packages:

| Package        | Purpose                                                  |
|----------------|----------------------------------------------------------|
| `numpy`        | Numerical operations                                     |
| `pandas`       | Reading and manipulating tabular csv data                |
| `scikit-learn` | Impute missing value, mutual information scaling, etc.   |
| `tqdm`         | Progress bar for bootstraping                            |
| `joblib`       | (Only for CPU mode) parallel computation support         |
| `torch`        | (Only for GPU mode) required for CUDA acceleration       |

You can install all core packages via pip or conda:

### pip
```bash
pip install numpy pandas scikit-learn tqdm
```
To support CPU parallelization, add:
```bash
pip install joblib
```
To support GPU acceleration, add:
```bash
pip install torch
```

### conda
```bash
conda create -n pknn python numpy pandas scikit-learn tqdm
conda activate pknn
```
To support CPU parallelization, add:
```bash
conda install joblib
```
To support GPU acceleration, add:
```bash
pip install torch
```

## run P-KNN
You can run the P-KNN joint calibration from the command line using:
```
python P_KNN.py \
  --query_csv path/to/query.csv \
  --output_dir path/to/output_folder \
  --calibration_csv path/to/calibration_data.csv \
  --regularization_csv path/to/regularization_data.csv \
  --tool_list SIFT_score,FATHMM_score,VEST4_score \
  --calibration_label ClinVar_annotation \
  --p_prior 0.0441 \
  --n_calibration_in_window 100 \
  --frac_regularization_in_window 0.03 \
  --normalization rank \
  --impute True \
  --mi_scaling True \
  --n_bootstrap 100 \
  --bootstrap_alpha_error 0.05 \
  --device GPU \
  --batch_size 512 \
  --cpu_parallel True \
  --query_chunk_size 512000
```
### Required arguments
- **query_csv**: Path to your query variant file with raw scores to be calibrated.
- **output_dir**: Directory where result CSV and log files will be written.
### Optional files
- **calibration_csv**: Calibration data file path (default: calibration_data_dbNSFP52.csv).
- **regularization_csv**: Regularization data file path (default: regularization_data_dbNSFP52.csv).
### Optional paremeters
- **tool_list**: Comma-separated prediction tool names to use for scoring (default: tools from dbNSFPv5.2a whose training data did not overlap with variants in calibration_data_dbNSFP52.csv).
- **calibration_label**: Column name in calibration file containing binary pathogenic/benign labels, default is ClinVar_annotation.
- **p_prior**: Prior probability of pathogenicity (default: 0.0441).
- **n_calibration_in_window**: Min # of calibration samples per local region (default: 100).
- **frac_regularization_in_window**: Fraction of regularization data to regularize each region (default: 0.03).
- **normalization**: Score normalization method (rank or z) (default: rank).
- **impute**: Use KNN imputation for missing values (True or False) (default: True).
- **mi_scaling**: Apply mutual information-based scaling (True or False) (default: True).
- **n_bootstrap**: Number of bootstrap iterations (default: 100).
- **bootstrap_alpha_error**: Alpha level for credible intervals (e.g. 0.05 for 95% CI) (default: 0.05).
### Execution settings
- **device**: Run on GPU or CPU (default: GPU).
- **batch_size**: Batch size for GPU mode (default: 512).
- **cpu_parallel**: Enable CPU multiprocessing (default: True).
- **query_chunk_size**: Split query into chunks of this size to reduce memory usage (default: None).

## Estimate memory requirment
You can estimate the memory requirment of P-KNN from the command line using:
```
python P_KNN_memory_estimator.py \
  --n_tools 27 \
  --n_query 512000 \
  --n_calibration 11000 \
  --n_regularization 350000 \
  --n_bootstrap 100 \
  --n_cpu_threads 1 \
  --batch_size 512 \
  --dtype float64 \
  --index_dtype int64 \
  --cdist_overhead 1.3 \
  --sort_overhead 2 \
  --imputer_overhead 1.5 \
  --safety_factor 1.2 \
  --vram_gb 16 \
  --mode gpu
```
### Arguments
- **n_tools**: Number of predictive tools used in the model.
- **n_query**: Number of variants in the query dataset.
- **n_calibration**: Number of variants in the calibration dataset.
- **n_regularization**: Number of variants in the regularization dataset.
- **mode**: Memory estimation mode: cpu or gpu (default: gpu).
- **n_bootstrap**: Number of bootstrap iterations (default: 100).
### Argument for GPU mode:
- **batch_size**: Query batch size for GPU mode (default: 512).
- **vram_gb**: Available GPU memory in GiB (used to check for OOM risk; optional).
### Argument for CPU parallel computing mode:
- **n_cpu_threads**: Number of CPU threads for parallel execution (default: 1).

### Optional argument
- **dtype**: Floating point precision (float32 or float64) (default: float64).
- **index_dtype**: Index data type (int32 or int64) (default: int64).
- **cdist_overhead**: Overhead multiplier for pairwise distance computation (default: 1.3).
- **sort_overhead**: Overhead multiplier for sorting and top-k operations (default: 2.0).
- **imputer_overhead**: Overhead multiplier for imputation memory use (default: 1.5).
- **safety_factor**: Final safety margin multiplier (default: 1.2).
