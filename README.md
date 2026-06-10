# P-KNN: Joint Calibration of Pathogenicity Prediction Tools
**Pathogenicity-K-Nearest-Neighbor (P-KNN)** is a command-line tool for genome-wide, non-parametric calibration of multiple variant pathogenicity prediction scores. It transforms raw scores from various prediction tools into clinically interpretable metrics:
- Posterior probabilities of a variant being pathogenic or benign.
- Log likelihood ratio (LLR) evidence strength, compatible with the [ACMG/AMP Bayesian framework](https://www.sciencedirect.com/science/article/pii/S1098360021017718?via%3Dihub) for clinical variant interpretation.

**P-KNN** represents each variant as a point in a multidimensional space, where each dimension corresponds to a prediction tool's score. Using a labeled dataset of pathogenic and benign variants, it applies a local K-nearest neighbor (KNN) framework combined with bootstrap estimation to conservatively estimate pathogenicity based on the proportion of pathogenic neighbors.

P-KNN requires two key datasets:
- **Calibration dataset**: A labeled set of pathogenic and benign variants used to estimate posterior probabilities.
- **Regularization dataset**: An unlabeled set of variants that reflects the general distribution of variants across the human genome. This dataset is used to regularize the minimum search radius for K-nearest neighbors, preventing overly narrow local neighborhoods and improving generalizability.

![Calibration Concept](https://github.com/Brandes-Lab/P-KNN/blob/main/Calibration_concept.jpg)

## Requirements
P-KNN is written in **Python 3** and depends on the following packages:
| Package        | Purpose                                                  |
|----------------|----------------------------------------------------------|
| `numpy`        | Numerical operations                                     |
| `pandas`       | Reading and manipulating tabular CSV data                |
| `scikit-learn` | Imputation, mutual information scaling, etc.             |
| `tqdm`         | Progress bar for bootstraping                            |
| `joblib`       | (CPU mode only) Parallel computation support             |
| `torch`        | (GPU mode only) CUDA acceleration                        |

### Compatibility
P-KNN requires **Python 3.9 or later** and is compatible with a range of package versions. The minimum supported versions are:

| Package           | Minimum Version |
|-------------------|-----------------|
| `numpy`           | 1.24            |
| `pandas`          | 1.5.3           |
| `scikit-learn`    | 1.2.0           |
| `tqdm`            | 4.60.0          |
| `huggingface_hub` | 0.16.2          |
| `joblib`          | 1.2.0           |
| `torch`           | 2.0.0           |

### Tested Versions
P-KNN was developed and tested with the following package versions:
```text
python==3.13.7
numpy==2.3.3
pandas==2.3.2
scikit-learn==1.7.2
torch==2.8.0
tqdm==4.67.1
huggingface_hub==0.34.6
joblib==1.5.2
```

## Installation

### Option 1: Install via pip (recommended)
```bash
pip install "P_KNN[all]"  # Choose 'cpu' or 'gpu' to install the specific version, or 'all' to install both.
```

### Option 2: Install from source
```bash
git clone https://github.com/Brandes-Lab/P-KNN.git
cd P-KNN
pip install ".[all]"  # Choose 'cpu' or 'gpu' to install the specific version, or 'all' to install both.
```

### Option 3: Install via conda environment file
```bash
git clone https://github.com/Brandes-Lab/P-KNN.git
cd P-KNN
conda env create -f environment.yml
conda activate P_KNN
pip install ".[all]"
```

### Installation Options
- **cpu**: Installs the CPU-only version with multiprocessing support.
- **gpu**: Installs the GPU-enabled version with CUDA acceleration.
- **all**: Installs both CPU and GPU versions for full compatibility.

> **Tip**: If you're installing the GPU or full version, it's recommended to have at least 8GB of RAM available during installation. Otherwise, you can install the CPU version first and install Torch separately afterward.

> **Tip**: If you're unsure which version to install, use `all` to ensure full compatibility.

> **Note on GPU support**: PyTorch's CUDA-enabled version must be installed separately by specifying the appropriate index URL. If `torch.cuda.is_available()` returns `False` after installation, please reinstall PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version. For example:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

*Alternatively, you may run the scripts directly by downloading P_KNN.py, P_KNN_CPU.py, P_KNN_GPU.py, and P_KNN_memory_estimator.py from the [P_KNN](https://github.com/Brandes-Lab/P-KNN/tree/main/P_KNN) subfolder, configure them manually, and execute them as standalone Python scripts.*

## Configure P-KNN
After installing P-KNN, configure the default dataset paths by running:
```bash
P_KNN_config
```
This interactive script guides you through downloading the default datasets from [HuggingFace](https://huggingface.co/datasets/brandeslab/P-KNN/tree/main/dataset4commandline). You will be prompted with the following steps:

**1. Download default datasets (~200 MB)?**
Enter `yes` to proceed.

**2. Enter folder to save datasets**
Specify a local directory where the datasets will be saved.

**3. Choose dataset version**
Enter `academic` or `commercial`:
- `academic`: calibration_data_dbNSFP52.csv, regularization_data_dbNSFP52.csv
- `commercial`: calibration_data_dbNSFP52c.csv, regularization_data_dbNSFP52c.csv

> **Note**: For commercial use, please choose the commercial version and obtain a [dbNSFP license](https://www.dbnsfp.org/license).

**4. Download test file and run installation test (~5 minutes)?**
Enter `yes` to download a small test file (~60 KB) and optionally run an automated installation test. The test runs P-KNN on 100 variants using 5 prediction tools and verifies that the output log likelihood ratio (LLR) values match the expected reference results within a tolerance of ±0.1. A `PASSED` message confirms your installation is working correctly.

### Path Configuration
Once datasets are downloaded, the script automatically updates the default dataset paths used by P-KNN for future runs.

*If you prefer running P_KNN.py as a standalone Python script and would like to use the default datasets, please download them manually and modify the default paths in the argument parser:*
```python
parser.add_argument('--calibration_csv', default='/put the path to default calibration dataset here/',
                    help='Path to the calibration data CSV file. Default: calibration_data_dbNSFP52.csv')

parser.add_argument('--regularization_csv', default='/put the path to default regularization dataset here/',
                    help='Path to the regularization data CSV file. Default: regularization_data_dbNSFP52.csv')
```
*If you prefer to use your own calibration and regularization datasets, you can skip configuration and manually specify their paths when running P-KNN (see Run P-KNN below).*

### Preparing datasets
When preparing your query dataset or custom calibration and regularization datasets, each row should represent a single variant. The columns can include:
- **Variant identifiers** such as chromosome, position, reference and alternate alleles, or other unique identifiers.
- **Prediction scores** from various tools: it's recommended to use column names ending with `_score` so that P-KNN can automatically detect and include them.
- **Pathogenicity label**: For calibration datasets, a pathogenicity label is required. If the column is named `ClinVar_annotation`, P-KNN will automatically recognize it as the label column.

Here's a conceptual example of the dataset format:
| chromosome | position | ... | prediction_tool_1_score | prediction_tool_2_score | ... | ClinVar_annotation |
|------------|----------|-----|------------------------|------------------------|-----|-------------------|
| 1          | 955677   | ... | 0.77                   | 2.14                   | ... | 0                 |
| 1          | 977396   | ... | 0.25                   | 1.80                   | ... | 0                 |
| 1          | 978801   | ... | 0.04                   | 1.02                   | ... | 1                 |


## Run P-KNN
You can run P-KNN joint calibration from the command line using the default datasets downloaded during `P_KNN_config` with only the required arguments:
```bash
P_KNN \
  --query_csv path/to/query.csv \
  --output_dir path/to/output_folder
```
You can also customize P-KNN using a full set of configurable parameters. For example:
```bash
P_KNN \
  --query_csv path/to/query.csv \
  --output_dir path/to/output_folder \
  --calibration_csv path/to/calibration_data.csv \
  --regularization_csv path/to/regularization_data.csv \
  --tool_list Tool1_score,Tool2_score,Tool3_score,Tool4_score \
  --calibration_label ClinVar_annotation \
  --p_prior 0.0441 \
  --n_calibration_in_window 100 \
  --frac_regularization_in_window 0.03 \
  --normalization rank \
  --impute True \
  --mi_scaling True \
  --n_bootstrap 100 \
  --bootstrap_alpha_error 0.05 \
  --device auto \
  --batch_size 512 \
  --cpu_parallel True \
  --query_chunk_size 512000
```

### Required Arguments
- **query_csv**: Path to your query variant CSV file containing raw scores to be calibrated.
- **output_dir**: Directory where the result CSV and log files will be saved.

### Optional Files
- **calibration_csv**: Path to the calibration data CSV file. If you used the configuration script, the default path will be set automatically.
- **regularization_csv**: Path to the regularization data CSV file. The default path will be set during configuration.

### Optional Parameters
- **tool_list**: Comma-separated list of prediction tool columns to use for calibration (e.g., SIFT_score,FATHMM_score,VEST4_score). Default: `auto` (automatically detects `*_score` columns present in all input files).
- **calibration_label**: Column name in the calibration CSV containing binary labels (default: `ClinVar_annotation`).
- **p_prior**: Prior probability of a variant being pathogenic (default: 0.0441 according to [ClinGen](https://linkinghub.elsevier.com/retrieve/pii/S0002-9297(22)00461-X)).
- **n_calibration_in_window**: Minimum number of calibration variants per local window (default: 100).
- **frac_regularization_in_window**: Minimum fraction of regularization samples per window (default: 0.03).
- **normalization**: Score normalization method (`rank` or `z`, default: `rank`).
- **impute**: Whether to impute missing values with KNN imputation (default: `True`).
- **mi_scaling**: Whether to apply mutual information-based scaling (default: `True`).
- **n_bootstrap**: Number of bootstrap iterations for uncertainty estimation (default: 100).
- **bootstrap_alpha_error**: One-tailed alpha value for credible intervals (e.g. 0.05 for 95% CI, default: 0.05).

### Execution Settings
- **device**: Computation device (`GPU`, `CPU`, or `auto`, default: `auto`, which auto-detects GPU if available).
- **batch_size**: Batch size for GPU processing (default: 512).
- **cpu_parallel**: Whether to run CPU computations in parallel (default: `True`).
- **query_chunk_size**: Split query into chunks to reduce memory usage (optional, default: `None`).

## Estimate Memory Requirements
You can estimate the memory requirements of P-KNN from the command line using:
```bash
P_KNN_memory_estimator \
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
- **mode**: Memory estimation mode: `cpu` or `gpu` (default: `gpu`).
- **n_bootstrap**: Number of bootstrap iterations (default: 100).

### Arguments for GPU Mode
- **batch_size**: Query batch size for GPU mode (default: 512).
- **vram_gb**: Available GPU memory in GiB (used to check for OOM risk; optional).

### Arguments for CPU Parallel Computing Mode
- **n_cpu_threads**: Number of CPU threads for parallel execution (default: 1).

### Optional Arguments
- **dtype**: Floating point precision (`float32` or `float64`) (default: `float64`).
- **index_dtype**: Index data type (`int32` or `int64`) (default: `int64`).
- **cdist_overhead**: Overhead multiplier for pairwise distance computation (default: 1.3).
- **sort_overhead**: Overhead multiplier for sorting and top-k operations (default: 2.0).
- **imputer_overhead**: Overhead multiplier for imputation memory use (default: 1.5).
- **safety_factor**: Final safety margin multiplier (default: 1.2).

## Related Resources
- **Precomputed score dataset**: [Hugging Face brandeslab/P-KNN](https://huggingface.co/datasets/brandeslab/P-KNN) These precomputed scores are derived from dbNSFP. Users are strictly bound by the [dbNSFP licensing terms](https://www.dbnsfp.org/license). For commercial use, you must obtain a commercial license directly from dbNSFP.
- **Gene based precomputed score viewer**: [P-KNN-Viewer](https://huggingface.co/spaces/brandeslab/P-KNN-Viewer)
- **Manuscript**: [P-KNN: Maximizing variant classification evidence through joint calibration of multiple pathogenicity prediction tools](https://doi.org/10.1101/2025.09.24.678417)
- **dbNSFP License**: [dbNSFP Commercial Use Requirements](https://www.dbnsfp.org/license)
