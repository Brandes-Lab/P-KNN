# P-KNN: Joint Calibration of Pathogenicity Prediction Tools
**Pathogenicity-K-Nearest-Neighbor (P-KNN)** is a command-line tool for genome-wide, non-parametric calibration of multiple variant pathogenicity prediction scores. It transforms raw scores from various prediction tools into clinically interpretable metrics:
- Posterior probabilities of a variant being pathogenic or benign.
- Log likelihood ratio (LLR) evidence strength, compatible with the [ACMG/AMP Bayesian framework](https://www.sciencedirect.com/science/article/pii/S1098360021017718?via%3Dihub) for clinical variant interpretation.

**P-KNN** represents each variant as a point in a multidimensional space, where each dimension corresponds to a prediction tool’s score. Using a labeled dataset of pathogenic and benign variants, it applies a local K-nearest neighbor (KNN) framework combined with bootstrap estimation to conservatively estimate pathogenicity based on the proportion of pathogenic neighbors.

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

### Tested Versions
P-KNN was developed and tested with the following package versions:
```{text}
python==3.13.7
numpy==2.3.3
pandas==2.3.2
scikit-learn==1.7.2
torch==2.8.0
tqdm==4.67.1
huggingface_hub==0.34.6
joblib==1.5.2
```
Note: Compatibility with other versions may vary. If you encounter issues, we recommend matching these versions in a virtual environment.

## Installation
To install P-KNN, clone the repository and use pip to install dependencies:
```bash
git clone https://github.com/Brandes-Lab/P-KNN.git
cd P-KNN
pip install .[all]  # Choose 'cpu' or 'gpu' to install the specific version, or 'all' to install both CPU and GPU versions.
```
### Installation Options
- **cpu**: Installs the CPU-only version with multiprocessing support.
- **gpu**: Installs the GPU-enabled version with CUDA acceleration.
- **all**: Installs both CPU and GPU versions for full compatibility.
Tip: If you're installing the GPU or full version, it's recommended to have at least 8GB of RAM available during installation. Otherwise, you can install the CPU version first and install Torch separately afterward.
Tip: If you're unsure which version to install, use all to ensure full compatibility.

*Alternatively, you can manually install the required packages using pip or conda before installing P-KNN. You may also run the scripts directly by downloading P_KNN.py, P_KNN_CPU.py, P_KNN_GPU.py, and P_KNN_memory_estimator.py in [P_KNN](https://github.com/Brandes-Lab/P-KNN/tree/main/P_KNN) subfolders, configure them manually and executing them as standalone Python scripts.*

## Configure P-KNN
After installing P-KNN, you can configure the default dataset paths by running:
```bash
P_KNN_config
```
This script will guide you through download the default dataset from [HuggingFace](https://huggingface.co/datasets/brandeslab/P-KNN/tree/main/dataset4commandline). Here's what it does:

### Dataset Options
You’ll be prompted to choose between academic and commercial versions of the calibration and regularization datasets (~200 MB total):
- Academic: calibration_data_dbNSFP52.csv, regularization_data_dbNSFP52.csv
- Commercial: calibration_data_dbNSFP52c.csv, regularization_data_dbNSFP52c.csv
**Note: For commercial use, please choose commercial version and optain a [dbNSFP license](https://www.dbnsfp.org/license).**

You will also be prompted to download a optional small test file (~60 KB) for validating the installation.

### Path Configuration
The script will ask you to specify a folder to save the datasets. Once selected, it automatically updates the default dataset paths used by P-KNN for future runs.

*If you prefer running P_KNN.py as a python script and would like to use the default dataset, please download manually and you can modify the default paths in the argument parser:*
```Python
parser.add_argument('--calibration_csv', default='/put the path to default calibration dataset here/',
                    help='Path to the calibration data CSV file. Default: calibration_data_dbNSFP52.csv')

parser.add_argument('--regularization_csv', default='/put the path to default regularization dataset here/',
                    help='Path to the regularization data CSV file. Default: regularization_data_dbNSFP52.csv')
```
*If you prefer to use your own calibration and regularization datasets, you can skip configuration and manually specify their paths when running P-KNN (see Run P_KNN below).*

### Preparing datasets
When preparing your query dataset or custom calibration and regularization datasets, each row should represent a single variant. The columns can include:
- **Variant identifiers** such as chromosome, position, reference and alternate alleles, or other unique identifiers
- **Prediction scores** from various tools: it's recommended to use column names ending with _score so that P-KNN can automatically detect and include them.
- **pathogenicity label**: For calibration datasets, a pathogenicity label is required. If the column is named ClinVar_annotation, P-KNN will automatically recognize it as the label column.

Here’s a conceptual example of the dataset format:
| chromosome | position | ... | prediction_tool_1_score | prediction_tool_2_score | ... | ClinVar_annotation |
|------------|----------|-----|------------------------|------------------------|-----|-------------------|
| 1          | 955677   | ... | 0.77                   | 2.14                   | ... | 0                 |
| 1          | 977396   | ... | 0.25                   | 1.80                   | ... | 0                 |
| 1          | 978801   | ... | 0.04                   | 1.02                   | ... | 1                 |


## Run P-KNN
You can run P-KNN joint calibration from the command line using the default dataset downloaded during `P_KNN_config` with only the required arguments:
```bash
P_KNN \
  --query_csv path/to/query.csv \
  --output_dir path/to/output_folder \
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

### Optional paremeters
- **tool_list**: Comma-separated list of prediction tool columns to use for calibration (e.g., SIFT_score,FATHMM_score,VEST4_score). Default: auto (automatically detects *_score columns present in all input files).
- **calibration_label**: Column name in the calibration CSV containing binary labels (default: ClinVar_annotation).
- **p_prior**: Prior probability of a variant being pathogenic (default: 0.0441 according to [ClinGen](https://linkinghub.elsevier.com/retrieve/pii/S0002-9297(22)00461-X)).
- **n_calibration_in_window**: Minimum number of calibration variants per local window (default: 100).
- **frac_regularization_in_window**: Minimum fraction of regularization samples per window (default: 0.03).
- **normalization**: Score normalization method ("rank" or "z", default: rank).
- **impute**: Whether to impute missing values with KNN imputation (default: True).
- **mi_scaling**: Whether to apply mutual information-based scaling (default: True).
- **n_bootstrap**: Number of bootstrap iterations for uncertainty estimation (default: 100).
- **bootstrap_alpha_error**: One-tailed alpha value for credible intervals (e.g. 0.05 for 95% CI, default: 0.05).

### Execution settings
- **device**: Computation device ("GPU", "CPU", or "auto", default: auto, which auto-detect GPU if available).
- **batch_size**: Batch size for GPU processing (default: 512).
- **cpu_parallel**: Whether to run CPU computations in parallel (default: True).
- **query_chunk_size**: Split query into chunks to reduce memory usage (optional, default: None).

## Estimate memory requirment
You can estimate the memory requirment of P-KNN from the command line using:
```
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
