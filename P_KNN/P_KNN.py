#!/usr/bin/env python3
import argparse
import os
import sys
import gc
import logging
import traceback
import pandas as pd
import numpy as np
from math import ceil


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser(description='Run P-KNN Joint Calibration.')

    parser.add_argument('--query_csv', required=True,
                        help='Path to input query CSV file containing variant scores to be calibrated.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where the output CSV and log files will be saved.')

    parser.add_argument('--calibration_csv', default=os.path.join(os.path.dirname(__file__), 'calibration_data_dbNSFP52.csv'),
                        help='Path to the calibration data CSV file. Default: path to calibration_data_dbNSFP52.csv')
    parser.add_argument('--regularization_csv', default=os.path.join(os.path.dirname(__file__), 'regularization_data_dbNSFP52.csv'),
                        help='Path to the regularization data CSV file. Default: path toregularization_data_dbNSFP52.csv')

    parser.add_argument('--tool_list', type=str, default="auto",
                        help='Comma-separated list of tool columns to use for calibration (e.g., SIFT_score,FATHMM_score,VEST4_score). Default: auto (auto-detect *_score columns present in all input csv files).')
    parser.add_argument('--calibration_label', type=str, default='ClinVar_annotation',
                        help='Column name in calibration CSV that contains binary labels (e.g., ClinVar_annotation).')

    parser.add_argument('--p_prior', type=float, default=0.0441,
                        help='Prior probability of a variant being pathogenic. Default: 0.0441')
    parser.add_argument('--n_calibration_in_window', type=int, default=100,
                        help='Minimum number of calibration variants used in each local window. Default: 100')
    parser.add_argument('--frac_regularization_in_window', type=float, default=0.03,
                        help='Minimum fraction of regularization samples to include in each local window. Default: 0.03')
    parser.add_argument('--normalization', default='rank',
                        help='Normalization method for scores (e.g., "rank" or "z"). Default: rank')
    parser.add_argument('--impute', type=str2bool, default=True,
                        help='Whether to impute missing values with KNNImputer. Accepts True or False. Default: True')

    parser.add_argument('--mi_scaling', type=str2bool, default=True,
                        help='Whether to apply mutual information scaling. Accepts True or False. Default: True')
    parser.add_argument('--n_bootstrap', type=int, default=100,
                        help='Number of bootstrap iterations for estimating prediction uncertainty. Default: 100')
    parser.add_argument('--bootstrap_alpha_error', type=float, default=0.05,
                        help='One-tailed alpha value (1 - confidence) used in bootstrap-derived credible intervals. Default: 0.05')

    parser.add_argument('--device', default='auto', choices=['GPU', 'CPU', 'auto'],
                        help='Which device to use for computation. Choose GPU, CPU, or auto. Default: auto (auto-detect GPU if available)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for GPU processing. Ignored if device=CPU. Default: 512 (suitable for VRAM 16GB)')
    parser.add_argument('--cpu_parallel', type=str2bool, default=True,
                        help='Whether to run CPU computations in parallel. Accepts True or False. Ignored if device=GPU. Default: True')

    parser.add_argument('--query_chunk_size', type=int, default=None,
                        help='Optional: split query into chunks of this size to reduce memory usage (e.g., 2560000 for 10GB memory in GPU mode).')

    args = parser.parse_args()

    # Auto-detect device if needed
    if args.device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                args.device = 'GPU'
            else:
                args.device = 'CPU'
        except ImportError:
            args.device = 'CPU'

    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    query_name = os.path.splitext(os.path.basename(args.query_csv))[0]
    output_csv = os.path.join(args.output_dir, f"P_KNN_{query_name}.csv")
    notool_csv = os.path.join(args.output_dir, f"query_notool_{query_name}.csv")
    log_path = os.path.join(args.output_dir, f"P_KNN_{query_name}.log")
    setup_logger(log_path)

    if os.path.exists(output_csv):
        backup_csv = f"{output_csv}_backup_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        logging.warning(f"Output CSV file '{output_csv}' already exists. Renaming it to '{backup_csv}'.")
        os.rename(output_csv, backup_csv)

    try:
        logging.info("===== Start P-KNN =====")
        logging.info(f"Device: {args.device}")
        logging.info(f"Query CSV: {args.query_csv}")
        logging.info(f"Output CSV: {output_csv}")
        logging.info(f"Calibration CSV: {args.calibration_csv}")
        logging.info(f"Regularization CSV: {args.regularization_csv}")
        
        # Load calibration and regularization once
        if not os.path.exists(args.query_csv):
            logging.error(f"Query CSV file '{args.query_csv}' does not exist.")
            sys.exit(1)
        if os.path.getsize(args.query_csv) == 0:
            logging.error(f"Query CSV file '{args.query_csv}' is empty.")
            sys.exit(1)
        if not os.path.exists(args.calibration_csv):
            logging.error(f"Calibration CSV file '{args.calibration_csv}' does not exist.")
            sys.exit(1)
        if not os.path.exists(args.regularization_csv):
            logging.error(f"Regularization CSV file '{args.regularization_csv}' does not exist.")
            sys.exit(1)
        if not os.access(args.output_dir, os.W_OK):
            logging.error(f"Output directory '{args.output_dir}' is not writable.")
            sys.exit(1)

        # Automatic detecting calibration_label and tool_list
        calibration_data = pd.read_csv(args.calibration_csv, low_memory=False)
        regularization_data = pd.read_csv(args.regularization_csv, low_memory=False)
        query_data_head = pd.read_csv(args.query_csv, low_memory=False, nrows=1)

        if args.calibration_label not in calibration_data.columns:
            logging.error(f"Calibration label '{args.calibration_label}' not found in calibration data.")
            sys.exit(1)

        if args.tool_list.strip().lower() == "auto":
            # end with _score and exist in all three files
            query_score_cols = [col for col in query_data_head.columns if col.endswith('_score')]
            calibration_cols = set(calibration_data.columns)
            regularization_cols = set(regularization_data.columns)
            auto_tool_list = [col for col in query_score_cols if col in calibration_cols and col in regularization_cols]
            if not auto_tool_list:
                logging.error("No matching *_score columns found in all three files.")
                sys.exit(1)
            tool_list = auto_tool_list
            logging.info(f"Auto-detected tool_list: {tool_list}")
        else:
            tool_list = args.tool_list.split(',')
            # check if all files have these columns
            missing = []
            for col in tool_list:
                if col not in query_data_head.columns or col not in calibration_data.columns or col not in regularization_data.columns:
                    missing.append(col)
            if missing:
                logging.error(f"The following tool columns are missing in one or more files: {missing}")
                sys.exit(1)
            logging.info(f"Using user-specified tool_list: {tool_list}")
            
        for df, name in zip([calibration_data, regularization_data],
                            ['calibration_data', 'regularization_data']):
            df[tool_list] = df[tool_list].apply(lambda col: pd.to_numeric(col, errors='coerce'))
            missing_ratio = df.isna().sum()[tool_list] / len(df)
            logging.info(f'Missing ratio in {name}:\n{missing_ratio}')
            if df[tool_list].isna().all(axis=1).sum() > 0:
                logging.warning(f'Excluded {df[tool_list].isna().all(axis=1).sum()} of {len(df)} variants from {name} due to missing scores in all prediction tools.')

        calibration_data = calibration_data[calibration_data[tool_list].notna().any(axis=1)].reset_index(drop=True)
        regularization_data = regularization_data[regularization_data[tool_list].notna().any(axis=1)].reset_index(drop=True)

        calibration_feature = calibration_data[tool_list].to_numpy()
        regularization_feature = regularization_data[tool_list].to_numpy()
        calibration_label = calibration_data[args.calibration_label].to_numpy()

        del calibration_data, regularization_data  # Free memory

        chunk_size = args.query_chunk_size
        reader = pd.read_csv(args.query_csv, chunksize=chunk_size, low_memory=False) if chunk_size else [pd.read_csv(args.query_csv, low_memory=False)]
        total_rows = sum(1 for _ in open(args.query_csv)) - 1 if chunk_size else None
        total_chunks = ceil(total_rows / chunk_size) if chunk_size else 1

        header_written = False
        for i, query_data in enumerate(reader):
            logging.info(f"Processing chunk {i+1}/{total_chunks}..." if chunk_size else "Processing full query...")

            query_data[tool_list] = query_data[tool_list].apply(lambda col: pd.to_numeric(col, errors='coerce'))
            missing_ratio = query_data[tool_list].isna().sum() / len(query_data)
            logging.info(f'Missing ratio in chunk {i+1}/{total_chunks}:\n{missing_ratio}' if chunk_size else f'Missing ratio in query data:\n{missing_ratio}')

            if query_data[tool_list].isna().all(axis=1).sum() > 0:
                query_notool = query_data[query_data[tool_list].isna().all(axis=1)]
                query_notool.to_csv(notool_csv, index=False, mode='a', header=not header_written)
                logging.warning(f"{len(query_notool)} query variants missing all tools. Saved to {notool_csv}.")
                query_data = query_data[query_data[tool_list].notna().any(axis=1)].reset_index(drop=True)

            query_feature = query_data[tool_list].to_numpy()
            query_metadata = query_data.drop(columns=tool_list)

            del query_data  # Free memory
            gc.collect()

            # ===== Dynamic Import Based on Device =====
            if args.device == 'GPU':
                try:
                    from .P_KNN_GPU import get_bootstrap_KNN_score_gpu, get_P_KNN_ACMG_score
                except ImportError:
                    from P_KNN_GPU import get_bootstrap_KNN_score_gpu, get_P_KNN_ACMG_score
                test_results_array = get_bootstrap_KNN_score_gpu(
                    calibration_feature, query_feature, regularization_feature,
                    calibration_label, args.p_prior, None,
                    args.n_calibration_in_window, args.frac_regularization_in_window,
                    args.normalization, args.impute, args.mi_scaling,
                    args.n_bootstrap, args.batch_size
                    )
            else:
                try:
                    from .P_KNN_CPU import get_bootstrap_KNN_score, get_P_KNN_ACMG_score
                except ImportError:
                    from P_KNN_CPU import get_bootstrap_KNN_score, get_P_KNN_ACMG_score
                test_results_array = get_bootstrap_KNN_score(
                    calibration_feature, query_feature, regularization_feature,
                    calibration_label, args.p_prior, None,
                    args.n_calibration_in_window, args.frac_regularization_in_window,
                    args.normalization, args.impute, args.mi_scaling,
                    args.n_bootstrap, args.cpu_parallel
                )

            P_KNN_pathogenic, P_KNN_benign, ACMG_scores = get_P_KNN_ACMG_score(
                test_results_array, args.bootstrap_alpha_error, args.p_prior, None)

            query_metadata['P_KNN_posterior_probability(pathogenic)'] = P_KNN_pathogenic
            query_metadata['P_KNN_posterior_probability(benign)'] = P_KNN_benign
            query_metadata['P_KNN_log_likelihood_ratio(evidence_strength)'] = ACMG_scores

            query_metadata.to_csv(output_csv, index=False, mode='a', header=not header_written)
            header_written = True

        logging.info(f"Results written to {output_csv}")
        logging.info("===== P-KNN Finished Successfully =====")
        sys.exit(0)

    except Exception as e:
        if 'chunk_size' in locals() and chunk_size:
            chunk_info = f"chunk {i+1}/{total_chunks}" if 'i' in locals() and 'total_chunks' in locals() else "initialization"
        else:
            chunk_info = "full query"
        logging.error(f"Error occurred during {chunk_info}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()