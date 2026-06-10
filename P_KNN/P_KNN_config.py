import os
import sys
from huggingface_hub import hf_hub_download


def run_test(test_csv, calibration_csv, regularization_csv, output_dir):
    import subprocess
    import numpy as np
    import pandas as pd

    TOLERANCE = 0.1
    RESULT_COL = "P_KNN_log_likelihood_ratio(evidence_strength)"
    LLR_EXPECTED_COL = "LLR_expected"
    TOOL_LIST = "REVEL_score,BayesDel_noAF_score,MutPred2_score,ESM1b_score,AlphaMissense_score"

    os.makedirs(output_dir, exist_ok=True)
    print("Running P-KNN test (this will take ~5 minutes)...")

    cmd = [
        "P_KNN",
        "--query_csv", test_csv,
        "--output_dir", output_dir,
        "--calibration_csv", calibration_csv,
        "--regularization_csv", regularization_csv,
        "--tool_list", TOOL_LIST,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("P-KNN test run failed with error:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    output_csv = os.path.join(output_dir, "P_KNN_Test.csv")
    output = pd.read_csv(output_csv)
    test_data = pd.read_csv(test_csv)

    diff = np.abs(output[RESULT_COL].to_numpy() - test_data[LLR_EXPECTED_COL].to_numpy())
    max_diff = diff.max()
    mean_diff = diff.mean()
    n_fail = (diff > TOLERANCE).sum()

    print(f"  Max absolute difference:  {max_diff:.4f}")
    print(f"  Mean absolute difference: {mean_diff:.4f}")
    print(f"  Variants exceeding tolerance ({TOLERANCE}): {n_fail}/{len(diff)}")

    if n_fail > 0:
        print(f"WARNING: {n_fail} variants exceeded tolerance of {TOLERANCE}. Your results may differ from the reference.")
    else:
        print(f"PASSED: All variants within tolerance of {TOLERANCE}. Installation looks correct!")


def main():
    ans = input("Download default calibration and regularization dataset (total 200 MB)? (yes/no): ").strip().lower()
    if ans not in ("yes", "y"):
        print("Skipped download of default datasets. Please indicate the paths to your own datasets when running P_KNN.")
        return

    folder = input("Enter folder to save datasets: ").strip()
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    version = input("Which version to download? (academic/commercial): ").strip().lower()
    if version == "academic":
        files = ["calibration_data_dbNSFP52.csv", "regularization_data_dbNSFP52.csv"]
    elif version == "commercial":
        print("Note: Commercial version still requires a license from dbNSFP.")
        files = ["calibration_data_dbNSFP52c.csv", "regularization_data_dbNSFP52c.csv"]
    else:
        print(f"Unknown version '{version}'. Please enter 'academic' or 'commercial'.")
        sys.exit(1)

    for fname in files:
        print(f"Downloading {fname} ...")
        hf_hub_download(
            repo_id="brandeslab/P-KNN",
            filename=f"dataset4commandline/{fname}",
            repo_type="dataset",
            local_dir=folder
        )

    test_ans = input("Do you want to download a test file and run a ~5 minute test? (yes/no): ").strip().lower()
    if test_ans in ("yes", "y"):
        test_file = "Test.csv"
        print(f"Downloading {test_file} ...")
        hf_hub_download(
            repo_id="brandeslab/P-KNN",
            filename=f"dataset4commandline/{test_file}",
            repo_type="dataset",
            local_dir=folder
        )

        run_ans = input("Run test now? (yes/no): ").strip().lower()
        if run_ans in ("yes", "y"):
            run_test(
                test_csv=os.path.join(folder, "dataset4commandline", test_file),
                calibration_csv=os.path.join(folder, "dataset4commandline", files[0]),
                regularization_csv=os.path.join(folder, "dataset4commandline", files[1]),
                output_dir=os.path.join(folder, "test_output")
            )

    print("Download complete.")
    update_default_paths(folder, files)


def update_default_paths(folder, files):
    import re
    import P_KNN

    pkg_dir = os.path.dirname(P_KNN.__file__)
    pknn_path = os.path.join(pkg_dir, "P_KNN.py")
    print("Config script path:", __file__)
    print("Target P_KNN.py path:", pknn_path)

    with open(pknn_path, "r", encoding="utf-8") as f:
        code = f.read()

    calib_path = os.path.join(folder, "dataset4commandline", files[0])
    reg_path = os.path.join(folder, "dataset4commandline", files[1])

    def replace_default_line(code, arg_name, new_path):
        pattern = rf"(default=).*"
        replacement = rf"\1r'{new_path}',"
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if f"--{arg_name}" in line and "default=" in line:
                lines[i] = re.sub(pattern, replacement, line)
        return "\n".join(lines)

    code = replace_default_line(code, "calibration_csv", calib_path)
    code = replace_default_line(code, "regularization_csv", reg_path)

    with open(pknn_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Default paths in P_KNN.py updated.")


if __name__ == "__main__":
    main()