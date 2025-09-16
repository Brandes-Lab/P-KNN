import os
from huggingface_hub import hf_hub_download

def main():
    ans = input("Download default calibration and regularization dataset (total 200 MB)? (yes/no): ").strip().lower()
    if ans not in ("yes", "y"):
        print("Skipped download of default datasets. Please indicate the paths to your own datasets when running P_KNN.")
        return

    folder = input("Enter folder to save datasets: ").strip()
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    version = input("Which version to download? (academic/commercial)").strip()
    if version=="academic":
        files = ["calibration_data_dbNSFP52.csv", "regularization_data_dbNSFP52.csv"]
    elif version=="commercial":
        print("Note: Commercial version still requires a license from dbNSFP.")
        files = ["calibration_data_dbNSFP52c.csv", "regularization_data_dbNSFP52c.csv"]
        
    for fname in files:
        print(f"Downloading {fname} ...")
        hf_hub_download(
            repo_id="brandeslab/P-KNN",
            filename=f"dataset4commandline/{fname}",
            repo_type="dataset",
            local_dir=folder
        )

    test_ans=input("Do you want to download a test file (about 60 KB)? (yes/no): ")
    if test_ans in ("yes", "y"):
        test_file = "Test.csv"
        print(f"Downloading {test_file} ...")
        hf_hub_download(
            repo_id="brandeslab/P-KNN",
            filename=f"dataset4commandline/{test_file}",
            repo_type="dataset",
            local_dir=folder
        )

    print("Download complete.")

    # update P_KNN.py default paths
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