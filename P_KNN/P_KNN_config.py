import os
from huggingface_hub import hf_hub_download

def main():
    ans = input("Download default calibration and regularization dataset (total 200 MB)? (yes/no): ").strip().lower()
    if ans not in ("yes", "y"):
        print("Skipped download.")
        return

    folder = input("Enter folder to save datasets: ").strip()
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    files = ["calibration_data_dbNSFP52.csv", "regularization_data_dbNSFP52.csv"]
    for fname in files:
        print(f"Downloading {fname} ...")
        hf_hub_download(
            repo_id="brandeslab/P-KNN",
            filename=f"dataset4commandline/{fname}",
            repo_type="dataset",
            local_dir=folder,
            local_dir_use_symlinks=False
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

    code = re.sub(
        r"parser\.add_argument\(\s*['\"]--calibration_csv['\"].*?default\s*=\s*['\"][^'\"]*['\"]",
        f"parser.add_argument('--calibration_csv', default=r'{calib_path}'",
        code,
        flags=re.DOTALL
    )
    code = re.sub(
        r"parser\.add_argument\(\s*['\"]--regularization_csv['\"].*?default\s*=\s*['\"][^'\"]*['\"]",
        f"parser.add_argument('--regularization_csv', default=r'{reg_path}'",
        code,
        flags=re.DOTALL
    )

    with open(pknn_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Default paths in P_KNN.py updated.")

if __name__ == "__main__":
    main()
