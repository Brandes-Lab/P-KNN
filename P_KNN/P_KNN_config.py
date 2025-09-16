import os
from huggingface_hub import hf_hub_download

def main():
    ans = input("Download default calibration and regularization dataset (total 200 MB)? (yes/no): ").strip().lower()
    if ans not in ("yes", "y"):
        print("Skipped download.")
        return

    folder = input("Enter folder to save datasets: ").strip()
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

    # 修改 P_KNN.py 預設路徑
    update_default_paths(folder, files)

def update_default_paths(folder, files):
    import re
    pknn_path = os.path.join(os.path.dirname(__file__), "P_KNN.py")
    with open(pknn_path, "r", encoding="utf-8") as f:
        code = f.read()
    code = re.sub(
        r"parser\.add_argument\('--calibration_csv', default='[^']*'",
        f"parser.add_argument('--calibration_csv', default=r'{os.path.join(folder, files[0])}'",
        code
    )
    code = re.sub(
        r"parser\.add_argument\('--regularization_csv', default='[^']*'",
        f"parser.add_argument('--regularization_csv', default=r'{os.path.join(folder, files[1])}'",
        code
    )
    with open(pknn_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Default paths in P_KNN.py updated.")

if __name__ == "__main__":
    main()