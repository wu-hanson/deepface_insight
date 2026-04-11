import os
import shutil
import zipfile

def download_and_create_data_folder(force_rebuild=False):
    project_dir = os.path.expanduser("~/cs7150_project")
    zip_file = os.path.join(project_dir, "real-vs-ai-generated-faces-dataset.zip")
    extract_dir = os.path.join(project_dir, "tmp_extract")
    data_dir = os.path.join(project_dir, "data")

    dataset_dir = os.path.join(data_dir, "dataset")
    real_dir = os.path.join(data_dir, "source", "real", "ffhq")
    gan_dir = os.path.join(data_dir, "source", "fake", "GAN")
    diffusion_dir = os.path.join(data_dir, "source", "fake", "Diffusion")
    faceswap_dir = os.path.join(data_dir, "source", "fake", "FaceSwap")

    # if already built, skip everything
    if (
        not force_rebuild
        and os.path.exists(os.path.join(dataset_dir, "train"))
        and os.path.exists(os.path.join(dataset_dir, "test"))
        and os.path.exists(os.path.join(dataset_dir, "validate"))
        and os.path.exists(real_dir)
        and os.path.exists(gan_dir)
        and os.path.exists(diffusion_dir)
        and os.path.exists(faceswap_dir)
    ):
        print("Data folder already exists. Skipping download and extraction.")
        return

    # cleanup only if rebuilding or structure is incomplete
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    os.makedirs(data_dir, exist_ok=True)

    # download only if zip is missing
    if not os.path.exists(zip_file):
        print("Downloading dataset...")
        os.system(
            f'kaggle datasets download -d philosopher0808/real-vs-ai-generated-faces-dataset -p "{project_dir}"'
        )
    else:
        print("Zip already exists, skipping download.")

    # extract
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(extract_dir)

    # create final folders
    os.makedirs(os.path.join(data_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "source", "real"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "source", "fake", "GAN"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "source", "fake", "Diffusion"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "source", "fake", "FaceSwap"), exist_ok=True)

    # move dataset splits
    shutil.move(
        os.path.join(extract_dir, "dataset", "dataset", "train"),
        os.path.join(data_dir, "dataset", "train")
    )
    shutil.move(
        os.path.join(extract_dir, "dataset", "dataset", "test"),
        os.path.join(data_dir, "dataset", "test")
    )
    shutil.move(
        os.path.join(extract_dir, "dataset", "dataset", "validate"),
        os.path.join(data_dir, "dataset", "validate")
    )

    # move real source
    shutil.move(
        os.path.join(extract_dir, "data_source", "data_source", "ffhq"),
        os.path.join(data_dir, "source", "real", "ffhq")
    )

    # move fake sources into grouped folders
    shutil.move(
        os.path.join(extract_dir, "data_source", "data_source", "fake", "thispersondoesnotexist"),
        os.path.join(data_dir, "source", "fake", "GAN", "thispersondoesnotexist")
    )
    shutil.move(
        os.path.join(extract_dir, "data_source", "data_source", "fake", "sfhq"),
        os.path.join(data_dir, "source", "fake", "GAN", "sfhq")
    )
    shutil.move(
        os.path.join(extract_dir, "data_source", "data_source", "fake", "stable_diffusion"),
        os.path.join(data_dir, "source", "fake", "Diffusion", "stable_diffusion")
    )
    shutil.move(
        os.path.join(extract_dir, "data_source", "data_source", "fake", "faceswap"),
        os.path.join(data_dir, "source", "fake", "FaceSwap", "faceswap")
    )

    # cleanup temp extract only
    shutil.rmtree(extract_dir, ignore_errors=True)

    print("DONE")
    print(f"Data available at: {data_dir}")



def clean_source_folders():
    base = os.path.expanduser("~/cs7150_project/data/source")

    real_src = os.path.join(base, "real", "ffhq")
    real_dst = os.path.join(base, "real")

    gan_dst = os.path.join(base, "fake", "GAN")
    gan_sources = [
        os.path.join(gan_dst, "sfhq"),
        os.path.join(gan_dst, "thispersondoesnotexist"),
    ]

    diff_src = os.path.join(base, "fake", "Diffusion", "stable_diffusion")
    diff_dst = os.path.join(base, "fake", "Diffusion")

    fs_src = os.path.join(base, "fake", "FaceSwap", "faceswap")
    fs_dst = os.path.join(base, "fake", "FaceSwap")

    def move_all_files(src, dst):
        if not os.path.exists(src):
            print(f"Missing: {src}")
            return

        moved = 0
        for root, _, files in os.walk(src):
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst, file)

                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(dst, f"{name}_{i}{ext}")):
                        i += 1
                    dst_file = os.path.join(dst, f"{name}_{i}{ext}")

                shutil.move(src_file, dst_file)
                moved += 1

        print(f"Moved {moved} files from {src} -> {dst}")

    move_all_files(real_src, real_dst)

    for src in gan_sources:
        move_all_files(src, gan_dst)

    move_all_files(diff_src, diff_dst)
    move_all_files(fs_src, fs_dst)

    # remove old empty folders
    for root, dirs, files in os.walk(base, topdown=False):
        if root != base and not os.listdir(root):
            os.rmdir(root)

    print("DONE")