import os
import shutil
def download_dataset(dir, command, zip):
    if not os.path.exists(f"{dir}"):
        print("Downloading Dataset...")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

        os.system(f"{command} -w -p {dir}")
        import zipfile
        with zipfile.ZipFile(f"{dir}/{zip}","r") as zip_ref:    
            zip_ref.extractall(dir)