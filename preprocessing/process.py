import os
from pathlib import Path
from preprocessing.bias_correction import bias_correction
from preprocessing.skull_strip import run_synthstrip
from preprocessing.resize import image_resample
from preprocessing.normalize import normalization
import multiprocessing as mp

os.environ['SITK_SHOW_COMMAND'] = '/Applications/Slicer.app/Contents/MacOS/Slicer'

def process_subject(subject_folder):
    print(f"Processing {subject_folder}")
    #raw = os.path.join("../dataset/training_data1_v2", subject_folder)
    file_list = os.listdir(subject_folder)

    for file in file_list:
        if "seg" in file:
            continue
        bias = os.path.join("../dataset/bias_corrected", file)
        skull = os.path.join("../dataset/skull_stripped", file)
        resize = os.path.join("../dataset/resized", file)
        final = os.path.join("../dataset/final", file)
        image = os.path.join(subject_folder, file)
        bias_correction(image, bias)
        run_synthstrip(bias, skull)
        image_resample(skull, resize)
        normalization(resize, final)
        return final

if __name__ == "__main__":
    directory = '../dataset/training_data1_v2'
    os.makedirs("../dataset/bias_corrected", exist_ok=True)
    os.makedirs("../dataset/skull_stripped", exist_ok=True)
    os.makedirs("../dataset/resized", exist_ok=True)
    os.makedirs("../dataset/final", exist_ok=True)

    os.listdir(directory)
    p = Path(directory)

    subject_folder = "../dataset/training_data1_v2/BraTS-GLI-02314-100"
    process_subject(subject_folder)
    # for item in p.iterdir():
    #     if item.is_dir():
    #         process_subject(item)
    #         break
    #         # input_folders = [f for f in os.listdir(directory)]
    #         # with mp.Pool(processes=4) as pool:
    #         #     results = pool.map(process_subject, input_folders)
    #         print("Pipeline complete.")
