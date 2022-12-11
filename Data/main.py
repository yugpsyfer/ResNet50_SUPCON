import os
import shutil
import zipfile
import tarfile
import pandas as pd
import json

mini_image_net = "../Inputs/mini_image_net/"
zip_file_name  = "mini_imagenet.zip"
output_path = "mini_image_net_merged"
labels_path = "../Inputs/Labels/wordnet_details.txt"

report_stats = dict()


def extract_files():
    with zipfile.ZipFile(mini_image_net + zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(mini_image_net)

    os.remove(mini_image_net + zip_file_name)
    tar_files = os.listdir(mini_image_net)

    for file in tar_files:
        file_name = file.split(".")
        if file_name[1] != "tar":
            raise NameError

        with tarfile.open(mini_image_net+file) as tar_fp:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_fp, path=mini_image_net)

        os.remove(mini_image_net+file)


def merge_all_splits():
    try:
        os.mkdir(mini_image_net+output_path)
    except:
        print("FILE ALREADY EXISTS.\nI WILL NOW BEGIN TO REMOVE OLD FILES")
        os.rmdir(mini_image_net+output_path)
        print("OLD FILES DELETED.\nMAKING NEW DIRs")
        os.mkdir(mini_image_net+output_path)

    splits = os.listdir(mini_image_net)
    splits.remove(output_path)

    for file in splits:
        images_folder = os.listdir(mini_image_net+file)
        for folder in images_folder:
            images = os.listdir(mini_image_net+file+"/" + folder)
            for image in images:
                src = mini_image_net+file+"/"+folder + "/"+image
                img_name = image[0:9]
                report_stats[img_name] = report_stats.get(img_name) + 1
                img_num = image[9:]
                dst = mini_image_net+output_path+"/"+img_name + "_" + img_num
                shutil.copy(src, dst)


if __name__ == '__main__':
    print("Obtaining files and making one unified dataset")

    labels = pd.read_csv(labels_path,delimiter=" ")
    labels = labels['wdnet_id']
    labels = labels.to_numpy().flatten()

    for label in labels:
        report_stats[label] = 0

    try:
        os.path.isfile(mini_image_net+zip_file_name)
        extract_files()

    except:
        raise FileNotFoundError

    merge_all_splits()
    print("DATA PREPARATION COMPLETED")

    with open('data_stats.json','w') as fp:
        json.dump(report_stats, fp)

