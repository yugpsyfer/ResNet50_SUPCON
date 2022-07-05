import os
import shutil

original = "../Inputs/Labels/wordnet_details.txt"
subset = "../Inputs/Labels/subset_mini_imagenet.txt"
data_dir = "../Inputs/mini_image_net_merged/"
save_data_dir = "../Inputs/mini_imagenet_subset/"

labels = {}
# super_ = []
with open(original, 'r') as fp:
    for i in fp.readlines():
        v = i.split(" ")[0]
        k = i.split(" ")[1].lower()
        if k != 'name':
            labels[k] = v
            # super_.append(i.split(" ")[2])


ids = []

with open(subset, 'r') as fp:
    for i in fp.readlines():
        i = i.split('\n')[0]
        ids.append(labels[i])
        print(labels[i] + " " + i)



# ids = set(ids)
# all_images = os.listdir(data_dir)
#
# for i in all_images:
#     k = i.split("_")[0]
#     if k in ids:
#         dst = save_data_dir + i
#         src = data_dir + i
#         shutil.copy(src, dst)