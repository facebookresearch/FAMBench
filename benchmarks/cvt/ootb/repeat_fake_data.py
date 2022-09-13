import os
import sys

def repeat_files(folder, times):
    classes = os.listdir(folder)
    ori_classes = [c for c in classes if c.startswith('n')]
    ORI_NUM_CLASSES = len(ori_classes)

    assert len(classes) % ORI_NUM_CLASSES == 0
    cur_repeat = len(classes) // ORI_NUM_CLASSES

    # If cur_repeat is smaller, add more repeats
    for i in range(cur_repeat, times):
        for c in ori_classes:
            new_class = f"{i}_{c}"
            os.system(f"cp {folder}/{c} {folder}/{new_class} -r")

    # If times is smaller, remove extra repeats
    for i in range(times, cur_repeat):
        for c in ori_classes:
            class_to_del = f"{i}_{c}"
            os.system(f"rm {folder}/{class_to_del} -r")

if __name__ == "__main__":
    data_path = "DATASET/imagenet"
    sub_folders = ["train", "val"]

    l = len(sys.argv)

    assert l <= 2

    times = 5 if l == 1 else int(sys.argv[1])

    for f in sub_folders:
        folder = os.path.join(data_path, f)
        repeat_files(folder, times)
