import os
import shutil  # High-level file operations  高级的文件操作模块 对os中文件操作的补充


def check_creat(floder):
    if os.path.exists(floder) and os.path.isdir(floder):
        shutil.rmtree(floder)
    try:
        os.mkdir(floder)
    except OSError:
        print("Creation of the testing directory %s failed" % floder)
    else:
        print("Successfully created the testing directory %s " % floder)


def makefloder(path):
    folder_path = path
    root = folder_path
    pre_path = os.path.join(folder_path, 'Process')
    generated_image_path = os.path.join(folder_path, 'Test_Predict')
    # label_threshold_path = os.path.join(folder_path, 'Label_threshold')
    # pre_threshold_path = os.path.join(folder_path, 'Pre_threshold')
    feature_visual_path = os.path.join(folder_path, 'feature_visual')
    checkpoint_path = os.path.join(folder_path, 'CheckPoint')

    # subfolder = [pre_path, generated_image_path, label_threshold_path, pre_threshold_path, checkpoint_path]
    subfolder = [pre_path, generated_image_path, feature_visual_path, checkpoint_path]
    check_creat(root)
    for f in subfolder:
        check_creat(f)
    return 0








