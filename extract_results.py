import os
import shutil
from glob import glob
import argparse
import sys
sys.path.append(sys.path[0]+'/Core/GAT')

def extract_Base_result(result_path):
    res = []
    files = sorted(glob(os.path.join(result_path, '*seg.tif')))
    res.extend(files[:])
    return res

def extract_GAT_result(result_path, folder):
    res = []
    res_files = sorted(glob(os.path.join(result_path, folder, '*seg_gat.tif')))
    res.extend(res_files[0:12])
    return res
def extract_slic_result(result_path):
    res = []
    res_files = sorted(glob(os.path.join(result_path, '*segslic.png')))
    res.extend(res_files[0:12])
    return res
'''
# #Base results 33
save_path = 'GAT_0725_33'
if not os.path.exists(save_path):
    os.mkdir(save_path)
os.makedirs(os.path.join(save_path, 'UNet'), exist_ok=True)
src_files = []
for i in range(5):
    path = os.path.join('OCTA_'+str(i+1), 'UNet', 'Result','Test_Predict_36')
    src_files.extend(extract_Base_result(path, i*22,(i+1)*22))
for file in src_files:
    shutil.copyfile(file, os.path.join(save_path, 'UNet', os.path.basename(file)))

'''
def main(args):
    #Base results 66
    save_path = args.name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    src_files = []
    os.makedirs(os.path.join(save_path, 'UNet'), exist_ok=True)
    pre_result_path = 'Core'
    path = os.path.join(pre_result_path, 'UNet', 'Result','Test_Predict')
    src_files.extend(extract_Base_result(path))
    for file in src_files:
        shutil.copyfile(file, os.path.join(save_path, 'UNet', os.path.basename(file)))
    path = os.path.join(pre_result_path, 'GAT')
    results = glob(path+'/result*')
    for result_path in results:
        src_files = []
        folder = result_path.split('/')[-1]
        os.makedirs(os.path.join(save_path, 'GAT', folder), exist_ok=True)
        src_files.extend(extract_GAT_result(path,  folder=folder))
        for file in src_files:
            shutil.copyfile(file, os.path.join(save_path, 'GAT', folder, os.path.basename(file)))
    src_files = []
    path = os.path.join(pre_result_path, 'GAT', 'Buildgraph', 'graph_info66_'+folder[-1], 'images')
    src_files.extend(extract_slic_result(path))
    os.makedirs(os.path.join(save_path, 'slic'), exist_ok=True)
    for file in src_files:
        shutil.copyfile(file, os.path.join(save_path, 'slic', os.path.basename(file)))


#GAT results
# save_path = 'UNet-GAT_1124_66_1'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)

# src_files = []
# for i in range(4):
#     path = os.path.join('OCTA_UNET_'+str(i+1), 'GAT')
#     src_files.extend(extract_GAT_result(path, i*12))

# for file in src_files:
#     shutil.copyfile(file, os.path.join(save_path, os.path.basename(file)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='GAT_final_66',
                        help="name of the folder for saving final result")
    args = parser.parse_args()

    main(args)