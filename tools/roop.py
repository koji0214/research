# %%
import os
import sys
from glob import glob
import pandas as pd
# %%

def roop(in_list,func,ras=False):
    # in_listは条件ごとのリスト
    # Young or Elderly > subject No > RAS condition
    res_list = {}
    for s in in_list:
        sub_list = {}
        for tt in in_list[s]:
            sub = in_list[s][tt]
            if ras:
                res = {}
                for t in sub:
                    res[t] = func(sub[t], s+"_"+tt+"_"+t)
            else:
                sub_t = [sub[t] for t in sub]
                res = func(sub_t,s+"_"+tt)

            sub_list[tt] = res
        res_list[s] = sub_list
    return res_list

# %%
def read_EMG(isyoung = ["Young","Elderly"], ras = ["noRAS1","RAS90","RAS100","RAS110"]):
    
    os.chdir('/Users/koji/Desktop/research/script/script')
    sys.path.append("./../")
    from tools.EMG2 import EMG
    file_path = "../../data/Data_original/"

    isyoung = isyoung
    ras = ras
    emg_list = {}
    for isy in isyoung:
        young_list = {}
        subjects = len(glob(file_path+isy+"/EMG/sub*"))
        print(isy, subjects)

        for sub in range(subjects):
            sub_list = {}
            for _ras in ras:
                data_path = glob(file_path+isy+f"/EMG/sub0{sub+1}/*{_ras}*.mat")[0]
                sub_list[_ras] = EMG(data_path)
            young_list[f"sub{sub+1}"] = sub_list
        emg_list[isy] = young_list

    return emg_list

def to_list(dict):
    res = []
    for isy in dict:
        for sub in dict[isy]:
            mini_res = [dict[isy][sub][ras] for ras in dict[isy][sub]]
            res.append(mini_res)
    return res

def to_DataFrame(list): # 現状のデータ数での変換。本来は辞書型のデータからidxを作成する。
    idx = pd.MultiIndex.from_arrays([
    ['Young', 'Young', 'Young', 'Young','Elderly','Elderly','Elderly','Elderly','Elderly'],
    ['sub1', 'sub2', 'sub3', 'sub4','sub1', 'sub2', 'sub3', 'sub4','sub5']],
    names=['Y_or_E', 'subject'])

    res = pd.DataFrame(list,index=idx, 
                       columns=["noRAS1","RAS90","RAS100","RAS110"])
    return res

def open_dict(dict):
    return to_DataFrame(to_list(dict))
    