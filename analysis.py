#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:30:12 2023

@author: bmhi
"""
# %%
import os
os.getcwd()
# os.chdir('')
# %%
# 0123
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from EMG import EMG
from load import load
# %%
# 1周期の長さを比較して左右均等か確認する
eeg_path = '../data/Data_original/Young/EEG/Nagashima_noRAS1_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_noRAS1.mat'
eeg, emg = load(eeg_path, emg_path)
emg_100 = emg
print(emg.mean_length)
emg.culc_cadence("Lt")
print(emg.mean_length)

# %%
from EEG import add_cadence
eeg_path = '../data/Data_original/Young/EEG/Nagashima_RAS110_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_RAS110.mat'
eeg, emg = load(eeg_path, emg_path)
eeg = add_cadence(eeg, emg_100, 110)

# %%
eeg.crop(10,)
events = mne.find_events(eeg, stim_channel = "cadence")
event_dict = {"stimu":1}
epochs = mne.Epochs(eeg, events, event_id=event_dict, tmin=-0.5, tmax=1.5,preload=True)
epochs.plot_image(picks=["Cz"])
plt.show()

evoked = epochs.average()
evoked.plot_joint(picks="eeg")
plt.show()
# %%
eeg.plot()

# %%
# 0124
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from EMG import EMG
from load import load

from EEG import add_RAS
eeg_path = '../data/Data_original/Young/EEG/Nagashima_RAS110_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_RAS110.mat'
eeg, emg = load(eeg_path,emg_path)
eeg = add_RAS(eeg, 125)
eeg.crop(10,)


# %%
tmin = -0.2
tmax = 60/125 + tmin
events = mne.find_events(eeg, stim_channel = "cadence")
event_dict = {"stimu":1}
epochs = mne.Epochs(eeg, events, event_id=event_dict, tmin=tmin, tmax=tmax,preload=True)
epochs.plot_image()

evoked = epochs.average()
evoked.plot_joint(picks="eeg")
plt.show()
# %%
# 0125
# EEG前処理からRASとの関連まで
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from EMG import EMG
from load import load

from EEG import add_RAS
eeg_path = '../data/Data_original/Young/EEG/Nagashima_RAS110_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_RAS110.mat'
bpm = 125
eeg, emg = load(eeg_path,emg_path)
eeg = add_RAS(eeg, bpm)    # EEGのイベントデータにRASのタイミングデータを追加する
eeg.crop(10,)    # 開始初期はアーチファクトが大きいので除外する

# %%
# preprocessing
raw = eeg.copy()
raw.filter(l_freq = 0.2, h_freq=100, method="iir")  # 0.2~100Hzでバンドパスフィルタリング
raw_downsampled = raw.copy().resample(sfreq = 200)  # downsampling
events = mne.find_events(raw_downsampled, stim_channel="Heel")
mne.Epochs(raw_downsampled, events=events,tmax = 1)['2'].average().plot_joint()
sensors = raw_downsampled.plot_sensors(show_names=True)
# %%
# raw_downsampled.info["bads"].extend([])  # 今回のデータで不良電極を認めないので、このステップを飛ばして解析を行う
raw_downsampled.set_eeg_reference(ref_channels="average", projection=True)  # rereference

# from asrpy import ASR   # なんでかよくわからんけどASRは内部関数でエラーが出ているので、実行できなかったのでスキップ
                          # 正直してもいいのかよくわからないから除外してもいい気がしている。明らかに大きいえ振幅がどれだけ混入しているかかな。
# asr = ASR(sfreq=raw_downsampled.info["sfreq"], cutoff = 20)
# asr.fit(raw_downsampled)
# raw_asr = asr.transform(raw_downsampled)

raw_asr = raw_downsampled.copy()
ica = mne.preprocessing.ICA(n_components=15, max_iter="auto",random_state=10)  # ICAでどの成分がノイズ成分か判断する
ica.fit(raw_asr)
ica.plot_sources(raw_asr)
ica.plot_components()
# %%
events = mne.find_events(raw_asr, stim_channel="Heel")
epochs_heel = mne.Epochs(raw_asr, events, tmin = -0.2, tmax = 1)
events = mne.find_events(raw_asr, stim_channel="cadence")
epochs_ras = mne.Epochs(raw_asr, events, tmin = -0.2, tmax = 0.5)["1"]
epochs_heel.average().plot_joint(picks="eeg")
epochs_ras.average().plot_joint(picks="eeg")

# %%
# ica.plot_properties(epochs_heel)  # なぜかわからんけどカーネルを再起動した時は一度このスクリプトを切り替えると実行できる。
ica.plot_properties(epochs_heel, picks=[0,1,2,3,4])
# %%
ica.plot_properties(epochs_heel, picks=[6,9,8,14])
# %%
ica.plot_properties(epochs_heel, picks=[6,9,8,14])
# %%
ica.exclude = [0,1,4]  # この被験者の場合は0、1が眼球、4が歩行アーチファクト
raw_ica = raw_asr.copy()
ica.apply(raw_ica)
raw_ica.plot()
# %%
events = mne.find_events(raw_ica, stim_channel="cadence")
epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax = 1)
epochs.average().plot_joint(picks="eeg")
# %%
events = mne.find_events(raw_ica, stim_channel="Heel")
epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax = 1)
epochs.average().plot_joint(picks="eeg")

"""
この被験者ではヒールコンタクトが正確に同定できているところとできていないところがあり、アルゴリズムを変更しなければならない可能性がある。
また、RASのタイミングでエポッキングしてみたが、電極レベルのERPからは特定のタイミングで反応しているような神経活動は確認できなかった。
しかし、この被験者ではRASとヒールコンタクトのタイミングがほぼ一致していたため、ICAで除外してしまった可能性もある。
IC2が後半に行くにかけて成分の割合が高くなっていた。これはなんでか？
"""
"""
追記、
ヒールコンタクト同定のアルゴリズムはフットセンサーの最大振幅の1/2の信号以下を0にし、それ以外のところの一番初めをヒールコンタクトにしていたが、
最大振幅の3/4をヒールコンタクトとすることにした。こちらはヒールコンタクトのうち左のものを正確に捉えることができていない。
2/3にしたところ、左ヒールコンタクトは今までより格段に精度良く同定できているが、まだ抜けているエポックが存在している
sensorデータから色々確認したところ、2段階の山があるところがあり、これが影響していた。
遊脚中に踵が設置していることはなかったので、今回は1/3以下を0にする処理で行ったところ問題が解決した。
"""
"""
この被験者は136秒以降がややRASとヒールコンタクトのタイミングがずれているので、これを確認してみる？
"""
# %%
eeg_path = '../data/Data_original/Young/EEG/Nagashima_RAS90_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_RAS90.mat'
bpm = 102
eeg, emg = load(eeg_path,emg_path)
eeg = add_RAS(eeg, bpm)    # EEGのイベントデータにRASのタイミングデータを追加する
eeg.crop(10,)    # 開始初期はアーチファクトが大きいので除外する
# 左足が連チャンするタイミングが1回だけ19秒くらいにあるけどまあいいか
# %%
# preprocessing
raw = eeg.copy()
raw.filter(l_freq = 0.2, h_freq=100, method="iir")  # 0.2~100Hzでバンドパスフィルタリング
raw_downsampled = raw.copy().resample(sfreq = 200)  # downsampling
events = mne.find_events(raw_downsampled, stim_channel="Heel")
mne.Epochs(raw_downsampled, events=events,tmax = 1)['2'].average().plot_joint()
# sensors = raw_downsampled.plot_sensors(show_names=True)
# %%
# raw_downsampled.info["bads"].extend([])  # 今回のデータで不良電極を認めないので、このステップを飛ばして解析を行う
raw_downsampled.set_eeg_reference(ref_channels="average", projection=True)  # rereference

# from asrpy import ASR   # なんでかよくわからんけどASRは内部関数でエラーが出ているので、実行できなかったのでスキップ
                          # 正直してもいいのかよくわからないから除外してもいい気がしている。明らかに大きいえ振幅がどれだけ混入しているかかな。
# asr = ASR(sfreq=raw_downsampled.info["sfreq"], cutoff = 20)
# asr.fit(raw_downsampled)
# raw_asr = asr.transform(raw_downsampled)

raw_asr = raw_downsampled.copy()
ica = mne.preprocessing.ICA(n_components=15, max_iter="auto",random_state=10)  # ICAでどの成分がノイズ成分か判断する
ica.fit(raw_asr)
ica.plot_sources(raw_asr)
ica.plot_components()
# %%
events = mne.find_events(raw_asr, stim_channel="Heel")
epochs_heel = mne.Epochs(raw_asr, events, tmin = -0.2, tmax = 1)
events = mne.find_events(raw_asr, stim_channel="cadence")
epochs_ras = mne.Epochs(raw_asr, events, tmin = -0.2, tmax = 0.5)["1"]
epochs_heel.average().plot_joint(picks="eeg")
epochs_ras.average().plot_joint(picks="eeg")

# %%
ica.plot_properties(epochs_heel)  # なぜかわからんけどカーネルを再起動した時は一度このスクリプトを切り替えると実行できる。
# ica.plot_properties(epochs_heel, picks=[0,1,2,3,4])
# %%
ica.plot_properties(epochs_heel, picks=[5,7,12,13])
# %%
ica.plot_properties(raw_asr, picks=[5,7,12,13])
# %%
ica.exclude = [0,2,1]  # この被験者の場合は0、2が眼球、1が歩行アーチファクト、4もあやしい
raw_ica = raw_asr.copy()
ica.apply(raw_ica)
raw_ica.plot()
# %%
events = mne.find_events(raw_ica, stim_channel="cadence")
epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax = 1)
epochs.average().plot_joint(picks="eeg")
# %%
events = mne.find_events(raw_ica, stim_channel="Heel")
epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax = 1)
epochs.average().plot_joint(picks="eeg")

# %%
eeg_path = '../data/Data_original/Young/EEG/Nagashima_RAS100_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_RAS100.mat'
bpm = 114
eeg, emg = load(eeg_path,emg_path)
eeg = add_RAS(eeg, bpm)    # EEGのイベントデータにRASのタイミングデータを追加する
eeg.crop(10,)    # 開始初期はアーチファクトが大きいので除外する
# 左足が連チャンするタイミングが1回だけ19秒くらいにあるけどまあいいか
# %%
# preprocessing
raw = eeg.copy()
raw.filter(l_freq = 0.2, h_freq=100, method="iir")  # 0.2~100Hzでバンドパスフィルタリング
raw_downsampled = raw.copy().resample(sfreq = 200)  # downsampling
events = mne.find_events(raw_downsampled, stim_channel="Heel")
mne.Epochs(raw_downsampled, events=events,tmax = 1)['2'].average().plot_joint()
# sensors = raw_downsampled.plot_sensors(show_names=True)
# %%
# raw_downsampled.info["bads"].extend([])  # 今回のデータで不良電極を認めないので、このステップを飛ばして解析を行う
raw_downsampled.set_eeg_reference(ref_channels="average", projection=True)  # rereference

# from asrpy import ASR   # なんでかよくわからんけどASRは内部関数でエラーが出ているので、実行できなかったのでスキップ
                          # 正直してもいいのかよくわからないから除外してもいい気がしている。明らかに大きいえ振幅がどれだけ混入しているかかな。
# asr = ASR(sfreq=raw_downsampled.info["sfreq"], cutoff = 20)
# asr.fit(raw_downsampled)
# raw_asr = asr.transform(raw_downsampled)

raw_asr = raw_downsampled.copy()
ica = mne.preprocessing.ICA(n_components=15, max_iter="auto",random_state=10)  # ICAでどの成分がノイズ成分か判断する
ica.fit(raw_asr)
ica.plot_sources(raw_asr)
ica.plot_components()
# %%
events = mne.find_events(raw_asr, stim_channel="Heel")
epochs_heel = mne.Epochs(raw_asr, events, tmin = -0.2, tmax = 1)
events = mne.find_events(raw_asr, stim_channel="cadence")
epochs_ras = mne.Epochs(raw_asr, events, tmin = -0.2, tmax = 0.5)["1"]
epochs_heel.average().plot_joint(picks="eeg")
epochs_ras.average().plot_joint(picks="eeg")

# %%
# ica.plot_properties(epochs_heel)  # なぜかわからんけどカーネルを再起動した時は一度このスクリプトを切り替えると実行できる。
ica.plot_properties(epochs_heel, picks=[0,1,2,3,4])
# %%
ica.plot_properties(epochs_heel, picks=[5,6,7,14])
# %%
ica.plot_properties(raw_asr, picks=[5,7,12,13])
# %%
ica.exclude = [0,2,1]  # この被験者の場合は0、2が眼球、1が歩行アーチファクト、3、14が怪しい
raw_ica = raw_asr.copy()
ica.apply(raw_ica)
raw_ica.plot()
# %%
events = mne.find_events(raw_ica, stim_channel="cadence")
epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax = 1)
epochs.average().plot_joint(picks="eeg")
# %%
events = mne.find_events(raw_ica, stim_channel="Heel")
epochs = mne.Epochs(raw_ica, events, tmin=-0.2, tmax = 1)
epochs.average().plot_joint(picks="eeg")
