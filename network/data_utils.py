import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import librosa
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from chaos import hxf3dchaos
import time
___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            # _, fn, gender, codec, tag, label = line.strip().split(" ")
            _, fn,  codec, tag, label = line.strip().split(" ")
            file_list.append(fn)
            d_meta[fn] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            fn = line.strip()
            #key = line.strip()
            file_list.append(fn)

        return d_meta, file_list
    else:
        for line in l_meta:
            # _, fn, gender, codec, tag, label = line.strip().split(" ")
            _, fn,  codec, tag, label = line.strip().split(" ")
            file_list.append(fn)
            d_meta[fn] = 1 if label == "bonafide" else 0
        return d_meta, file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x



class Dataset_ASVspoof2024_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""

        self.args = args
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.aug_num = 3
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # X, fs = librosa.load(self.base_dir + 'flac/' + utt_id + '.flac', sr=16000)
        X, fs = librosa.load(str(self.base_dir) + '/' + utt_id + '.wav', sr=16000)
        # print("X.shape:", X.shape)
        X_pad = pad_random(X, self.cut)
        # print("X_pad.shape:", X_pad.shape)
        x_inp = Tensor(X_pad)
        print("x_inp.shape:", x_inp.shape)
        print("x_inp:",x_inp)
        y = self.labels[utt_id]
        return x_inp, y, 0

class Dataset_ASVspoof2024_dev(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X, fs = librosa.load(self.base_dir / f"{utt_id}.wav", sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[utt_id]
        return x_inp, y, utt_id

class Dataset_ASVspoof2024_eval(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X, fs = librosa.load(self.base_dir / f"{utt_id}.wav", sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        # y = self.labels[utt_id]
        return x_inp, 0, utt_id

class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir / + 'flac/' + utt_id + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[utt_id]
        return x_inp, y, utt_id



# --------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr, args, algo):

    # Data process by Convolutive noise (1st algo)
    if algo == 0:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
    elif algo == 1:
        feature = feature


    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
                                     args.maxG, sr)

        # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW,
                                     args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                         args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
                                         args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature



if __name__ == "__main__":
    X, fs = librosa.load(r"E:\xxxx\xxx.flac", sr=16000)
    X3, fs3 = librosa.load(r"E:\xxxx\xxx.flac")
    X2, fs2 = sf.read(r"E:\xxxx\xxx.flac")

    # print(fs, fs3, fs2)

    # is_equal1 = np.array_equal(X, X3)
    # print(is_equal1)
    # is_equal2 = np.array_equal(X, X2)
    # print(is_equal2)
    # print("X.shape", X.shape)
    # print(X)
    # X_pad = pad(X, 64600)
    # X_inp = Tensor(X_pad)
    # print("X_inp.shape", X_inp.shape)
    # print(X_inp)
    # print("X3.shape", X3.shape)
    # print(X3)
    np.set_printoptions(precision=15)
    print("X2.shape", X2.shape)
    print(X2)
    T1 = time.time()
    en_chaos = hxf3dchaos(u0=0.3, v0=0.4, w0=0.5, iterationsvalue=64600, round=1)
    print("en_chaos.shape", en_chaos.shape)
    print(en_chaos)
    # en_chaos_inp = Tensor(en_chaos)
    X2_pad = pad(X2, 64600)
    print(type(X2_pad))
    print("X2_pad.shape", X2_pad.shape)
    print(X2_pad)
    X2_inp = Tensor(X2_pad)
    # X2_inp_enc = X2_inp+1
    # print("X2_inp.shape", X2_inp.shape)torch.Size([64600])
    # print(X2_inp)
    en_X2_pad = X2_pad+en_chaos
    print("en_X2_pad:", en_X2_pad)
    T2 = time.time()
    print('程序运行时间:%s秒' % ((T2 - T1)))
    de_X2_pad = en_X2_pad-en_chaos
    print("de_X2_pad", de_X2_pad)
    de_X2_inp=Tensor(de_X2_pad)
    print("de_X2_inp.shape", de_X2_inp.shape)
    print(de_X2_inp)
    is_equalX2 = np.array_equal(de_X2_inp, X2_inp)
    print(X2_inp)
    print(type(X2_inp))
    print(is_equalX2)



