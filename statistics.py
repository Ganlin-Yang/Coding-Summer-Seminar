import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import model
from PIL import Image
from matplotlib.image import imread


def sum_dict(all_result, one_pic_result, pic_num, ckpt_num):
    for i in range(ckpt_num):
        one_pic=one_pic_result[i]
        all=all_result[i]
        for key, value in one_pic.items():
            if key in all:
                all[key] += value/pic_num
            else:
                all[key] = value/pic_num
    
    return all_result

def get_item(all_result, item, ckpt_num):
    temp = []
    if item == "bpp":
        for i in range(ckpt_num):
            temp.append(all_result[i]["bpp"])
    
    if item == "psnr":
        for i in range(ckpt_num):
            temp.append(all_result[i]["psnr"])
    
    return temp

def file_get(model_name, idx=['0.0067','0.013','0.025','0.0483'], mode="RD-Curve", test=None):
    bpp=[]
    psnr=[]
    enc_time=[]
    dec_time=[]

    if mode == "RD-Curve":
        with open(model_name+'.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] in idx:
                    bpp.append(row[3])
                    psnr.append(row[4])
                    enc_time.append(row[5])
                    dec_time.append(row[6])
    
        return list(map(float, bpp)), list(map(float, psnr)), list(map(float, enc_time)),list(map(float, dec_time))
    
    elif mode == "test-Curve":
        lmbda_idx=['0.0067','0.013','0.025','0.0483']
        idx=list(map(str, idx))
        with open(model_name+'_'+str(lmbda_idx[test])+'.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] in idx:
                    bpp.append(row[4])
                    psnr.append(row[5])
    
        return list(map(float, bpp)), list(map(float, psnr))

def get_JPEG(jpeg_dir="./jpeg"):
    if not os.path.exists(jpeg_dir): os.makedirs(jpeg_dir)
    # 生成JPEG图片
    for i in range(24):
        for j in range(5,105,5):
            jpeg_dir_idx = os.path.join(jpeg_dir,str(int(j/5)))
            if not os.path.exists(jpeg_dir_idx): os.makedirs(jpeg_dir_idx)
            img=Image.open("/data1/liubj/kodak/kodim{:0>2d}.png".format(i+1))
            img.save(jpeg_dir_idx+"/kodim{:0>2d}.jpg".format(i+1), quality=j, subsampling=2)

def psnr(original, compare):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
    if isinstance(compare, str):
        compare = np.array(Image.open(compare).convert('RGB'), dtype=np.float32)
    
    # original 与 compare 范围不同
    original = np.multiply(original, 255.)
    mse = np.mean(np.square(original-compare))
    peak_signal_noise_ratio = np.clip(np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return peak_signal_noise_ratio

def jpeg_test(jpeg_dir="./jpeg"):
    jpeg_bpp = np.array([os.path.getsize(jpeg_dir+"/{:d}/kodim{:0>2d}.jpg".format(j+1, i+1))*8/
        (imread(jpeg_dir+"/{:d}/kodim{:0>2d}.jpg".format(j+1, i+1)).size//3) for i in range(24) for j in range(20)]).reshape(24,20)
    jpeg_bpp = np.mean(jpeg_bpp, axis=0)

    jpeg_psnr = np.array([psnr(np.asarray(imread("/data1/liubj/kodak/kodim{:0>2d}.png".format(i+1))), 
        np.asarray(imread(jpeg_dir+"/{:d}/kodim{:0>2d}.jpg".format(j+1, i+1)))) for i in range(24) for j in range(20)]).reshape(24,20)
    jpeg_psnr = np.mean(jpeg_psnr, axis=0)

    print(jpeg_bpp)
    print(jpeg_psnr)

    return jpeg_bpp, jpeg_psnr

def plot_RDCurve():
    # get these data from CompressAI-master\results\kodak
    JPEG_bpp=[0.32661437988281244,
    0.42312622070312506,
    0.5083855523003472,
    0.5878660413953993,
    0.6601265801323783,
    0.728946261935764,
    0.786026848687066,
    0.8497187296549479,
    0.9060007731119791,
    0.9643800523546006,
    1.0372119479709203,]
    JPEG_psnr=[23.779894921045457,
    26.57723034342358,
    28.042246379237767,
    29.04180914810682,
    29.78473021842612,
    30.378313496658652,
    30.903164761925012,
    31.307827225129476,
    31.70484530103775,
    32.05865273799422,
    32.39929573599792,]

    JPEG2000_bpp=[1.198240492078993,
    0.7982796563042536,
    0.5988286336263021,
    0.47928958468967026,
    0.3986248440212674,
    0.3418426513671875,
    0.2985627916124132,]
    JPEG2000_psnr=[35.680719121855866,
    33.4899612266554,
    32.08974009483378,
    31.07457563807384,
    30.30076938092785,
    29.696116673252778,
    29.191735537599026,]

    compressai_fac_bpp=[0.2878078884548611,
    0.44037882486979174,
    0.6481357150607638,
    0.9669765896267358,]
    compressai_fac_psnr=[29.616915231568353,
    31.27708728897609,
    32.956122820153084,
    35.380922291056244,]

    compressai_hyper_bpp=[0.3198581271701389,
    0.47835625542534727,
    0.6686876085069443,
    0.9388258192274304,]
    compressai_hyper_psnr=[30.972162072759534,
    32.83818257445048,
    34.52626403063645,
    36.74334835426406,]
      
    fig, ax = plt.subplots(figsize=(7, 4))

    bpp, psnr=file_get("Factorized")
    plt.plot(np.array(bpp), np.array(psnr), label="Factorized", marker='x', color='red')

    bpp, psnr=file_get("Hyperprior")
    plt.plot(np.array(bpp), np.array(psnr), label="Hyperprior", marker='x', color='blue')

    # bpp, psnr=file_get("CheckerboardAutogressive")
    # plt.plot(np.array(bpp), np.array(psnr), label="CheckerboardAutogressive", marker='x', color='purple')

    # JPEG
    plt.plot(np.array(JPEG_bpp), np.array(JPEG_psnr), label="JPEG", marker=',', markersize=1, color='green')

    # # another JPEG
    # bpp, psnr = jpeg_test()
    # plt.plot(np.array(bpp)[:14], np.array(psnr)[:14], label="JPEG_group2", marker=',', markersize=1, color='yellow')
    
    # JPEG2000
    plt.plot(np.array(JPEG2000_bpp), np.array(JPEG2000_psnr), label="JPEG2000", marker=',', color='orange')

    # compressai_fac
    plt.plot(np.array(compressai_fac_bpp), np.array(compressai_fac_psnr), label="compressai_fac", marker=',', markersize=1, color='yellow')

    #compressai_hyper
    plt.plot(np.array(compressai_hyper_bpp), np.array(compressai_hyper_psnr), label="compressai_hyper", marker=',', markersize=1, color='purple')

    
    plt.title("R-D Curve")
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig("R-D_Curve.jpg")

    print("plot successfully")

def plot_time():
    fig, ax = plt.subplots(figsize=(7, 4))

    bpp, _, enc_time, dec_time=file_get("factorized")
    plt.plot(np.array(bpp), np.array(enc_time), 'o', label="fact_enc_time", marker='x', color='red')
    plt.plot(np.array(bpp), np.array(dec_time), 'o', label="fact_dec_time", marker='o', color='red')

    bpp, _, enc_time, dec_time=file_get("hyperprior")
    plt.plot(np.array(bpp), np.array(enc_time), 'o', label="hyper_enc_time", marker='x', color='blue')
    plt.plot(np.array(bpp), np.array(dec_time), 'o', label="hyper_dec_time", marker='o', color='blue')

    plt.title("time_contrast")
    plt.xlabel('bpp')
    plt.ylabel('time(s)')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig("time_contrast.jpg")

def plot_test_curve():
    x=[120, 130, 140, 150, 160, 170, 180, 190, 200]
    lmbda=[0.0067, 0.013, 0.025, 0.0483]

    for i in range(len(lmbda)):
        fig, ax1 = plt.subplots(figsize=(7, 4))

        bpp, psnr = file_get("hyperprior", idx=x, mode="test-Curve", test=i)
        print(bpp)
        print(psnr)

        line1, = ax1.plot(x, bpp, color="blue", marker='o', label="bpp")
        ax1.set_xlabel("epoch_num")
        ax1.set_ylabel("bpp")
    
 
        ax2 = ax1.twinx()
        line2, = ax2.plot(x, psnr, color="red", marker='o', label="psnr")
        ax2.set_ylabel("psnr")

        plt.legend(handles=[line1, line2], labels=['bpp','psnr'], loc='best')
        plt.title("test_curve: hyper_lmbda_"+str(lmbda[i]))
        fig.savefig("hyper_lmbda_"+str(lmbda[i])+".jpg")

def para_num():
    model_name = ["Factorized","Hyperprior","JointAutoregressiveHierarchicalPriors","CheckerboardAutogressive"]
    para_dict = {}
    for i in range(len(model_name)):
        _model = getattr(model, model_name[i])()
        para = sum(x.numel() for x in _model.parameters())
        para_dict[model_name[i]] = para
    
    print(para_dict)

    with open('./statistics/para_num.csv', 'w') as f:  
                writer = csv.writer(f)
                for k, v in para_dict.items():
                    writer.writerow([k, v])
    print('Wrile para_dict to: ./statistics/para_num.csv')
