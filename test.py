import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import model
from data_loader import KodakDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from coder import Coder
import time
from tqdm import tqdm
from data_utils import write_image, crop, cal_psnr
from statistics import plot_RDCurve, sum_dict, get_item, para_num

@torch.no_grad()
def test(test_dataloader, ckptdir_list, outdir, resultdir, model_name='Factorized'):
    # load data
    assert model_name in ["Factorized", "Hyperprior", "JointAutoregressiveHierarchicalPriors", "CheckerboardAutogressive"]

    # 对于Factorized，前两个低码率点N=128，M=192，后两个高码率点N=192，M=320
    # 对于Hyperprior，前两个低码率点N=128，M=192，后两个高码率点N=192，M=320
    # 对于JointAutoregressiveHierarchicalPriors，前两个低码率点N=M=192，后两个高码率点N=192，M=320
    # 对于CheckerboardAutogressive，前两个低码率点N=M=192，后两个高码率点N=192，M=320
    if model_name == 'JointAutoregressiveHierarchicalPriors' or not (torch.cuda.is_available()):
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    outdir_init = os.path.join(outdir, model_name)
    resultdir_init = os.path.join(resultdir, model_name)

    tag = 1
    lmbda = [0.0067, 0.013, 0.025, 0.0483]
    all_result = []
    for i in range(len(ckptdir_list)):
        all_result.append({})

    for i, images in enumerate(tqdm(test_dataloader)):
        # 对测试图像分辨率做一个判断
        images = crop(images)
        x = images.to(device)
        H, W = x.shape[-2:]
        num_pixel = H * W
        one_pic_result = []
        for idx, ckptdir in enumerate(ckptdir_list):

            # load model

            if idx <= 1:
                model_ = getattr(model, args.model_name).to(device)
            else:
                model_ = getattr(model, args.model_name)(N=120, M=320).to(device)
            # 部署熵模型到指定的设备
            model_.entropy_bottleneck.to(args.entropy_device)

            outdir = os.path.join(outdir_init, 'lmbda_'+str(lmbda[idx]))
            resultdir = os.path.join(resultdir_init, 'lmbda_'+str(lmbda[idx]))

            if not os.path.exists(outdir): os.makedirs(outdir)
            if not os.path.exists(resultdir): os.makedirs(resultdir)

            print('=' * 10, tag, '=' * 10)

            filename = os.path.join(outdir, str(i + 1))
            print('output filename:\t', filename)

            # load checkpoints
            assert os.path.exists(ckptdir)
            ckpt = torch.load(ckptdir)
            print('load checkpoint from \t', ckptdir)
            # 解决多卡训练出来的模型checkpoint的key都自动加上了'module'
            model_.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()})
            coder = Coder(model=model_, filename=filename)

            # postfix: lmbda index
            postfix_idx = '_lmbda' + str(lmbda[idx])

            # encode
            start_time = time.time()
            _ = coder.encode(x, postfix=postfix_idx)
            print('Enc Time:\t', round(time.time() - start_time, 3), 's')
            time_enc = round(time.time() - start_time, 3)

            # decode
            start_time = time.time()
            x_dec = coder.decode(postfix=postfix_idx)  # 从文件中读取数据然后解码，解码后对x_dec限定范围[0, 1]
            x_dec = torch.clamp(x_dec, min=0.0, max=1.0)
            print('Dec Time:\t', round(time.time() - start_time, 3), 's')
            time_dec = round(time.time() - start_time, 3)

            # bitrate
            if args.model_name == "Factorized":
                postfix_list = ['_F.bin', '_H.bin']
            elif args.model_name == "CheckerboardAutogressive":
                postfix_list = ['_Fanchor.bin', '_Fnon_anchor.bin', '_Fz.bin', '_H.bin']
            else:
                postfix_list = ['_Fy.bin', '_Fz.bin', '_H.bin']

            bits = np.array([os.path.getsize(filename + postfix_idx + postfix) * 8 \
                             for postfix in postfix_list])

            bpps = (bits / num_pixel).round(3)
            print('num_pixel:', num_pixel)
            print('bits:\t', sum(bits), '\nbpps:\t', sum(bpps).round(3))

            # 重建图片
            start_time = time.time()
            # print(x_dec.detach().cpu().numpy().squeeze().shape)
            write_image(filename + postfix_idx + '_dec.png', x_dec.detach().cpu().numpy().squeeze())
            print('Write image Time:\t', round(time.time() - start_time, 3), 's')
            # 计算失真PSNR
            psnr = cal_psnr(x, x_dec=x_dec)
            print(f'psnr:  {psnr}')

            # save results
            results = {}
            results["num_pixels"] = num_pixel
            results["bits"] = sum(bits).round(3)
            results["bpp"] = sum(bpps).round(3)
            results['psnr'] = psnr
            results["time(enc)"] = time_enc
            results["time(dec)"] = time_dec

            one_pic_result.append(results)

            # 之前版本写入单个文件的操作
            '''all_results = results.copy()
            csv_name = os.path.join(resultdir, filename.split('/')[-1]+postfix_idx+'.csv')

            with open(csv_name, 'w') as f:  
                writer = csv.writer(f)
                for k, v in results.items():
                    writer.writerow([k, v])
            print('Wrile results to: \t', csv_name)'''

        tag = tag + 1

        all_result = sum_dict(all_result, one_pic_result, pic_num=24, ckpt_num=len(ckptdir_list))

    if not os.path.exists("./statistics"): os.makedirs("./statistics")

    df = pd.DataFrame(all_result, index=[0.0067, 0.013, 0.025, 0.0483])
    csv_name = "./statistics/" + model_name + ".csv"
    df.to_csv(csv_name, index_label=True)
    print('Wrile results to: \t', csv_name)

    return all_result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    parser.add_argument("--dataset_path", default='/data1/liubj/kodak')
    parser.add_argument("--test_batch_size", default=1)

    parser.add_argument("--model_name", default='Factorized',help="other implemntation: 'Hyperprior', 'JointAutoregressiveHierarchicalPriors', 'CheckerboardAutogressive'")
    parser.add_argument("--load_model_epoch",default=199,help="加载模型所对应的epoch")
    parser.add_argument("--entropy_device", default='cpu',help="熵模型的device, 'cuda' or 'cpu'")

    args = parser.parse_args()

    ckptdir_list = ['./ckpts/' + args.model_name + '/epoch_' + str(args.load_model_epoch) + '_lmbda_0.0067' + '.pth',
                    './ckpts/' + args.model_name + '/epoch_' + str(args.load_model_epoch) + '_lmbda_0.013' + '.pth',
                    './ckpts/' + args.model_name + '/epoch_' + str(args.load_model_epoch) + '_lmbda_0.025' + '.pth',
                    './ckpts/' + args.model_name + '/epoch_' + str(args.load_model_epoch) + '_lmbda_0.0483' + '.pth']

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_dataset = KodakDataset(args.dataset_path, test_transforms)
    samplerC = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,  # if out of memory, set 'False'
        sampler=samplerC
    )

    all_result = test(test_dataloader, ckptdir_list, args.outdir, args.resultdir, args.model_name)

    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(get_item(all_result, "bpp", ckpt_num=len(ckptdir_list))),
             np.array(get_item(all_result, "psnr", ckpt_num=len(ckptdir_list))), label="D1", marker='x', color='red')

    plt.title(args.model_name)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join("./statistics", args.model_name + '.jpg'))

    plot_RDCurve()


