
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    print("ai/train.py: from ai.~ import * 로 시도합니다.")
    from ai.model import *
    from ai.dataset import *
    from ai.util import *
except Exception as e:
    print(e)
    print("train.py: from ~ import * 로 시도합니다.")
    from model import *
    from dataset import *
    from util import *

import matplotlib.pyplot as plt

from torchvision import transforms

from tqdm import tqdm  # tqdm 라이브러리 추가

def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr_g = args.lr_g
    lr_d = args.lr_d
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float64)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr_g)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))

    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'png'))

    ## 네트워크 학습하기
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286, 286, nch)),
                                              RandomCrop((ny, nx)),
                                              Normalization(mean=0.5, std=0.5)])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=transform_train,
                                task=task, opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)


        transform_val = transforms.Compose([Resize(shape=(286, 286, nch)),
                                            RandomCrop((ny, nx)),
                                            Normalization(mean=0.5, std=0.5)])

        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'),
                              transform=transform_val,
                              task=task, opts=opts)
        loader_val = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=True, num_workers=8)

        num_data_val = len(dataset_val)
        num_batch_val = np.ceil(num_data_val / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix":
        netG = Pix2Pix(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=2 * nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    # fn_denorm = lambda x, mean, std: np.clip((x * std) + mean, 0, 1)
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                        netG=netG, netD=netD,
                                                        optimG=optimG, optimD=optimD)

        for epoch in tqdm(range(st_epoch + 1, num_epoch + 1), desc='Epoch Progress', unit='epoch'):
            netG.train()
            netD.train()

            loss_G_l1_train = []
            loss_G_gan_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            batch_iterator = tqdm(loader_train, desc=f'Training Epoch {epoch}/{num_epoch}', unit='batch')

            for batch, data in enumerate(batch_iterator, 1):

                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                output = netG(input)

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                real = torch.cat([input, label], dim=1)
                fake = torch.cat([input, output], dim=1)

                pred_real = netD(real)
                pred_fake = netD(fake.detach())

                loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad(netD, False)
                optimG.zero_grad()

                fake = torch.cat([input, output], dim=1)
                pred_fake = netD(fake)

                loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = fn_l1(output, label)
                loss_G = loss_G_gan + wgt * loss_G_l1

                loss_G.backward()
                optimG.step()

                # 손실함수 계산
                loss_G_l1_train += [loss_G_l1.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                # === 추가된 부분: 진행 상황에 손실 함수를 출력하기 위해 set_postfix() 사용. === #
                batch_iterator.set_postfix({
                    "Loss_G_L1": np.mean(loss_G_l1_train),
                    "Loss_G_GAN": np.mean(loss_G_gan_train),
                    "Loss_D_Real": np.mean(loss_D_real_train),
                    "Loss_D_Fake": np.mean(loss_D_fake_train),
                })

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN L1 %.4f | GEN GAN %.4f | "
                      "DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_l1_train), np.mean(loss_G_gan_train),
                       np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))
                
                if batch % 10 == 0:
                    # Tensorboard 저장하기
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                    label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                    input = np.clip(input, a_min=0, a_max=1)
                    label = np.clip(label, a_min=0, a_max=1)
                    output = np.clip(output, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                    writer_train.add_image('input', input, id, dataformats='NHWC')
                    writer_train.add_image('label', label, id, dataformats='NHWC')
                    writer_train.add_image('output', output, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G_l1', np.mean(loss_G_l1_train), epoch)
            writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            with torch.no_grad():
                netG.eval()
                netD.eval()

                loss_G_l1_val = []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = []

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                    output = netG(input)

                    # backward netD
                    # set_requires_grad(netD, True)
                    # optimD.zero_grad()

                    real = torch.cat([input, label], dim=1)
                    fake = torch.cat([input, output], dim=1)

                    pred_real = netD(real)
                    pred_fake = netD(fake.detach())

                    loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    # loss_D.backward()
                    # optimD.step()

                    # backward netG
                    # set_requires_grad(netD, False)
                    # optimG.zero_grad()

                    fake = torch.cat([input, output], dim=1)
                    pred_fake = netD(fake)

                    loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                    loss_G_l1 = fn_l1(output, label)
                    loss_G = loss_G_gan + wgt * loss_G_l1

                    # loss_G.backward()
                    # optimG.step()

                    # 손실함수 계산
                    loss_G_l1_val += [loss_G_l1.item()]
                    loss_G_gan_val += [loss_G_gan.item()]
                    loss_D_real_val += [loss_D_real.item()]
                    loss_D_fake_val += [loss_D_fake.item()]

                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | "
                          "GEN L1 %.4f | GEN GAN %.4f | "
                          "DISC REAL: %.4f | DISC FAKE: %.4f" %
                          (epoch, num_epoch, batch, num_batch_val,
                           np.mean(loss_G_l1_val), np.mean(loss_G_gan_val),
                           np.mean(loss_D_real_val), np.mean(loss_D_fake_val)))
                    
                    if batch % 10 == 0:
                        # Tensorboard 저장하기
                        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                        label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                        output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                        input = np.clip(input, a_min=0, a_max=1)
                        label = np.clip(label, a_min=0, a_max=1)
                        output = np.clip(output, a_min=0, a_max=1)

                        id = num_batch_train * (epoch - 1) + batch

                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                        writer_val.add_image('input', input, id, dataformats='NHWC')
                        writer_val.add_image('label', label, id, dataformats='NHWC')
                        writer_val.add_image('output', output, id, dataformats='NHWC')

                writer_val.add_scalar('loss_G_l1', np.mean(loss_G_l1_val), epoch)
                writer_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
                writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch)
                writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch)

            if epoch % 50 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()
        writer_val.close()

def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr_g = args.lr_g
    lr_d = args.lr_d
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float64)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr_g)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)
    result_dir_test = result_dir  # 직접 result_dir (즉, /static/map)에 저장
    if not os.path.exists(result_dir_test):
        os.makedirs(result_dir_test)
    """
    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))
    """
    ## 네트워크 학습하기
    if mode == "test":
        transform_test = transforms.Compose([Normalization(mean=0.5, std=0.5)])
        # transform_test = transforms.Compose([Resize(shape=(nx, ny, nch)), Normalization(mean=0.5, std=0.5)])

        # test 디렉토리 추가 없이 직접 data_dir 사용
        dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts, mode=mode)
        # dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts, mode=mode)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix":
        netG = Pix2Pix(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=2 * nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)


    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == "test":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG.eval()

            for batch, data in enumerate(loader_test, 1):
                # forward pass
                input = data['input'].to(device).float()
                print(f"Original Input Shape: {input.shape}")
                
                if input.shape[2] > 256 or input.shape[3] > 256:
                    # 🔹 원본 크기 저장
                    input_np = fn_tonumpy(input).squeeze().astype(np.float32)

                    # 🔹 큰 이미지를 패치들로 분할
                    input_patches, positions, img_shape = split_image_into_patches(input_np, patch_size=256)
                    """
                    # 디렉토리가 없으면 생성
                    if not os.path.exists(os.path.join(result_dir_test, 'png')):
                        os.makedirs(os.path.join(result_dir_test, 'png'))
                    """
                    output_patches = []
                    for i, patch in enumerate(input_patches):
                        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device).float()
                        output_patch = netG(patch_tensor)

                        output_patch_np = output_patch.cpu().detach().numpy()
                        output_patch_np = (output_patch_np + 1) / 2  # [-1, 1] → [0, 1]
                        output_patch_np = np.transpose(output_patch_np.squeeze(), (1, 2, 0))  # (3, 256, 256) → (256, 256, 3)

                        output_patches.append(output_patch_np)

                    # 🔹 패치들을 다시 합치기
                    output_patches = np.array(output_patches).astype(np.float32)  # (N, H, W, C)
                    output_image = merge_patches_to_image(output_patches, positions, img_shape)

                    output_image = preprocess_image(output_image)

                    # 🔹 결과 저장
                    output_image = np.clip(output_image, 0, 1)
                    plt.imsave(os.path.join(result_dir_test, 'map.png'), output_image)
                    # plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % batch), output_image)

                else:
                    output = netG(input)

                    # 🔹 배치 크기 확인
                    batch_size_current = input.shape[0]  # 현재 배치 크기 (ex. 1 또는 2 이상)

                    # 🔹 데이터 변환: 배치 차원 제거
                    input_np = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output_np = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                    if batch_size_current == 1:  
                        # 🔹 단일 이미지일 경우 차원 조정
                        input_np = np.squeeze(input_np, axis=0)  # (1, H, W, C) -> (H, W, C)
                        output_np = np.squeeze(output_np, axis=0)

                        id = batch  # 배치 1개일 경우 ID 그대로 사용

                        np.save(os.path.join(result_dir_test, '%04d_input.npy' % id), input_np)
                        np.save(os.path.join(result_dir_test, '%04d_output.npy' % id), output_np)
                        """
                        np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_np)
                        np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_np)
                        """
                        input_np = np.clip(input_np, a_min=0, a_max=1)
                        output_np = np.clip(output_np, a_min=0, a_max=1)

                        output_np = preprocess_image(output_np)
                        
                        plt.imsave(os.path.join(result_dir_test, 'map_input.png'), input_np)
                        plt.imsave(os.path.join(result_dir_test, 'map.png'), output_np)
                        """
                        plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_np)
                        plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_np)
                        """
                    else:  
                        # 🔹 배치 크기가 2 이상일 경우 기존 방식
                        for j in range(batch_size_current):
                            id = batch_size * (batch - 1) + j

                            input_np = input_np[j]
                            output_np = output_np[j]

                            np.save(os.path.join(result_dir_test, '%04d_input.npy' % id), input_np)
                            np.save(os.path.join(result_dir_test, '%04d_output.npy' % id), output_np)
                            """
                            np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_np)
                            np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_np)
                            """
                            input_np = np.clip(input_np, a_min=0, a_max=1)
                            output_np = np.clip(output_np, a_min=0, a_max=1)

                            output_np = preprocess_image(output_np)

                            plt.imsave(os.path.join(result_dir_test, 'map_input.png'), input_np)
                            plt.imsave(os.path.join(result_dir_test, 'map.png'), output_np)
                            """
                            plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_np)
                            plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_np)
                            """

def split_image_into_patches(image, patch_size=256):
    """ 큰 이미지를 patch_size × patch_size 크기로 자르는 함수 """
    h, w, c = image.shape
    patches = []
    positions = []

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    for y in range(0, padded_image.shape[0], patch_size):
        for x in range(0, padded_image.shape[1], patch_size):
            patch = padded_image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches, dtype=np.float32), positions, (h, w, c)

def merge_patches_to_image(patches, positions, original_shape):
    """ 패치들을 다시 원래 크기의 이미지로 합치는 함수 """
    h, w, c = original_shape
    patch_size = patches.shape[1]

    padded_h = ((h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    merged_image = np.zeros((padded_h, padded_w, c), dtype=np.float32)

    for patch, (y, x) in zip(patches, positions):
        merged_image[y:y+patch_size, x:x+patch_size, :] = patch

    return merged_image[:h, :w, :]

def preprocess_image(image):
    # 🔹 0~255 범위 변환
    image = (image * 255).astype(np.uint8)
    
    # 🔹 미디언 필터 (노이즈 제거)
    image = cv2.medianBlur(image, 3)  # 3x3 필터 적용 (노이즈 제거)
    
    # 🔹 모폴로지 연산 (작은 점 제거)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Opening 연산 적용
    
    # 🔹 0~1 범위로 다시 정규화
    image = image.astype(np.float32) / 255.0
    
    return image
