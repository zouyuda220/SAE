import argparse
import datetime
import math
import os
import time
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.PointShift_Dataset_UCF_QNRF import PointShift_Dataset_UCF_QNRF
from models.model_unet_encoder_vgg16bn import U_Net_encoder_vgg16bn


def get_arguments():

    parser = argparse.ArgumentParser(description="UNet Network")

    parser.add_argument("--saved_dir", type=str, default='./saved_dir',
                        help="Where to save snapshots of the model.")

    parser.add_argument('--dataset_root', type=str, default='/data/zyd/UCF_QNRF/Train',
                        help='path where the dataset is')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--task_id', required=True, help='task id to use')
    parser.add_argument('--epochs', default=100, help='training epochs')
    parser.add_argument('--batchsize', default=8, help='training epochs')

    return parser.parse_args()


args = get_arguments()
print(args)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)

    if not os.path.exists(os.path.join(args.saved_dir, args.task_id)):
        os.makedirs(os.path.join(args.saved_dir, args.task_id))

    # train_view_path = os.path.join(args.saved_dir, args.task_id, "train_view")
    # train_vieworient_path = os.path.join(args.saved_dir, args.task_id, "train_view_orient")

    # if not os.path.exists(train_view_path):
    #     os.makedirs(train_view_path)

    # if not os.path.exists(train_vieworient_path):
    #     os.makedirs(train_vieworient_path)

    if not os.path.exists(os.path.join(args.saved_dir, args.task_id, 'my.log')):
        with open(os.path.join(args.saved_dir, args.task_id, 'my.log'), 'w'):
            pass

    model = U_Net_encoder_vgg16bn(output_ch=2)
    model.train()
    model.cuda()

    criterion = nn.MSELoss(reduction='none')
    # criterion = nn.MAELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=5e-4, lr=1e-4)

    dataloader_train = DataLoader(PointShift_Dataset_UCF_QNRF(root_path=args.dataset_root), batch_size=args.batchsize, shuffle=True,
                                  pin_memory=True, num_workers=4, drop_last=True)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # if epoch % 10 == 0 or epoch < 10:
        #     os.makedirs(os.path.join(train_view_path, str(epoch)))
        #     os.makedirs(os.path.join(train_vieworient_path, str(epoch)))

        losses = AverageMeter()
        time_start = time.time()

        for i_iter, batch_data in enumerate(dataloader_train):
            print(i_iter, '/', len(dataloader_train))
            img_tensor, target_dict = batch_data
            vis = target_dict['vis'].numpy()
            height, width = img_tensor.shape[2:]
            target_shift = np.zeros((args.batchsize, 2, height, width))
            mask = np.zeros((args.batchsize, 2, height, width))
            points_x = target_dict['points_x'].numpy()
            points_y = target_dict['points_y'].numpy()
            shifted_x_int = target_dict['shifted_x'].numpy()
            shifted_y_int = target_dict['shifted_y'].numpy()

            for i in range(args.batchsize):
                target_shift[i, 0, shifted_y_int[i][vis[i]], shifted_x_int[i][vis[i]]] = points_x[i][vis[i]] - shifted_x_int[i][vis[i]]
                target_shift[i, 1, shifted_y_int[i][vis[i]], shifted_x_int[i][vis[i]]] = points_y[i][vis[i]] - shifted_y_int[i][vis[i]]
                mask[i, :, shifted_y_int[i][vis[i]], shifted_x_int[i][vis[i]]] = 1.

            shift = torch.tensor(target_shift).clone().cuda().float()
            mask = torch.tensor(mask).clone().cuda()

            img_path = target_dict['image_path'][0]

            optimizer.zero_grad()

            pred_flux = model(img_tensor.cuda())
            loss = criterion(pred_flux, shift)

            # total_num = mask.sum()
            loss = (loss * mask).sum()
            losses.update(val=loss.item())
            loss.backward()
            optimizer.step()

            # if (epoch % 10 == 0 or epoch < 10) and i_iter%2==0:

            #     points_x = target_dict['points_x'].numpy()[0]  # int(x) coordinate of original points shape[n]
            #     points_y = target_dict['points_y'].numpy()[0]  # int(y) coordinate of original points shape[n]

            #     height, width = img_tensor.shape[2:]

            #     shift_predict_x = pred_flux.detach().cpu().numpy()[0][0]
            #     shift_predict_y = pred_flux.detach().cpu().numpy()[0][1]
            #     img_view = target_dict['raw_img'].numpy()[0]

            #     image = Image.fromarray(img_view)
            #     draw = ImageDraw.Draw(image)

            #     for index in range(np.count_nonzero(vis[0])):

            #         coord_x_prenoising_int = np.clip(points_x[index], 0, width - 1).astype(np.int32)
            #         coord_y_prenoising_int = np.clip(points_y[index], 0, height - 1).astype(np.int32)

            #         shifted_coord_x_red = np.round(coord_x_prenoising_int + shift_predict_x[coord_y_prenoising_int, coord_x_prenoising_int]).astype(np.int32)
            #         shifted_coord_y_red = np.round(coord_y_prenoising_int + shift_predict_y[coord_y_prenoising_int, coord_x_prenoising_int]).astype(np.int32)
            #         shifted_coord_x_red = np.clip(shifted_coord_x_red, 0, width - 1)
            #         shifted_coord_y_red = np.clip(shifted_coord_y_red, 0, height - 1)

            #         if coord_x_prenoising_int == shifted_coord_x_red and coord_y_prenoising_int == shifted_coord_y_red:
            #             draw.point((coord_x_prenoising_int, coord_y_prenoising_int), fill=(255, 255, 0))
            #         else:
            #             draw.point((coord_x_prenoising_int, coord_y_prenoising_int), fill=(0, 255, 0))  # 原点
            #             draw.point((shifted_coord_x_red, shifted_coord_y_red), fill=(255, 0, 0))  # 加了噪声的原点

            #     image.save(os.path.join(train_view_path, str(epoch), os.path.basename(img_path)).replace('jpg', 'png'))


            #     angle_shift = 180 / math.pi * np.arctan2(shift_predict_y, shift_predict_x)
            #     norm_shift = np.sqrt(shift_predict_x ** 2 + shift_predict_y ** 2)
            #     fig = plt.figure(figsize=(24, 12))

            #     ax1 = fig.add_subplot(121)
            #     ax1.set_title('Norm')
            #     ax1.set_autoscale_on(True)
            #     im1 = ax1.imshow(norm_shift, cmap="jet")
            #     plt.colorbar(im1, shrink=0.5)

            #     ax2 = fig.add_subplot(122)
            #     ax2.set_title('Angle')
            #     ax2.set_autoscale_on(True)
            #     im2 = ax2.imshow(angle_shift, cmap="jet")
            #     plt.colorbar(im2, shrink=0.5)

            #     plt.savefig(os.path.join(train_vieworient_path, str(epoch), os.path.basename(img_path)).replace('jpg', 'png'))
            #     plt.close(fig)

        if epoch % 10 == 0 or epoch < 10:
            torch.save(model.state_dict(), os.path.join(args.saved_dir, args.task_id, str(epoch) + '.pth'))

        train_time = time.time() - time_start
        with open(os.path.join(args.saved_dir, args.task_id, 'my.log'), 'a') as f:
            f.write('Epoch: {epoch}, Loss: {loss}, Train_time: {train_time}'.format(epoch=epoch, loss=losses.avg, train_time=train_time))
            f.write('\n')

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
