
from __future__ import print_function

import argparse
import os
import time, platform
from skimage import io
from sklearn.metrics import precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt

import cv2
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from losses import *
from model import DexiNed
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result,count_parameters)

IS_LINUX = True if platform.system()=="Linux" else False
def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device,
                    log_interval_vis, tb_writer, args=None):

    # Put model in training mode
    model.train()
    # l_weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1]  # for bdcn ori loss
     # before [0.6,0.6,1.1,1.1,0.4,0.4,1.3] [0.4,0.4,1.1,1.1,0.6,0.6,1.3],[0.4,0.4,1.1,1.1,0.8,0.8,1.3]
    l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3] # New BDCN  loss
    # l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
    #             [0.1, 1.], [0.1, 1.], [0.1, 1.],
    #             [0.01, 4.]]  # for cats loss
    loss_avg =[]
    for batch_id, sample_batched in enumerate(dataloader): # tai du lieu tu dataset_val theo tung anh
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        preds_list = model(images)
        # loss = sum([criterion(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss
        loss = sum([criterion(preds, labels,l_w) for preds, l_w in zip(preds_list,l_weight)]) # bdcn_loss
        # loss = sum([criterion(preds, labels) for preds in preds_list])  #HED loss, rcf_loss
        optimizer.zero_grad() #Dat gradient cua tat ca tham so ve 0
        loss.backward() # dung thuat toan backpropagation tham truyen nguoc
        optimizer.step() # cap nhat cac tham so cua model
        loss_avg.append(loss.item())
        if epoch==0 and (batch_id==100 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss,epoch)

        if batch_id % 5 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), loss.item()))
        if batch_id % log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[2])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), loss.item())

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg


def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None):
    # XXX This is not really validation, but testing
    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images)
            # print('pred shape', preds[0].shape)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names,img_shape=image_shape,
                                     arg=arg)


def test(checkpoint_path, dataloader, model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]

            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)

            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()

    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))

def testPich(checkpoint_path, dataloader, model, device, output_dir, args):
    """dataloader: tai du lieu tu dataset_val theo tung anh
    output_dir: result"""
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            # images2 = images[:, [1, 0, 2], :, :]  #GBR
            images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images)
            preds2 = model(images2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=1,
                        help='Already set the dataset for testing choice: 0 - 8')
    # ----------- test -------0--


    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data] # max 8
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir'] # duong dan toi dataset test
    is_testing =True#  current test -352-SM-NewGT-2AugmenPublish

    # Training settings
    TRAIN_DATA = DATASET_NAMES[0] # BIPED=0, MDBD=6
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX) # duong dan toi dataset train
    train_dir = train_inf['data_dir'] # duong dan toi thu muc dataset train


    # Data parameters
    parser.add_argument('--input_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=test_inf['data_dir'], # duong dan toi input cho validation
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=r'C:\Codes\DexiNed\checkpoints',
                        help='the path to output the results.')
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default=test_inf['test_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--train_list',
                        type=str,
                        default=train_inf['train_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--is_testing',type=bool,
                        default=is_testing,
                        help='Script in testing mode.')
    parser.add_argument('--double_img',
                        type=bool,
                        default=False,
                        help='True: use same 2 imgs changing channels')  # Just for test
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='use previous trained data')  # Just for test
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='10/10_model.pth',# 4 6 7 9 14
                        help='Checkpoint path from which to restore model weights from.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf['img_height'],
                        help='Image height for testing.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=50,
                        help='The number of batches to wait before printing test predictions.')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='Number of training epochs (default: 25).')
    parser.add_argument('--lr',
                        default=1e-4,
                        type=float,
                        help='Initial learning rate.')
    parser.add_argument('--wd',
                        type=float,
                        default=1e-8,
                        metavar='WD',
                        help='weight decay (Good 1e-8) in TF1=0') # 1e-8 -> BIRND/MDBD, 0.0 -> BIPED
    parser.add_argument('--adjust_lr',
                        default=[10,15],
                        type=int,
                        help='Learning rate step size.') #[5,10]BIRND [10,15]BIPED/BRIND
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=16,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--img_width',
                        type=int,
                        default=352,
                        help='Image width for training.') # BIPED 400 BSDS 352/320 MDBD 480
    parser.add_argument('--img_height',
                        type=int,
                        default=352,
                        help='Image height for training.') # BIPED 480 BSDS 352/320
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()
    return args

def calculate_ods_ois(pred_folder, gt_folder, thresholds=np.linspace(0.0, 1.0, 100), distance=10):
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('.png', '.jpg'))])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])
    print(len(pred_files), len(gt_files))
    assert len(pred_files) == len(gt_files)

    all_precisions = []
    all_recalls = []

    for thresh in thresholds:
        tps, fps, fns = 0, 0, 0
        for pred_file, gt_file in zip(pred_files, gt_files):
            pred = cv2.imread(os.path.join(pred_folder, pred_file), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(gt_folder, gt_file), cv2.IMREAD_GRAYSCALE)
            if pred is None or gt is None:
                continue

            pred = pred.astype(np.float32) / 255.0
            pred = (1.0 - pred) > thresh  # nhị phân hóa theo ngưỡng
            gt = (gt > 127).astype(np.uint8)

            # Dilation để hỗ trợ matching khoảng cách
            kernel = np.ones((2 * distance + 1, 2 * distance + 1), np.uint8)
            gt_dilated = cv2.dilate(gt, kernel)

            # TP = pred đúng vị trí với gt (cho phép dịch vài pixel)
            tp = np.sum(pred & (gt_dilated == 1))
            fp = np.sum(pred & (gt_dilated == 0))
            fn = np.sum((gt == 1) & (cv2.dilate(pred.astype(np.uint8), kernel) == 0))

            tps += tp
            fps += fp
            fns += fn

        precision = tps / (tps + fps + 1e-8)
        recall = tps / (tps + fns + 1e-8)
        all_precisions.append(precision)
        all_recalls.append(recall)

    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    f1_scores = 2 * all_precisions * all_recalls / (all_precisions + all_recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx], thresholds[best_idx], np.mean(f1_scores)


def calculate_ap(pred_folder, gt_folder, thresholds=np.linspace(0.0, 1.0, 100), distance=10):
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('.png', '.jpg'))])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])
    assert len(pred_files) == len(gt_files)

    all_precisions = []
    all_recalls = []

    for thresh in thresholds:
        tps, fps, fns = 0, 0, 0
        for pred_file, gt_file in zip(pred_files, gt_files):
            pred = cv2.imread(os.path.join(pred_folder, pred_file), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(gt_folder, gt_file), cv2.IMREAD_GRAYSCALE)
            if pred is None or gt is None:
                continue

            pred = pred.astype(np.float32) / 255.0
            pred = (1.0 - pred) > thresh  # Binary threshold
            gt = (gt > 127).astype(np.uint8)

            kernel = np.ones((2 * distance + 1, 2 * distance + 1), np.uint8)
            gt_dilated = cv2.dilate(gt, kernel)

            tp = np.sum(pred & (gt_dilated == 1))
            fp = np.sum(pred & (gt_dilated == 0))
            fn = np.sum((gt == 1) & (cv2.dilate(pred.astype(np.uint8), kernel) == 0))

            tps += tp
            fps += fp
            fns += fn

        precision = tps / (tps + fps + 1e-8)
        recall = tps / (tps + fns + 1e-8)
        all_precisions.append(precision)
        all_recalls.append(recall)

    # Sort recall for monotonicity
    sorted_indices = np.argsort(all_recalls)
    all_recalls = np.array(all_recalls)[sorted_indices]
    all_precisions = np.array(all_precisions)[sorted_indices]

    # Compute AP using the trapezoidal rule
    ap = np.trapz(all_precisions, all_recalls)
    return ap


def main(args):
    """Main function."""
    tb_writer = None

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")


    training_dir = os.path.join(args.output_dir,args.train_data)     # output_dir la duong dan toi checkpoints,
                                                                     # train_data la DATASET_NAME[0] = BIPED
    os.makedirs(training_dir,exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.train_data, args.checkpoint_data)  # checkpoint_data la 10/10_model.pth
    if args.tensorboard and not args.is_testing:
        # saving Model training settings
        training_notes = ['DexiNed, Xavier Normal Init, LR= ' + str(args.lr) + ' WD= '
                          + str(args.wd) + ' image size = ' + str(args.img_width)
                          + ' adjust LR='+ str(args.adjust_lr) + ' Loss Function= BDCNloss2. '
                          +'Trained on> '+args.train_data+' Tested on> '
                          +args.test_data+' Batch size= '+str(args.batch_size)+' '+str(time.asctime())]
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    # model = nn.DataParallel(model)
    ini_epoch =0
    if not args.is_testing:
        if args.resume:
            ini_epoch=11
            model.load_state_dict(torch.load(checkpoint_path,
                                         map_location=device))
            print('Training restarted from> ',checkpoint_path)
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     mean_bgr=args.mean_pixel_values[0:3] if len(
                                         args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode='train',
                                     arg=args
                                     )
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)

    dataset_val = TestDataset(args.input_val_dir,  # duong dan toi dataset
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list=args.test_list, arg=args
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)  # So tien trinh tai du lieu song song (16)
    # Testing
    if args.is_testing: # Mac dinh la True
        #
        # output_dir = os.path.join(args.res_dir, args.train_data+"2"+ args.test_data)  # result\BIPED2CLASSIC
        # print(f"output_dir: {output_dir}")
        # if args.double_img: # mac dinh la False
        #     # predict twice an image changing channels, then mix those results
        #     testPich(checkpoint_path, dataloader_val, model, device, output_dir, args)
        # else:
        #     test(checkpoint_path, dataloader_val, model, device, output_dir, args)
        #
        # num_param = count_parameters(model)
        # print('-------------------------------------------------------')
        # print('DexiNed # of Parameters:')
        # print(num_param)
        # print('-------------------------------------------------------')

        # Danh gia mang DexiNed tren data BIPED
        # pred_dir = r"C:\Codes\DexiNed\result\BIPED2BIPED\fused"
        # gt_dir = r"C:\Codes\DexiNed\result\BIPED2BIPED\label"

        # Danh gia mang DexiNed tren data BSDS
        pred_dir = r"C:\Codes\DexiNed\result\BIPED2BSDS\fused"
        gt_dir = r"C:\Codes\DexiNed\result\BIPED2BSDS\label"

        # Danh gia mang DexiNed tren data filter
        # pred_dir = r"C:\Codes\DexiNed\result\BIPED2BSDS\fused"
        # gt_dir = r"C:\Codes\DexiNed\result\BIPED2BSDS\label"

        # ods_f1, threshold, ois_f1 = calculate_ods_ois(pred_dir, gt_dir, distance=2)
        ods_f1, threshold, ois_f1 = calculate_ods_ois(pred_dir, gt_dir)
        print(f"ODS F1-score: {ods_f1:.3f}")
        print(f"OIS F1-score: {ois_f1:.3f}")

        # ap_score = calculate_ap(pred_dir, gt_dir, distance=2)
        ap_score = calculate_ap(pred_dir, gt_dir)
        print(f"Average Precision (AP): {ap_score:.3f}")
        return

    criterion = bdcn_loss2 # hed_loss2 #bdcn_loss2

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)

    # Main training loop
    seed=1021
    adjust_lr = args.adjust_lr
    lr2= args.lr
    for epoch in range(ini_epoch,args.epochs):
        if epoch%7==0:

            seed = seed+1000
            np.random.seed(seed) # Dat lai seed cho numpy
            torch.manual_seed(seed) # Dat lai seed cho torch tren CPU
            torch.cuda.manual_seed(seed) # Dat lai seed cho torch tren GPU
            print("------ Random seed applied-------------")
        # Create output directories
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = lr2*0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2

        print('Current learning rate> ', optimizer.param_groups[0]['lr'])
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('DexiNed, # of Parameters:')
    print(num_param)
    print('-------------------------------------------------------')

    output_dir = rf"C:\Codes\DexiNed\checkpoints/BIPED/{args.epochs}"
    model_save_path = os.path.join(output_dir, '/10_model.pth')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
