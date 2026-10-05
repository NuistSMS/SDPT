import argparse
import logging
import os
import random
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F


def train(trainloader, model, ce_loss, dice_loss, optimizer, scheduler, writer, iter_num):
    loss_sum = 0.0
    loss_ce_sum = 0.0
    loss_dice_sum = 0.0
    class_avg_dice_sum = 0.0
    loss_avg = 0.0
    loss_ce_avg = 0.0
    loss_dice_avg = 0.0
    class_avg_dice_avg = 0.0
    iterator = tqdm(total=len(trainloader), ncols=200)
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch[0].squeeze(0), sampled_batch[1].squeeze(0)
        image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
        label_batch[label_batch == 4] = 3

        outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch.long())
        loss_dice, class_wise_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert tensors to scalars before metric aggregation.
        class_avg_dice = sum(d.item() if hasattr(d, 'item') else d for d in class_wise_dice[1:]) / len(class_wise_dice[1:])

        loss_sum += loss.item()
        loss_ce_sum += loss_ce.item()
        loss_dice_sum += loss_dice.item()
        class_avg_dice_sum += class_avg_dice

        loss_avg = loss_sum / (i_batch + 1)
        loss_ce_avg = loss_ce_sum / (i_batch + 1)
        loss_dice_avg = loss_dice_sum / (i_batch + 1)
        class_avg_dice_avg = class_avg_dice_sum / (i_batch + 1)

        iter_num = iter_num + 1
        iterator.update(1)
        iterator.set_postfix({
            'loss': round(loss_avg, 3),
            'loss_ce': round(loss_ce_avg, 3),
            'loss_dice': round(loss_dice_avg, 3),
            'dice': f"{class_avg_dice_avg * 100:.2f}%"
        })
    iterator.close()

    # Record training metrics.
    logging.info(
        f"Iteration: {iter_num} | "
        f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f} | "
        f"Total Loss: {loss_avg:.5f} | "
        f"CE Loss: {loss_ce_avg:.5f} | "
        f"Dice Loss: {loss_dice_avg:.5f} | "
        f"Train Dice: {class_avg_dice_avg:.4f}"
    )
    writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
    writer.add_scalars('info/total_loss', {'train_total_loss': loss_avg}, iter_num)
    writer.add_scalars('info/loss_ce', {'train_loss_ce': loss_ce_avg}, iter_num)
    writer.add_scalars('info/loss_dice', {'train_loss_dice': loss_dice_avg}, iter_num)
    writer.add_scalars('info/dice', {'train_dice': class_avg_dice_avg}, iter_num)

    return model, iter_num


def val(valloader, model, ce_loss, dice_loss, writer, iter_num):
    loss_sum = 0.0
    loss_ce_sum = 0.0
    loss_dice_sum = 0.0
    class_avg_dice_sum = 0.0
    loss_avg = 0.0
    loss_ce_avg = 0.0
    loss_dice_avg = 0.0
    class_avg_dice_avg = 0.0

    iterator = tqdm(total=len(valloader), ncols=200, desc="Validation")
    model.eval()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch[0].squeeze(0), sampled_batch[1].squeeze(0)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            label_batch[label_batch == 4] = 3

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice, class_wise_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # Convert tensors to scalars before metric aggregation.
            class_avg_dice = sum(d.item() if hasattr(d, 'item') else d for d in class_wise_dice[1:]) / len(class_wise_dice[1:])

            loss_sum += loss.item()
            loss_ce_sum += loss_ce.item()
            loss_dice_sum += loss_dice.item()
            class_avg_dice_sum += class_avg_dice

            loss_avg = loss_sum / (i_batch + 1)
            loss_ce_avg = loss_ce_sum / (i_batch + 1)
            loss_dice_avg = loss_dice_sum / (i_batch + 1)
            class_avg_dice_avg = class_avg_dice_sum / (i_batch + 1)

            iterator.update(1)
            iterator.set_postfix({
                'val_loss': round(loss_avg, 3),
                'val_loss_ce': round(loss_ce_avg, 3),
                'val_loss_dice': round(loss_dice_avg, 3),
                'val_dice': f"{class_avg_dice_avg * 100:.2f}%"
            })

    iterator.close()

    writer.add_scalars('info/total_loss', {'val_total_loss': loss_avg}, iter_num)
    writer.add_scalar('info/loss_ce', loss_ce_avg, iter_num)
    writer.add_scalar('info/loss_dice', loss_dice_avg, iter_num)
    writer.add_scalar('info/dice', class_avg_dice_avg, iter_num)

    # Normalize the image before TensorBoard visualization.
    image = image_batch[1, 0:1, :, :]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    writer.add_image('val/Image', image, iter_num)

    outputs_idx = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
    pred_img = outputs_idx[1, ...].float() / 3.0
    writer.add_image('val/Prediction', pred_img, iter_num)

    labs = label_batch[1, ...].unsqueeze(0).float() / 3.0
    writer.add_image('val/GroundTruth', labs, iter_num)

    model.train()
    return loss_avg, class_avg_dice_avg, iter_num


def trainer_brats19(args, model):
    from datasets.dataset_brats19 import Brats19_dataset
    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size_slice = args.batch_size

    db_train = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, batchsize=batch_size_slice,
                               root=args.root_path, mode='train', list_file_name=args.train_list)
    db_val = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, batchsize=batch_size_slice, root=args.root_path,
                             mode='valid', list_file_name=args.val_list)

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    for param in model.parameters():
        param.requires_grad = True
    logging.info("All model parameters are unfrozen for backbone training.")

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # Cosine annealing learning-rate schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epochs, eta_min=1e-6)

    writer = SummaryWriter(args.output_dir + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # Keep the top 20 validation checkpoints.
    best_performance = 0.0
    top_k_models = []
    save_top_k = 20
    best_models_dir = os.path.join(args.output_dir, 'best_models')
    os.makedirs(best_models_dir, exist_ok=True)

    # Initialize the validation-history file.
    val_results_txt = os.path.join(args.output_dir, 'val_results.txt')
    with open(val_results_txt, 'w', encoding='utf-8') as f:
        f.write("Epoch\tVal_Loss\tVal_Dice\n")

    # Track elapsed time for an ETA estimate.
    train_start_time = time.time()

    for epoch_num in range(max_epoch):
        logging.info("Epoch {}/{}".format(epoch_num, max_epoch))

        # Training phase.
        model, iter_num = train(trainloader, model, ce_loss, dice_loss, optimizer, scheduler, writer, iter_num)

        # Validation phase.
        val_avg_loss, val_avg_dice, iter_num = val(valloader, model, ce_loss, dice_loss, writer, iter_num)

        # Append validation metrics.
        with open(val_results_txt, 'a', encoding='utf-8') as f:
            f.write(f"{epoch_num}\t{val_avg_loss:.5f}\t{val_avg_dice:.5f}\n")

        # Update the learning rate.
        scheduler.step()
        logging.info(f"Current LR: {optimizer.param_groups[0]['lr']}")

        # Maintain and save the top-20 validation checkpoints.
        current_model_path = os.path.join(best_models_dir, f'model_epoch_{epoch_num:03d}_dice_{val_avg_dice:.4f}.pth')
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'val_dice': val_avg_dice,
        }, current_model_path)

        top_k_models.append((val_avg_dice, current_model_path))
        top_k_models.sort(key=lambda x: x[0], reverse=True)

        if len(top_k_models) > save_top_k:
            worst_dice, worst_path = top_k_models.pop(-1)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                logging.info(
                    f"Removed Top-K overflow checkpoint: {os.path.basename(worst_path)} with Dice: {worst_dice:.4f}")

        if val_avg_dice > best_performance:
            print(f"Best Dice updated: {best_performance:.4f} -> {val_avg_dice:.4f} (saved to top-20 list)")
            logging.info(f"Best Dice updated: {best_performance:.4f} -> {val_avg_dice:.4f}")
            best_performance = val_avg_dice

        if epoch_num >= max_epoch - 10:
            save_epoch_path = os.path.join(args.output_dir, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_epoch_path)
            logging.info(f"Saved checkpoint for SWA: {save_epoch_path}")

        # Compute and print the estimated remaining time.
        elapsed_time = time.time() - train_start_time
        avg_time_per_epoch = elapsed_time / (epoch_num + 1)
        remaining_epochs = max_epoch - 1 - epoch_num
        eta_seconds = int(avg_time_per_epoch * remaining_epochs)

        eta_string = str(datetime.timedelta(seconds=eta_seconds))
        logging.info(f"Estimated time remaining: {eta_string}")
        print(f"Estimated time remaining: {eta_string}\n" + "-" * 50)

    writer.close()
    return "Training Finished!"
