# Adapated from Project-MONAI implementation:  
# https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb  

# Original papers:  
# https://arxiv.org/abs/2111.14791  
# https://arxiv.org/abs/2201.01266  


import monai
import nibabel
import tqdm
import numpy as np
import gc
import os
import shutil
import tempfile
import sys
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

def main(fold):
    ### Set up environment
    torch.cuda.empty_cache()
    gc.collect()

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Transforms
    num_samples = 4

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(
        #     keys=["image", "label"],
        #     pixdim=(1.5, 1.5, 2.0),
        #     mode=("bilinear", "nearest"),
        # ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)

    ### Load datas
    data_dir = "../data/"
    split_json = "dataset_fold_" + str(fold) + ".json"
    print("TRAINING ON FOLD " + str(fold))

    datasets = data_dir + split_json
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    set_track_meta(False)

    ### Set up model
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=17,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)
    weight = torch.load("../model_weights/model_swinvit.pt")
    model.load_from(weights=weight)

    ### Set up loss function and optimizer
    # Best trial: {'learning_rate': 0.0009649251413105664, 'weight_decay': 1.7889241681570157e-05}
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True) # Does this include background?
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0009649251413105664, weight_decay=1.7889241681570157e-05)
    scaler = torch.cuda.amp.GradScaler()

    ### Constants
    max_iterations = 25000
    print("max_iterations: ", max_iterations)
    eval_num = 500
    post_label = AsDiscrete(to_onehot=17)
    post_pred = AsDiscrete(argmax=True, to_onehot=17)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    ### Validation and Training Loops
    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    def train(global_step, train_loader, val_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, disable = True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            # print(x.shape)
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True, disable = True)
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print(
                        "Step: {} Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            global_step, dice_val_best, dice_val)
                    )
                else:
                    print(
                        "Step: {} Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            global_step, dice_val_best, dice_val
                        )
                    )
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                print("current_time: {}".format(current_time))
            global_step += 1
        return global_step, dice_val_best, global_step_best

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, val_loader, dice_val_best, global_step_best)
    
    model_f = "../model_weights/fold_" + str(fold) + "_model.pth"
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    torch.save(model.state_dict(), model_f)

    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    output_f = "../loss_curves/dice_loss_curves_fold_" + str(fold) + ".png"
    plt.savefig(output_f)
    # plt.show()


    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    if directory is None:
        shutil.rmtree(root_dir)


if __name__ == "__main__":
    main(sys.argv[1])