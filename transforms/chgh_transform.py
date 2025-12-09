from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    EnsureChannelFirstd, 
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    Rand3DElasticd
)
import numpy as np

def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=args.b_min, 
                b_max=args.b_max,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=4,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0, 1, 2], 
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.2, 
                rotate_range=(np.pi/12, np.pi/12, np.pi/12),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.1,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.1,
            ),
            RandAdjustContrastd(
                keys=["image"],
                gamma=(0.5, 1.5), 
                prob=0.2,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.2, 
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            ToTensord(keys=["image", "label"])
        ]
    )

def get_inf_transform(keys, args):
    if len(keys) == 2:
        mode = ("bilinear", "nearest")
    elif len(keys) == 3:
        mode = ("bilinear", "nearest", "nearest")
    else:
        mode = ("bilinear")
        
    return Compose(
        [
            LoadImaged(keys=keys),
            
            AddChanneld(keys=keys), 
            
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=mode,
            ),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=args.b_min, 
                b_max=args.b_max,
                clip=True,
                allow_missing_keys=True
            ),
            AddChanneld(keys=keys),
            
            ToTensord(keys=keys)
        ]
    )

def get_label_transform(keys):
    return Compose(
        [
            LoadImaged(keys=keys),
            
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            
            Orientationd(keys=keys, axcodes="RAS"),
            AddChanneld(keys=keys),
            
            ToTensord(keys=keys)
        ]
    )