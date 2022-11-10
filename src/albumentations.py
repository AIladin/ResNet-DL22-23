from albumentations.pytorch import ToTensorV2
import albumentations as A

train_augmentations = A.Compose(
    [
        A.ToFloat(),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.2, 0.2),
            rotate=(-15, 15),
            shear=(-5, 5),
            keep_ratio=True,
            p=0.5,
        ),
        ToTensorV2(),
    ]
)

test_augmentations = A.Compose(
    [
        A.ToFloat(),
        ToTensorV2(),
    ]
)
