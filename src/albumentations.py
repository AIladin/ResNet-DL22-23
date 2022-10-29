from albumentations.pytorch import ToTensorV2
import albumentations as A

train_augmentations = A.Compose(
    [
        A.ToFloat(),
        ToTensorV2(),
        # TODO define augmentations
    ]
)

test_augmentations = A.Compose(
    [
        A.ToFloat(),
        ToTensorV2(),
    ]
)
