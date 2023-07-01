from albumentations import Compose, CoarseDropout, pytorch, Normalize, HorizontalFlip, ShiftScaleRotate, ToGray, Flip, \
    PadIfNeeded


class AlbumentaionsTransforms(object):

    def gettraintransforms(self, mean, std):
        # Train Phase transformations

        albumentations_transform = Compose([
            HorizontalFlip(always_apply=True),
            #PadIfNeeded(32, 32, always_apply=True, p=0.5),
            # Flip(p=0.5,always_apply=True),
            CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                          fill_value=mean, mask_fill_value=None, always_apply=True, p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                             value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None,
                             rotate_method='largest_box', always_apply=True, p=0.5),
            # ToGray(always_apply=False, p=0.5),
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
        ])

        return albumentations_transform;

    def gettesttransforms(self, mean, std):
        # Test Phase transformations
        return Compose([
            Normalize(mean=mean, std=std, always_apply=True),
            pytorch.ToTensorV2(always_apply=True),
        ])
