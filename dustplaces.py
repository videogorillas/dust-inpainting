import collections
import os
import random

import PIL
import torch
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

Patch = collections.namedtuple('Patch', ['k', 'cnt', 'x', 'y'])


# scene358/000089,1,128,64,0.42,0.43,0.44
# $key,$feature_count,$x,$y,$mean_flow,$median_flow,$stddev_flow
def csv2patch(line: str) -> Patch:
    k, cnt, x, y = line.split(",")[0:4]
    return Patch(k, int(cnt), int(x), int(y))


class DustPlaces(data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train', masks_csv=None):
        super(DustPlaces, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.rnd = random.Random(42434445)

        allpaths = [img_root + "/" + x for x in os.listdir(img_root)]
        allpaths.sort()
        self.rnd.shuffle(allpaths)

        split80 = int(len(allpaths) * 80 / 100)
        if split == 'train':
            self.paths = allpaths[0:split80]
        else:
            self.paths = allpaths[split80:]

        if masks_csv is not None:
            with open(masks_csv) as f:
                patches = [csv2patch(line) for line in f]
                patches.sort(key=lambda p: p.cnt, reverse=True)
                patches = patches[0:len(patches) // 2]
                self.mask_paths = ['%s/%s_%sx%s_alpha.png' % (mask_root, patch.k, patch.x, patch.y) for patch in
                                   patches]
        else:
            self.mask_paths = [mask_root + "/" + x for x in os.listdir(mask_root)]
            self.mask_paths.sort()
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[self.rnd.randint(0, self.N_mask - 1)])
        mask = PIL.ImageOps.invert(mask)  # masks are inverted in the dataset

        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    t = transforms.ToTensor()
    dp = DustPlaces('/home/zhukov/clients/uk/dustdataset/256.clean', '/home/zhukov/clients/uk/dustdataset/masks', t, t,
                    split='train', masks_csv='/home/zhukov/clients/uk/dustdataset/selected_patches.csv')
    image, mask, gt = zip(*[dp[i] for i in range(1)])
    print(image, mask, gt)
