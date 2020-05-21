# mulit-patch-attack
The useful part is PatchTransformer and PatchApplier

After init PatchTransformer, PatchApplier, you can use:

adv_batch = PatchTransformer(adv_patch_list, patch_location_list, img_size, do_rotate=False, rand_loc=False)

to get mask Tensor adv_batch

and use PatchApplier to get img_patched

img_patched = PatchApplier(img_batch, adv_batch)


adv_patch_list : a list of patch tensor, size[3,?,?]

patch_location_list : a list of patch location tensor, size[2]

img_size: a value,eg:608

do_rotate: useless,please don't use

rand_loc: useless,please don't use

img_batch : batch of img, size[batch_size,3,?,?]
