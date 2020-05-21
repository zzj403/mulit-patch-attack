# mulit-patch-attack
First init PatchTransformer by

    PatchTransformer = PatchTransformer()

After init PatchTransformer, PatchApplier, you can use:

    img_patched = PatchTransformer(adv_patch_list, patch_location_list, img_size, img_clean)


to get patched image


    adv_patch_list : a list of patch tensor, size[3,?,?]

    patch_location_list : a list of patch location tensor, size[2]

    img_size : a value, eg:608

    img_clean : clean image, size[1,3,?,?] #!!important!: NOT [3,?,?]
