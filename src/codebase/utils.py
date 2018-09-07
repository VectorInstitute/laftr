def make_dir_if_not_exist(d, remove=False):
    import os
    if remove and os.path.exists(d):  # USE AT YOUR OWN RISK
        import shutil
        shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)

def make_comma_separated_args_dirname_friendly(templates, overrides, transfer_overrides):
    if len(templates) > 0:
        string = " ".join(templates) + "," + ",".join(overrides)
    else:
        string = ",".join(overrides)
    if len(transfer_overrides) > 0:
        string += "/" + ",".join(transfer_overrides)
    dirname = string.replace("--", "").replace(" ", ",").replace(".", "_").replace("=", "-").replace(",", "--")
    return dirname


def get_npz_basename(name, biased, even, **kwargs):
    if name == 'german':
        if biased:
            basename = 'german_valid_inds_biased_valid_inds.npz'
        elif even:
            raise ValueError("even {} dataset not supported".format(name))
        else:
            basename = 'german_valid_inds.npz'
    elif name == 'compas':
        if biased:
            basename = 'compas_vr_valid_inds_biased_valid_inds.npz'
        elif even:
            raise ValueError("even {} dataset not supported".format(name))
        else:
            basename = 'compas_vr_valid_inds.npz'
    elif name == 'health':
        if biased:
            basename = 'health_Y2_charlson_biased_valid_inds.npz'
        elif even:
            basename = 'health_Y2_charlson_equal_valid_and_test.npz'
        else:
            basename = 'health_Y2_charlson.npz'
    elif name == 'adult':
        if biased:
            raise ValueError("biased {} dataset not supported".format(name))
        elif even:
           basename = 'adult_equal_valid_and_test.npz' 
        else:
            basename = 'adult.npz'
    elif name == 'diabetes':
        if biased:
            raise ValueError("biased {} dataset not supported".format(name))
        elif even:
            raise ValueError("even {} dataset not supported".format(name))
        else:
            basename = 'diabetes_edwards_clean.npz'
    return basename
