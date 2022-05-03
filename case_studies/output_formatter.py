import numpy as np

#TODO: add cluster OG

def _format(x):
    return np.array(x).tolist()

def dump_all(X_before_scaling, y,
    inst_names, feat_names, label_names, label_colors,
    P_OG, Ps, Y_OG, Ys, repr_indices,
    emb_of_Ys, cluster_ids):
    return {
        **dump_info(X_before_scaling, y, P_OG, Y_OG),
        **dump_labels(inst_names, feat_names, label_names, label_colors),
        **dump_optfeat(Ps, Ys),
        **dump_repr(repr_indices, emb_of_Ys, cluster_ids),
    }

def dump_info(X_before_scaling, y, P_OG, Y_OG):
    return {
        'X_before_scaling': _format(X_before_scaling),
        'y': _format(y),
        'P_OG': _format(P_OG),
        'Y_OG': _format(Y_OG),
    }

def dump_labels(inst_names, feat_names, label_names, label_colors):
    return {
        'instance_names': _format(inst_names),
        'feat_names': _format(feat_names),
        'label_names': _format(label_names),
        'label_colors': _format(label_colors),
    }

def dump_optfeat(Ps, Ys):
    return {
        'Ps': [_format(each) for each in Ps],
        'Ys': [_format(each) for each in Ys],
    }

def dump_repr(repr_indices, emb_of_Ys, cluster_ids):
    return {
        'repr_indices': _format(repr_indices),
        'emb_of_Ys': _format(emb_of_Ys),
        'cluster_ids': _format(cluster_ids),
    }