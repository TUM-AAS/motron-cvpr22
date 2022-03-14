import torch


def remove_static_nodes(x: torch.Tensor, dynamic_nodes_idx: list):
    return x[..., dynamic_nodes_idx, :]


def add_static_nodes(q: torch.Tensor, static_nodes_idx, dynamic_nodes_idx):
    q_full_shape = list(q.shape)
    q_full_shape[-2] = q_full_shape[-2] + len(static_nodes_idx)

    q_full = torch.zeros(q_full_shape, device=q.device)
    q_full[..., 0] = 1.
    q_full[..., dynamic_nodes_idx, :] = q
    return q_full


def p_q_mode_output_transform(skeleton):
    def transform(output):
        model_out, y, y_org = output
        p_q = model_out
        q = add_static_nodes(p_q.weighted_mean, skeleton.static_nodes, skeleton.dynamic_nodes)
        return q, y_org  # We ignore the root rotation
    return transform


def position_mode_output_transform(skeleton):
    def transform(output):
        model_out, y, y_org = output
        p_q = model_out
        q = add_static_nodes(p_q.weighted_mean, skeleton.static_nodes, skeleton.dynamic_nodes)
        ns = q.shape[-2]
        pos_y = skeleton(y_org.reshape(-1, ns, 4)).reshape(-1, y.shape[1], ns, 3)
        pos = skeleton(q.reshape(-1, ns, 4)).reshape(-1, y.shape[1], ns, 3)
        return pos, pos_y
    return transform


def default_output_transform(skeleton):
    def transform(output):
        model_out, y, y_org = output
        p_q = model_out
        return p_q, y
    return transform


def motion_variance(motions):
    return motions.var(dim=-2).mean()
