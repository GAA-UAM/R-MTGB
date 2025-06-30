def _ensemble_pred(
    sigmoid_theta,
    p_meta,
    p_out,
    p_non_out,
    p_task,
):

    return p_meta + ((1 - sigmoid_theta) * p_non_out) + (sigmoid_theta * p_out) + p_task
