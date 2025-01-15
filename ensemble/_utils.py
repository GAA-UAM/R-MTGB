def _ensemble_pred(sigma_theta, common_prediction, tasks_prediction):
    return (1 - sigma_theta) * common_prediction + tasks_prediction
