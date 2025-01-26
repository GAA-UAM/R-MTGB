def _ensemble_pred(sigmoid_theta, common_prediction, tasks_prediction):
    return (1 - sigmoid_theta) * common_prediction + tasks_prediction
