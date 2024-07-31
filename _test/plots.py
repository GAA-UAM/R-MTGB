import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error


colors = ["orange", "green", "blue", "red", "purple", "brown", "pink", "gray"]


def training_score(model_mt, model_st, noise_mt, data_type, path):

    title = "Conventional MT" if not noise_mt else "Proposed MT"
    if noise_mt:
        np.savetxt(
            rf"{path}\GB_DataPooling_train_score.csv",
            model_st.train_score_,
            delimiter=",",
        )
    np.savetxt(rf"{path}\{title}_train_score.csv", model_mt.train_score_, delimiter=",")
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[1].plot(model_mt.train_score_, label=f"{title}", color="b")
    axs[0].plot(model_st.train_score_, label="GB (Data Pooling)", color="r")
    axs[0].grid(visible=True, axis="y", color="k", linestyle="-", linewidth=0.5)
    axs[1].grid(visible=True, axis="y", color="k", linestyle="-", linewidth=0.5)
    axs[0].set_xlabel("Boosting epochs")
    axs[0].set_ylabel("Training Score")
    axs[1].set_xlabel("Boosting epochs")
    fig.legend()
    fig.suptitle(f"Training Evolution")
    fig.tight_layout()
    fig.savefig(rf"{path}\{data_type}_{title}_training_score.png", dpi=400)


def training_score_task_specific(model_mt, data_type, path):

    title = "Task_Specific_loss_proposed_MT"
    np.savetxt(rf"{path}\{title}.csv", model_mt.train_score_r, delimiter=",")
    fig, axs = plt.subplots(1, 1, figsize=(7, 3))

    for task in range(model_mt.train_score_r.shape[1]):
        if task > 3:
            axs.plot(
                model_mt.train_score_r[:, task],
                label=f"task {task}",
                color=colors[task],
            )

    axs.set_xlabel("Boosting epochs")
    axs.set_ylabel("Training Score")
    axs.set_xlabel("Boosting epochs")
    fig.legend()
    fig.suptitle(f"Training Evolution - Task Specific")
    fig.tight_layout()
    fig.savefig(
        rf"{path}\{data_type}_{title}_training_score_task_specific.png", dpi=400
    )


def sigmoid_plot(model_mt, title, data_type, path):
    num_sigmoids = model_mt.sigmoid_.shape[1]
    fig, axs = plt.subplots(1, num_sigmoids, figsize=(7 * num_sigmoids // 2, 3))

    if num_sigmoids == 1:
        axs = [axs]

    colors = ["orange", "green", "blue", "red", "purple", "brown", "pink", "gray"]

    for i in range(num_sigmoids):
        axs[i].plot(
            model_mt.sigmoid_[:, i], label=f"sigmoid_{i}", color=colors[i % len(colors)]
        )
        axs[i].set_xlabel("Boosting epochs")
        axs[i].set_ylabel("sigmoid value")
        axs[i].grid(visible=True, axis="y", color="k", linestyle="-", linewidth=0.5)

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncols=(num_sigmoids),
    )
    fig.suptitle(f"sigmoid Evolution")
    fig.tight_layout()
    fig.savefig(rf"{path}\{title}_{data_type}_sigmoid_ev.png", dpi=400)
    np.savetxt(
        rf"{path}\{title}_{data_type}_sigmoid.csv", model_mt.sigmoid_, delimiter=","
    )


def _extract_records(
    score_mt,
    score_data_pooling,
    scores_mt_tasks,
    scores_gb_tasks,
    scores_data_pooling_tasks,
    title,
    noise_mt,
    path,
):

    title_ = "Conventional MT" if not noise_mt else "Proposed MT"
    data = {
        title_: [score_mt],
        "GB (Data Pooling)": [score_data_pooling],
    }

    for i in range(len(scores_mt_tasks)):
        task_label = f"(Task {i})"
        data[f"{title_}_{task_label}"] = [scores_mt_tasks[i]]
    for i in range(len(scores_mt_tasks)):
        task_label = f"(Task {i})"
        data[f"GB - ST {task_label}"] = [scores_gb_tasks[i]]
    for i in range(len(scores_data_pooling_tasks)):
        task_label = f"(Task {i})"
        data[f"GB (Data Pooling) {task_label}"] = [scores_data_pooling_tasks[i]]

    df = pd.DataFrame(data)
    gb_task_columns = [col for col in df.columns if col.startswith("GB - ST (Task")]
    df["GB - ST Average"] = df[gb_task_columns].mean(axis=1)

    csv_filename = rf"{path}\{title_}_{title}.csv"
    df.to_csv(csv_filename, index=True)


def confusion_plot(
    data_type, noise_mt, y_test, pred_st, pred_mt, task_test, preds_st, path
):

    num_tasks = len(np.unique(task_test))

    score_mt = accuracy_score(y_test, pred_mt)
    score_data_pooling = accuracy_score(y_test, pred_st)

    scores_mt_tasks = [
        accuracy_score(y_test[task_test == i], pred_mt[task_test == i])
        for i in range(num_tasks)
    ]
    scores_gb_tasks = [
        accuracy_score(y_test[task_test == i], preds_st[i]) for i in range(num_tasks)
    ]
    scores_data_pooling_tasks = [
        accuracy_score(y_test[task_test == i], pred_st[task_test == i])
        for i in range(num_tasks)
    ]

    title = "Conventional MT" if not noise_mt else "Proposed MT"

    fig, axs = plt.subplots(
        2, num_tasks + 1, figsize=(15, 7), sharex=False, sharey=True
    )

    sns.heatmap(confusion_matrix(y_test, pred_st), annot=True, fmt=".3e", ax=axs[0, 0])
    axs[1, 0].set_title(
        f" Data Pooling \n score: {accuracy_score(y_test, pred_st):.2f}"
    )
    sns.heatmap(confusion_matrix(y_test, pred_mt), annot=True, fmt=".3e", ax=axs[1, 0])
    axs[0, 0].set_title(f"{title} \n score: {accuracy_score(y_test, pred_mt):.2f}")

    for i in range(num_tasks):
        sns.heatmap(
            confusion_matrix(y_test[task_test == i], pred_mt[task_test == i]),
            annot=True,
            fmt=".3e",
            ax=axs[0, i + 1],
        )
        axs[0, i + 1].set_title(
            f"{title} Task {i} \n score: {accuracy_score(y_test[task_test == i], pred_mt[task_test == i]):.2f}"
        )
        sns.heatmap(
            confusion_matrix(y_test[task_test == i], preds_st[i]),
            annot=True,
            fmt=".3e",
            ax=axs[1, i + 1],
        )
        axs[1, i + 1].set_title(
            f"GB Task {i} \n score: {accuracy_score(y_test[task_test == i], preds_st[i]):.2f}"
        )

    fig.suptitle(title)
    fig.tight_layout()  # rect=[0, 0.03, 1, 0.95] # Adjust layout to fit title
    fig.savefig(rf"{path}\{data_type}_{title}_CF.png", dpi=400, bbox_inches="tight")

    _extract_records(
        score_mt,
        score_data_pooling,
        scores_mt_tasks,
        scores_gb_tasks,
        scores_data_pooling_tasks,
        data_type,
        noise_mt,
        path,
    )


def reg_plot(data_type, noise_mt, y_test, pred_st, pred_mt, task_test, preds_st, path):

    num_tasks = len(np.unique(task_test))

    score_mt = mean_squared_error(pred_mt, y_test)
    score_data_pooling = mean_squared_error(pred_st, y_test)

    scores_mt_tasks = [
        mean_squared_error(y_test[task_test == i], pred_mt[task_test == i])
        for i in range(num_tasks)
    ]
    scores_gb_tasks = [
        mean_squared_error(y_test[task_test == i], preds_st[i])
        for i in range(num_tasks)
    ]
    scores_data_pooling_tasks = [
        mean_squared_error(y_test[task_test == i], pred_st[task_test == i])
        for i in range(num_tasks)
    ]
    title = "Conventional MT" if not noise_mt else "Proposed MT"
    fig, axs = plt.subplots(
        2, num_tasks + 1, figsize=(14, 10), sharex=False, sharey=True
    )

    axs[0, 0].scatter(y_test, pred_st)
    axs[0, 0].set_xlabel("True Values")
    axs[0, 0].set_ylabel("Pred Values")
    axs[0, 0].set_title(
        f"Data Pooling \n MSE:{mean_squared_error(y_test, pred_st):.1e}"
    )

    axs[1, 0].scatter(y_test, pred_mt)
    axs[1, 0].set_xlabel("True Values")
    axs[1, 0].set_ylabel("Pred Values")
    axs[1, 0].set_title(f"{title} \n MSE:{mean_squared_error(y_test, pred_mt):.1e}")

    for i in range(num_tasks):
        axs[0, i + 1].scatter(y_test[task_test == i], pred_mt[task_test == i])
        axs[0, i + 1].set_xlabel(f"Task {i} True Values")
        axs[0, i + 1].set_ylabel("Pred Values")
        axs[0, i + 1].set_title(
            f"{title} Task {i} \n MSE:{mean_squared_error(y_test[task_test == i], pred_mt[task_test == i]):.1e}"
        )

        axs[1, i + 1].scatter(y_test[task_test == i], preds_st[i])
        axs[1, i + 1].set_xlabel(f"Task {i} True Values")
        axs[1, i + 1].set_ylabel("Pred Values")
        axs[1, i + 1].set_title(
            f"GB Task {i} \n MSE:{mean_squared_error(y_test[task_test == i], preds_st[i]):.1e}"
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(rf"{path}\{data_type}_{title}_reg.png", dpi=400, bbox_inches="tight")

    _extract_records(
        score_mt,
        score_data_pooling,
        scores_mt_tasks,
        scores_gb_tasks,
        scores_data_pooling_tasks,
        data_type,
        noise_mt,
        path,
    )


def predict_stage_plot(acc_mt, acc_gb, acc_gb_i, noise_mt, data_type, path):
    title = "Conventional MT" if not noise_mt else "Proposed MT"

    colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "tab:blue",
        "tab:red",
        "tab:green",
    ]
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", "-", "--", "-.", ":"]

    np.savetxt(rf"{path}\proposed_MT_staged_score.csv", acc_mt, delimiter=",")
    np.savetxt(rf"{path}\GB_DataPooling_staged_score.csv", acc_gb, delimiter=",")
    np.savetxt(rf"{path}\GB_task_independent_staged_score.csv", acc_gb_i, delimiter=",")

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(
        acc_mt, label=f"{title}_{data_type}", color=colors[0], linestyle=line_styles[1]
    )
    ax.plot(
        acc_gb,
        label=f"GB (Data Pooling)_{data_type}",
        color=colors[1],
        linestyle=line_styles[1],
    )

    for j, i in enumerate(range(acc_gb_i.shape[1])):
        color = colors[j + 2]
        line = line_styles[j + 2]
        ax.plot(
            acc_gb_i[:, i],
            label=f"GB (task independent {i})_{data_type}",
            color=color,
            linestyle=line,
            linewidth=3,
        )

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncols=2)
    fig.suptitle(f"Staged Accuracy of {data_type} problem")
    ax.set_xlabel("Boosting epochs")
    ax.set_ylabel("Score")
    ax.grid()
    fig.tight_layout()  # rect=[0, 0, 1, 0.92]
    fig.savefig(
        rf"{path}\{data_type}_{title}_accuracy.png", dpi=400, bbox_inches="tight"
    )
