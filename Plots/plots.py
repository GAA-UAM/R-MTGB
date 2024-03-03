import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

colors = ["r", "g", "b", "k", "y"]


def scatter(df, title, clf):
    fig, ax1 = plt.subplots(1, 3, figsize=(12, 5))
    if clf:
        for class_label in range(len(set(df["target"]))):
            ax1[0].scatter(
                df[(df["target"] == class_label) & (df["task"] == 0)].feature_0,
                df[(df["target"] == class_label) & (df["task"] == 0)].feature_1,
                color=colors[class_label],
            )

        for class_label in range(len(set(df["target"]))):
            ax1[1].scatter(
                df[(df["target"] == class_label) & (df["task"] == 1)].feature_0,
                df[(df["target"] == class_label) & (df["task"] == 1)].feature_1,
                color=colors[class_label],
                label=f"class label: {class_label}",
            )

        for class_label in range(len(set(df["target"]))):
            ax1[2].scatter(
                df[(df["target"] == class_label)].feature_0,
                df[(df["target"] == class_label)].feature_1,
                color=colors[class_label],
                label=f"class label: {class_label}",
            )
    else:
        ax1[0].scatter(
            df[(df["task"] == 0)].feature_0,
            df[(df["task"] == 0)].target,
            color=colors[0],
        )

        ax1[1].scatter(
            df[(df["task"] == 1)].feature_0,
            df[(df["task"] == 1)].target,
            color=colors[0],
        )

        ax1[2].scatter(
            df.feature_0,
            df.target,
            color=colors[0],
        )
    ax1[1].set_title("noised_data")
    ax1[0].set_title("original_data")
    ax1[2].set_title("all data")
    if clf:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncols=3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f"Dataset: {title}")


def boundaries(X, y, clf, axs, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axs.contourf(xx, yy, Z, alpha=0.4)
    axs.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())

    axs.set_title(title)
    axs.grid(True)


def surface(model, x_train, x_test, y_test, title):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    axs = fig.add_subplot(111, projection="3d")

    x1_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    x2_range = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    y_pred_mesh = model.predict(np.column_stack((x1_mesh.ravel(), x2_mesh.ravel())))
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

    axs.scatter(x_test[:, 0], x_test[:, 1], y_test, color="blue", label="True Values")
    axs.plot_surface(
        x1_mesh,
        x2_mesh,
        y_pred_mesh,
        color="red",
        alpha=0.3,
        label="Regression Surface",
    )
    axs.set_xlabel("Feature 1")
    axs.set_ylabel("Feature 2")
    axs.set_zlabel("y")
    axs.set_title("Regression Surface Visualization - " + title)
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=2)


def threeD_surface(model, x_train, x_test, y_test, title):

    x1_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    x2_range = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    y_pred_mesh = model.predict(np.column_stack((x1_mesh.ravel(), x2_mesh.ravel())))
    y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)

    fig = go.Figure()

    # Scatter plot for the true values
    fig.add_trace(
        go.Scatter3d(
            x=x_test[:, 0],
            y=x_test[:, 1],
            z=y_test.flatten(),
            mode="markers",
            marker=dict(color="blue"),
            name="True Values",
        )
    )

    # Surface plot for the regression surface
    fig.add_trace(
        go.Surface(
            x=x1_mesh,
            y=x2_mesh,
            z=y_pred_mesh,
            colorscale="Reds",
            opacity=0.7,
            name="Regression Surface",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            zaxis_title="y",
        ),
        title="Regression Surface Visualization - " + title,
        scene_camera=dict(eye=dict(x=-1.87, y=0.88, z=-0.64)),
    )

    fig.show()
