from functools import partial

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from phdtruel.errors import SupportedErrorsNames


def plot_error_map(
    got_errors_field,
    err_name: SupportedErrorsNames = "l_2",
    add_text=None,
    add_colorbar=None,
    **kwargs,
):
    shape = got_errors_field.shape
    has_many_values = sum(shape) >= 5 * 5
    if add_text is None:
        add_text = False if has_many_values else True
    if add_colorbar is None:
        add_colorbar = True if has_many_values else False

    s_0_values = np.linspace(0, 1, shape[0])
    s_1_values = np.linspace(0, 1, shape[1])

    fig, ax = plt.subplots()
    i = ax.imshow(got_errors_field, interpolation="none")
    ax.set(xlabel="s along 0", ylabel="s along 1")

    def formatter(x, pos, normalizer):
        return f"{x / normalizer:.2f}"

    ax.xaxis.set_major_formatter(partial(formatter, normalizer=shape[0] - 1))
    ax.yaxis.set_major_formatter(partial(formatter, normalizer=shape[1] - 1))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # ax.set_minor_locator(ticker.MultipleLocator(0.05))

    # if not has_many_values:
    #     ax.set_xticks(range(shape[1]), s_1_values)
    #     ax.set_yticks(range(shape[0]), s_0_values)
    ax.set_title(f"Error map ${err_name}$")

    if add_colorbar:
        fig.colorbar(i, ax=ax)

    if add_text:
        for i in range(len(s_0_values)):
            for j in range(len(s_1_values)):
                v = got_errors_field[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="k")
    return fig, ax


def insert_line_breaks(text, max_length):
    """
    Insert line breaks into a string without splitting words.

    Args:
        text (str): The input text.
        max_length (int): The maximum length of each line.

    Returns:
        str: The formatted text with line breaks.
    """
    words = text.split()
    result = ""
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            result += "\n"
            current_length = 0
        result += word + " "
        current_length += len(word) + 1

    return result.strip()


def plot_errors(
    df: pd.DataFrame,
    test_case: str,
    error_norm: SupportedErrorsNames,
    *,
    apply_colors: dict[str, str | list[str]] | None = None,
    date: str | None = None,
    semilogy: bool = False,
    legend: dict[str, str] | None = None,
):
    """
    Plot errors as a bar plot for a specific test case.

    Parameters:
    - df: Pandas DataFrame containing the error results and metadata
    - test_case: str, the test case to plot errors for
    - error_norm: str or list of str, the error norms to plot (e.g. 'l_1', 'l_2', etc.)
    """
    # Filter the dataframe to only include the desired test case
    # Filter the dataframe to only include the desired test case
    filtered_df = df[df["test_case"] == test_case]

    # If a date is specified, filter the dataframe to only include results from that date
    # if date is not None and date != "last":
    #     filtered_df = filtered_df[filtered_df["date"] == date]
    #
    # # Otherwise, select the most recent result for each method
    # else:
    #     filtered_df = filtered_df.sort_values("date", ascending=False).drop_duplicates(
    #         "method"
    #     )

    date = filtered_df["date"].iloc[0]

    methods = filtered_df["method"]
    methods = [insert_line_breaks(s, 40) for s in methods]
    errors = filtered_df[error_norm]

    colors = ["blue"] * len(errors)
    if apply_colors is None:
        apply_colors = {}
    for i, method in enumerate(methods):
        for color, texts in apply_colors.items():
            if not isinstance(texts, list):
                texts = [texts]
            for text in texts:
                if method.find(text) != -1:
                    colors[i] = color

    # Create a figure and axis object
    fig, ax = plt.subplots()
    fig.set_size_inches(1 * len(methods), 6)

    # Create a bar plot
    bars = ax.bar(methods, errors, color=colors)

    ax.set_xlabel("Method")
    ax.set_ylabel("Error")

    if semilogy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", which="both")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for bar, error in zip(bars.patches, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{error:.2f}",
            ha="center",
            va="bottom",
        )

    # Create proxy artists for the legend
    if legend is None:
        legend = {}

    patches = []
    for color, label in legend.items():
        patches.append(mpatches.Patch(color=color, label=label))

    # Add the legend
    fig.suptitle(
        f"Errors {error_norm} for {test_case} \ncomputed the {date}",
        horizontalalignment="right",
    )
    fig.legend(handles=patches, loc="outside upper right", ncols=2)

    fig.tight_layout()
    return fig, ax


def plot_errors_along_time(
    timesteps: list[float] | np.ndarray,
    label_errors: dict[str, np.ndarray],
):
    fig, ax = plt.subplots()
    ax.set_title("$l_2$ errors")
    for label, errors in label_errors.items():
        ax.plot(timesteps, errors, "x-", label=label)
    ax.legend(loc="best")
    return fig, ax
