# word_similarity/visualization.py
import matplotlib.pyplot as plt


def generate_boxplot(dataframes, labels, column='cos_sim', title='Synonym set cosine similarity'):
    """
    Generates a boxplot for the specified column in multiple DataFrames.

    Parameters:
    - dataframes (list of pd.DataFrame): List of DataFrames to plot.
    - labels (list of str): List of labels for each DataFrame.
    - column (str): Column name in the DataFrames to use for plotting.
    - title (str): Title for the boxplot.
    """
    plt.style.use('ggplot')
    plt.figure(dpi=600)

    fig, ax = plt.subplots(figsize=(16, 6))
    data = [df[column] for df in dataframes]
    
    boxplot = ax.boxplot(
        data, vert=True, showmeans=True, meanline=True,
        labels=labels, patch_artist=True,
        medianprops={'linewidth': 1, 'color': 'purple'},
        meanprops={'linewidth': 1, 'color': 'red'},
        whiskerprops={'linewidth': 1, 'color': 'black'}
    )
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    if len(dataframes) <= 5:
        offset_factor = 0.5  # Smaller gap for fewer dataframes
    elif len(dataframes) <= 10:
        offset_factor = 0.55

    for i, line in enumerate(boxplot['means']):
        x = line.get_xdata()[0]
        y = line.get_ydata()[0]
        ax.text(x + offset_factor, y + 0.02, f'Mean: {y:.2f}', va='center')
    
    plt.show()
