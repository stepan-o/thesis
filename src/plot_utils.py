import contextily as ctx
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def grouped_boxplot(df, year, max_price, plot_col='price_2016', group_col='csdname',
                    title=None, output='show'):
    """
    plot a grouped boxplot of column 'plot_col' in DataFrame 'df' grouped by column 'group_col'
    :param df: pandas DataFrame
    DataFrame with variables to be plotted
    :param year: int
    take Teranet subset from this year
    :param max_price: float
    include only Teranet records with price_2016 < max_price
    :param plot_col: string
    name of the column to plot
    :param group_col: string
    name of the column to group Teranet records by
    :param title: string
    title of the plot
    :param output: string, must be 'save' or 'show'
    :return: None, plots a boxplot
    """
    f, ax = plt.subplots(1, figsize=(12, 12))
    df.query('year == {0}'.format(year)).query('price_2016 < {0}'.format(max_price)) \
        .boxplot(column=plot_col, by=group_col, vert=False, ax=ax)
    ax.set_ylabel("Municipality", fontsize=18)
    ax.set_xlabel("Consideration amount (in 2016 CAD)", fontsize=18)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlim(0, max_price)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    if title:
        ax.set_title(title, fontsize=18)
    if output == 'show':
        plt.show()
    elif output == 'save':
        f.savefig('results/plots/boxplots/teranet_{0}_{1}_by_{2}.png'
                  .format(year, plot_col, group_col), dpi=200, bbox_inches='tight')
    else:
        raise ValueError("parameter 'output' for function 'grouped_boxplot' must be either 'show' or 'save'")
    plt.close(f)


def plot_count_mean_median(s, group_col, plot_col, figsize=(8, 8), tick_label_size=16,
                           ax1_leg_label='count', ax2_leg_label1='median', ax2_leg_label2='mean',
                           cust_xticks=False, xticks_rot=45, x_ticks_lab_size=16,
                           ax1_color='darkblue', ax2_color1='mediumspringgreen', ax2_color2='deeppink',
                           ax1_yticks_sep=True, ax2_yticks_sep=True,
                           ax1_ylabel="count", ax1_xlabel="x", ax2_ylabel="mean and median",
                           ylabel_size=16, xlabel_size=16,
                           plot_title="Count, mean, and median", title_size=20,
                           ax1_leg_loc='upper left', ax2_legend_loc='center right',
                           ax1_leg_size=16, ax2_leg_size=16,
                           save_path=None, save_dpi=300):
    """
    plot count, mean, and median from groups by group_col of values in plot_col
    :param s: pandas DataFrame
        subset to plot
    :param group_col: string
        name of column to group subset by
    :param plot_col: string
        name of column to plot
    :param figsize: (int or float, int or float)
        tuple with figure dimensions
    :param tick_label_size: int
        font size for tick labels
    :param ax1_leg_label: string
        legend label for left y axis
    :param ax2_leg_label1: string
        legend label for line 1 on right y axis
    :param ax2_leg_label2: string
        legend label for line 2 on right y axis
    :param cust_xticks: boolean
        whether to use x ticks with rotation and custom font size
    :param xticks_rot: int
        rotation (degrees) for custom x ticks
    :param x_ticks_lab_size: int
        font size for custom x ticks
    :param ax1_color: string
        line color on left y axis
    :param ax2_color1: string
        line 1 color on right y axis
    :param ax2_color2: string
        line 2 color on right y axis
    :param ax1_yticks_sep: boolean
        format y ticks on left y axis to include thousands separator (e.g., 1'045'543)
    :param ax2_yticks_sep: boolean
        format y ticks on right y axis to include thousands separator (e.g., 1'045'543)
    :param ax1_ylabel: string
        label for left y axis
    :param ax1_xlabel: string
        label for x axis
    :param ax2_ylabel: string
        label for right y axis
    :param ylabel_size: int
        y label size
    :param xlabel_size: int
        x label size
    :param plot_title: string
        title of the plot
    :param title_size: int
        font size for plot title
    :param ax1_leg_loc: string
        location of legend for left y axis
    :param ax2_legend_loc: string
        location of legend for right y axis
    :param ax1_leg_size: int
        size of legend font for left y axis
    :param ax2_leg_size: int
        size of legend font for right y axis
    :param save_path: string
        path to save figure (default=None results in plt.show)
    :param save_dpi: int
        DPI to use for saved figure
    :return: None
        plots and optionally saves the figure to file
    """
    plt.rcParams['xtick.labelsize'] = tick_label_size
    plt.rcParams['ytick.labelsize'] = tick_label_size

    ax = s.groupby(group_col)[plot_col].count().plot(figsize=figsize, label=ax1_leg_label, color=ax1_color)
    if cust_xticks:
        plt.xticks(rotation=xticks_rot, fontsize=x_ticks_lab_size)
    ax2 = ax.twinx()
    s.groupby(group_col)[plot_col].median().plot(ax=ax2, color=ax2_color1, label=ax2_leg_label1)
    s.groupby(group_col)[plot_col].mean().plot(ax=ax2, color=ax2_color2, label=ax2_leg_label2)

    if ax1_yticks_sep:
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if ax2_yticks_sep:
        ax2.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.grid(False)
    ax.set_ylabel(ax1_ylabel, fontsize=ylabel_size)
    ax.set_xlabel(ax1_xlabel, fontsize=xlabel_size)
    ax2.set_ylabel(ax2_ylabel, fontsize=ylabel_size)
    ax2.set_title(plot_title, fontsize=title_size)
    ax.legend(loc=ax1_leg_loc, fontsize=ax1_leg_size)
    ax2.legend(loc=ax2_legend_loc, fontsize=ax2_leg_size)
    if save_path:
        plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
    else:
        plt.show()


def plot_hist(ser, form_x=False, form_y=False, figsize=(14, 6), kde=False, x_label=None,
              plot_mean=True, plot_median=True, mean_xlift=1.1, med_xlift=0.7, sdev=True, sdev_xlift=1.3,
              title='Distribution', title_size=20,
              x_tick_size=14, y_tick_size=14, x_lab_size=16, y_lab_size=16, mean_med_size=14,
              act='show', save_path='distribution.png', dpi=300, save_only=True):
    """
    plot distribution of the provided series
    :param ser: numpy array or pandas Series
    series from which to plot distributions
    :param form_x: boolean
    whether to add thousands separator to the x tick labels
    :param form_y: boolean
    whether to add thousands separator to the y tick labels
    :param figsize: tuple (float, float)
    size of the figure (width, height)
    :param kde: boolean
    whether to plot kernel density estimation (default = histogram)
    :param x_label: string
    label to use for x axis
    :param plot_mean: boolean
    whether to plot the mean of the series
    :param plot_median: boolean
    whether to plot the median of the series
    :param mean_xlift: float
    caption lift along the x axis for the mean
    :param med_xlift: float
    caption lift along the x axis for the median
    :param sdev: boolean
    whether to plot the standard deviation of the series
    :param sdev_xlift: float
    caption lift along the x axis for standard deviation
    :param title: string
    plot title
    :param title_size: float
    fontsize to use for the plot title
    :param x_tick_size: float
    fontsize to use for x ticks
    :param y_tick_size: float
    fontsize to use for y ticks
    :param x_lab_size: float
    fontsize to use for x axis label
    :param y_lab_size: float
    fontsize to use for y axis label
    :param mean_med_size: float
    fontsize to use for mean, median and standard deviation
    :param act: string ('show' or 'save')
    whether to show or save the plot
    :param save_path: string
    where to save the plot (relative from script location)
    :param dpi: int
    resolution for saving the plot
    :param save_only: boolean
    save without displaying
    :return: None, plots and displays or saves the result
    """
    # create figure and axis
    f, ax = plt.subplots(1, figsize=figsize)

    # plot distribution
    sns.distplot(ser, kde=kde, ax=ax)

    # plot mean of the series
    if plot_mean:
        mean = ser.mean()
        ax.axvline(mean, linestyle='--', color='deeppink')
        ax.text(mean * mean_xlift, 0, 'Mean: {0:,.2f}'.format(mean), fontsize=mean_med_size, rotation=90)
    # plot median of the series
    if plot_median:
        median = ser.median()
        ax.axvline(median, linestyle='--', color='teal')
        ax.text(median * med_xlift, 0, 'Median: {0:,.2f}'.format(median), fontsize=mean_med_size, rotation=90)
    # print standard deviation of the series
    if sdev:
        ax.text(mean * sdev_xlift, 0, 'StDev: {0:,.2f}'.format(ser.std()), fontsize=mean_med_size, rotation=90)

    # format axes
    if form_x:
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if form_y:
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # configure axes parameters
    ax.set_title(title, fontsize=title_size)
    plt.xticks(fontsize=x_tick_size)
    plt.yticks(fontsize=y_tick_size)
    if kde:
        ax.set_ylabel('Kernel density estimation (KDE)', fontsize=y_lab_size)
    else:
        ax.set_ylabel('Count of records', fontsize=y_lab_size)

    if x_label:
        ax.set_xlabel(x_label, fontsize=x_lab_size)

    # save or show results
    if act == 'show':
        plt.show()
    elif act == 'save':
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print("Saved output plot to", save_path)
        if save_only:
            plt.close(f)


def plot_heatmap(df_to_plot):
    """
    a function to plot a heat map from a pandas DataFrame
    """
    # create figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(50, 10, as_cmap=True)

    # Draw the heat map with the mask and correct aspect ratio
    sns.heatmap(df_to_plot,
                cmap=cmap,
                vmax=1,
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5},
                annot=True,
                ax=ax)

    plt.show()


def plot_subset(df_full, plot_focus_id, focus_col, x_col, y_col1,
                format_axes=True, y_col2=None,
                p_x_label="x", p_y1_label="y1", p_y2_label="y2",
                get_address=True, return_subset=False, x_min=None, x_max=None, date=True):
    """
    Creates a subset from a provided DataFrame using parameters
    `plot_focus_id` to subset `focus_col` of the DataFrame `df_full`.

    Plots `x_col` as the x-axis, `y1_col` as a primary y-axis,
    and (optional) `y2_col` as a secondary y-axis.
    :param df_full: pandas DataFrame
        DataFrame with data to be plotted
    :param plot_focus_id: int or string
        index of interest, to create the subset
    :param focus_col: string
        column to use to create the subset from the DataFrame
    :param x_col: string
        column to sort the records by
    :param y_col1: string
        column to be plotted on 1st vertical axis
    :param format_axes: boolean
        whether to format y axes (default=True)
    :param y_col2: string
        column to be plotted on 2nd vertical axis
    :param p_x_label: string
        label to be used on the x axis
    :param p_y1_label: string
        label to be used for the primary y axis
    :param p_y2_label: string
        label to be used for the secondary y axis
    :param get_address: boolean
        whether to get the address and x y from the subset
        (default=True)
    :param return_subset: boolean
        whether to return the subset, address, and x y
        (default=False)
    :param x_min: int or string (must match x axis index)
        limit to be used on x axis
    :param x_max: int or string (must match x axis index)
        limit to be used on x axis
    :param date: boolean
        whether x is datetime indexed (needed for setting x lim)
        (default=True)
    :return: default: None, plots a chart
             optional:
                plot_subset: pandas DataFrame
                    DataFrame with the subset
                address: string
                    address obtained from the subset
                x, y: float
                    coordinates obtained from the subset
    """
    # generate subset by focus_col == focus_id
    subset_to_plot = df_full[df_full[focus_col] == plot_focus_id]
    if get_address:
        # get the address
        try:
            street_number = subset_to_plot['street_number'].mode()[0]
        except IndexError:
            street_number = ""
        try:
            street_name = subset_to_plot['street_name'].mode()[0]
        except IndexError:
            street_name = ""
        try:
            street_designation = subset_to_plot['street_designation'].mode()[0]
        except IndexError:
            street_designation = ""
        try:
            municipality = subset_to_plot['municipality'].mode()[0]
        except IndexError:
            municipality = ""
        address = "{0} {1} {2}, {3}" \
            .format(street_number,
                    street_name,
                    street_designation,
                    municipality)
        x = subset_to_plot['x'].mode()[0]
        y = subset_to_plot['y'].mode()[0]
    else:
        address = ""
        x = ""
        y = ""
    # create figure and axes
    fig, axis = plt.subplots(1, figsize=(6, 6))
    # generate plot
    subset_to_plot.plot(x=x_col, y=y_col1, ax=axis)
    # format number display on y axis
    if format_axes:
        axis.get_yaxis() \
            .set_major_formatter(plt.FuncFormatter(lambda tick, loc: "{:,}".format(int(tick))))
    # set plot title
    axis.set_title("Rolling sum of Teranet transactions"
                   "\nfor pin: {0}".format(plot_focus_id) +
                   "\n{0:,} total records".format(len(subset_to_plot)) +
                   "\naddress: {0}".format(address) +
                   "\nx: {0}, y: {1}".format(x, y))
    axis.set_ylabel(p_y1_label)
    axis.set_xlabel(p_x_label)
    axis.grid(linestyle=':')

    if y_col2:
        # plot on secondary axis
        axis2 = axis.twinx()
        subset_to_plot.plot(x=x_col, y=y_col2,
                            ax=axis2, color='orange', alpha=0.5)
        # format number display on y axis
        if format_axes:
            axis2.get_yaxis() \
                .set_major_formatter(plt.FuncFormatter(lambda tick, loc: "{:,}".format(int(tick))))
        axis2.legend(loc='center left')
        axis2.set_ylabel(p_y2_label)

    # set limits on x axis
    if x_min and x_max:
        if date:
            axis.set_xlim(left=pd.to_datetime(x_min), right=pd.to_datetime(x_max))
        else:
            axis.set_xlim(left=x_min, right=x_max)
    elif x_min:
        if date:
            axis.set_xlim(left=pd.to_datetime(x_min))
        else:
            axis.set_xlim(left=x_min)
    elif x_max:
        if date:
            axis.set_xlim(right=pd.to_datetime(x_max))
        else:
            axis.set_xlim(right=x_max)
    plt.show()

    if return_subset:
        return subset_to_plot, address, x, y


def map_subset(gdf_to_plot, plot_focus_ids,
               color_col=None, cmap='viridis', plot_alpha=0.01,
               plot_title="", title_font_size=18,
               zoom_center=None, zoom_radius=None):
    """
    map a subset from a GeoDataFrame
    :param gdf_to_plot: geopandas GeoDataFrame
        GeoDataFrame with records to be mapped
    :param plot_focus_ids: pandas Series
        ids to be mapped, used to generate subset of GeoDataFrame
    :param color_col: string
        column to use to color points
    :param cmap: string
        Matplotlib color map to be used to color points
        default: 'viridis'
    :param plot_alpha: float
        transparency of the points to be plotted
    :param plot_title:
        title to use for the map
    :param title_font_size:
        fontsize of the title
    :param zoom_center: tuple (int, int)
        central point of the zoom in CRS EPSG:3857 (Web Merkator projection, metres)
    :param zoom_radius: int or float
        radius of the zoom (x_min = zoom_center - radius, etc.)
        in meters (as per EPSG:3857)
    :return:
    """
    # plot results
    fig, axis = plt.subplots(1, figsize=(12, 12))
    gdf_to_plot.loc[plot_focus_ids].to_crs(epsg=3857) \
        .plot(column=color_col, cmap=cmap, legend=True,
              ax=axis, alpha=plot_alpha)
    if zoom_center:
        x_zoom, y_zoom = zoom_center
        axis.set_xlim(x_zoom - zoom_radius, x_zoom + zoom_radius)
        axis.set_ylim(y_zoom - zoom_radius, y_zoom + zoom_radius)

    # noinspection PyTypeChecker
    ctx.add_basemap(ax=axis,
                    url=ctx.sources.ST_TONER_HYBRID,
                    alpha=0.5)
    plt.title(plot_title, fontsize=title_font_size)
    plt.show()


def get_plot_title(counts, minrecords, id_lab):
    """
    helper function to generate plot title
    :param counts: pandas Series
        value counts from Teranet records (by pin, xy, da_id)
    :param minrecords: int
        minimum records per pin, xy, or da_id,
        used to filter Teranet counts
    :param id_lab: string
        label of the index (pin, xy, da_id, etc.)
    :return: title: string
        title of the plot
    """
    unique_ids = counts[counts > minrecords].index
    title = "{0}s with > {1:,} Teranet records" \
                .format(id_lab, minrecords) + \
            "\n{0:,} {1}s ({2:.5f}% of the total)" \
                .format(len(unique_ids),
                        id_lab,
                        len(unique_ids) / len(counts.index) * 100) + \
            "\n(coloured by count of records)"
    return title

