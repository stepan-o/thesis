import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pysal.lib.cg import alpha_shape_auto


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


def map_alpha(gdf, start, stop,
              color_col, idx_col, min_counts,
              dfd=None, action='show',
              x_col='x', y_col='y', crs='+init=epsg:4326',
              save_path_noctx='img/gen/noctx/',
              save_path_ctx='img/gen/ctx/',
              set_fixed_limits=True,
              record_counts=True):
    """

    :param gdf:
    :param start:
    :param stop:
    :param color_col:
    :param idx_col:
    :param min_counts:
    :param dfd:
    :param action:
    :param x_col:
    :param y_col:
    :param crs:
    :param save_path_noctx:
    :param save_path_ctx:
    :param set_fixed_limits:
    :param record_counts:
    :return:
    """
    def plot_map(cont, act):
        """
        internal function to plot a map
        :param cont:
        :param act:
        :return:
        """
        f, ax = plt.subplots(1, figsize=(12, 12))
        alpha.reset_index() \
            .plot(column=color_col, legend=True,
                  legend_kwds={'loc': 'lower right'},
                  ax=ax, alpha=0.5)
        for idx, mun in alpha.iterrows():
            mun_centroid = mun['geometry'].centroid
            ax.text(mun_centroid.x, mun_centroid.y, idx
                    + "\n{0:,} records".format(group_counts[idx]))
        ax.set_title("Alpha shapes of each GTHA municipality\n"
                     "based on Teranet records from {0} to {1}"
                     .format(min_idx,
                             max_idx), fontsize=20)
        if set_fixed_limits:
            ax.set_xlim(-8940996.776086302, -8723064.623629777)
            ax.set_ylim(5313237.739935117, 5555494.494204169)
        if cont:
            ctx.add_basemap(ax=ax, url=ctx.sources.ST_TONER_HYBRID,
                            alpha=0.5)
            if act == 'show':
                plt.show()
            elif act == 'save':
                plt.savefig(save_path_ctx +
                            str(min_idx)[:10]
                            + '_' +
                            str(max_idx)[:10])
        else:
            if act == 'show':
                plt.show()
            elif act == 'save':
                plt.savefig(save_path_noctx +
                            str(min_idx)[:10]
                            + '_' +
                            str(max_idx)[:10])

    # create subset from provided GeoDataFrame
    s = gdf.loc[start:stop].reset_index()
    # get min and max index from the subset
    min_idx = s[idx_col].min()
    max_idx = s[idx_col].max()
    print("{0} points in the subset from {1} to {2}."
          .format(len(s),
                  min_idx,
                  max_idx))
    # determine count of records in the subset by each alpha shape group
    group_counts = s.groupby(color_col)[x_col].count()
    if record_counts:
        # record number of counts
        dfd[int(start[:4])] = group_counts
    plot_list = \
        list(group_counts[group_counts > min_counts].index)
    mask = s[color_col].isin(plot_list)
    s = s[mask]
    alpha = s.groupby(color_col)[[x_col, y_col]] \
        .apply(lambda tab: alpha_shape_auto(tab.values))
    alpha = gpd.GeoDataFrame({'geometry': alpha}, crs=crs)
    alpha = alpha.to_crs(epsg=3857)
    plot_map(cont=False, act=action)
    plot_map(cont=True, act=action)
    return dfd
