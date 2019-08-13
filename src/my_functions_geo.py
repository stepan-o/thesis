import numpy as np
import pandas as pd
import geopandas as gpd
import pysal as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pysal
import contextily as ctx


def map_neighbours(gdf, wm, focus_id, focus_rad=0.025, geometry_col='geometry',
                   plot_title='Neighbours',
                   plot_buffer=False, buffer_rad=0, buffer_color='red', buffer_alpha=0.3,
                   base_color='black', base_linewidth=0.1, base_alpha=1,
                   focus_color='red', focus_linewidth=0, focus_alpha=1,
                   neighs_color='lime', neighs_linewidth=0, neighs_alpha=1):
    """
    map focus polygon from a GeoDataFrame and its neighbourhood
    :param buffer_alpha:
    :param buffer_color:
    :param buffer_rad:
    :param plot_buffer:
    :param plot_title:
    :param neighs_alpha:
    :param neighs_linewidth:
    :param neighs_color:
    :param focus_alpha:
    :param focus_linewidth:
    :param focus_color:
    :param base_alpha:
    :param base_linewidth:
    :param base_color:
    :param gdf: GeoDataFrame
                GeoDataFrame with polygons of interest
    :param wm:  PySal weight matrix
                weight matrix generated in PySal for GeoDataFrame
    :param focus_id: int or string
                index of the polygon to highlight from the GeoDataFrame
    :param focus_rad: float
                zoom radius for the map
    :param geometry_col: string
                name of the geometry column in the GeoDataFrame
    :return: None, plots a map
    """
    print(plot_title)
    print("Number of observations",
          wm.n)
    print("Average number of neighbors",
          wm.mean_neighbors)
    print("Min number of neighbors",
          wm.min_neighbors)
    print("Max number of neighbors",
          wm.max_neighbors)
    print("No of islands (observations disconnected):",
          len(wm.islands))
    card = pd.Series(wm.cardinalities)
    f, ax = plt.subplots(1)
    sns.distplot(card, rug=True, bins=20)
    ax.set_title("Distribution of cardinalities for " + plot_title)
    plt.show()
    # Setup figure
    f, ax = plt.subplots(1, figsize=(6, 6))
    # Plot base layer of polygons
    gdf.plot(ax=ax, facecolor=base_color,
             linewidth=base_linewidth, alpha=base_alpha)
    # Select focal polygon
    focus = gdf.loc[[focus_id], [geometry_col]]
    # Plot focal polygon
    focus.plot(facecolor=focus_color, linewidth=focus_linewidth,
               alpha=focus_alpha, ax=ax)
    if plot_buffer:
        focus.centroid.plot(ax=ax)
        focus.centroid.buffer(buffer_rad).plot(ax=ax,
                                               color=buffer_color,
                                               alpha=buffer_alpha)
    # Plot neighbors
    neis = gdf.loc[list(wm[focus_id].keys())]
    neis.plot(ax=ax, facecolor=neighs_color,
              linewidth=neighs_linewidth, alpha=neighs_alpha)
    # Title
    f.suptitle(plot_title + " of {0}".format(focus_id))
    # Style and display on screen
    ax.set_ylim(focus.centroid.y[0] - focus_rad, focus.centroid.y[0] + focus_rad)
    ax.set_xlim(focus.centroid.x[0] - focus_rad, focus.centroid.x[0] + focus_rad)
    plt.show()


def plot_moran(value_slag, value, plot_title=""):
    """
    plot a Moran plot for analyzing spatial autocorrelation
    """
    # Plot values
    sns.jointplot(x=value_slag,
                  y=value,
                  kind="reg")
    ax = plt.gca()
    # Add vertical and horizontal lines
    ax.axvline(0, c='k', alpha=0.5)
    ax.axhline(0, c='k', alpha=0.5)
    plt.suptitle(plot_title + " Moran plot for spatial autocorrelation")
    # Display
    plt.show()


def moran_i(series, wm, title=""):
    """
    compute and print Moran's I and its associated p-value
    from a Series and its associated weight matrix
    """
    mi = ps.explore.esda.moran.Moran(series, wm)
    print("Moran I for " + title)
    print("Moran I value = " + str(mi.I))
    print("Associated p-value = " + str(mi.p_sim))


def map_lisa(gdf, sgn_col='significant', quad_col='quadrant', plot_title="",
             zoom=0, left_lim=0, bottom_lim=0):
    """
    map LISA quadrants from supplied GeoDataFrame
    :param bottom_lim:
    :param left_lim:
    :param zoom:
    :param plot_title:
    :param gdf:
    :param sgn_col:
    :param quad_col:
    :return:
    """
    # Setup the figure and axis
    f, ax = plt.subplots(1, figsize=(9, 9))
    # Plot insignificant clusters
    ns = gdf.loc[gdf[sgn_col] == False, 'geometry']
    ns.plot(ax=ax, color='k')
    # Plot HH clusters
    hh = gdf.loc[(gdf[quad_col] == 1) & (gdf[sgn_col] == True), 'geometry']
    hh.plot(ax=ax, color='red')
    # Plot LL clusters
    ll = gdf.loc[(gdf[quad_col] == 3) & (gdf[sgn_col] == True), 'geometry']
    ll.plot(ax=ax, color='blue')
    # Plot LH clusters
    lh = gdf.loc[(gdf[quad_col] == 2) & (gdf[sgn_col] == True), 'geometry']
    lh.plot(ax=ax, color='#83cef4')
    # Plot HL clusters
    hl = gdf.loc[(gdf[quad_col] == 4) & (gdf[sgn_col] == True), 'geometry']
    hl.plot(ax=ax, color='#e59696')
    # Style and draw
    f.suptitle('LISA' + plot_title, size=20)
    f.set_facecolor('white')
    if zoom:
        ax.set_xlim(left_lim, left_lim + zoom)
        ax.set_ylim(bottom_lim, bottom_lim + zoom)
    plt.show()


def column_kde(series_to_plot, num_bins=7, split_type="quantiles", bw=0.15,
               plot_title="", xlabel="x", ylabel="y"):
    """
    v1.0
    function that plots: Kernel Density Estimation (KDE)
                         rugplot
                         shows a classification of the distribution based on 'num_bins' and 'split_type'

    Plots data from the global variable (GeoDataFrame) 'teranet_da_gdf'

    ----------------
    Input arguments: series_to_plot -- pandas Series -- series to be plotted

                     num_bins       -- int    -- number of bins to be used for the split of
                                                 the distribution (default=7)

                     split_type     -- str    -- type of the split of the distribution (default='quantiles')
                                                 must be either 'quantiles', 'equal_interval', or 'fisher_jenks'

                     bw             -- float  -- bandwidth to be used for KDE (default=0.15)

    --------
    Returns:     None, plots a KDE, rugplot, and bins of values in 'column_to_plot'
    """
    # generate a list of bins from the split of the distribution using type of split provided in 'split_type'
    if split_type == 'quantiles':
        classi = ps.Quantiles(series_to_plot, k=num_bins)
    elif split_type == 'equal_interval':
        classi = ps.Equal_Interval(series_to_plot, k=num_bins)
    elif split_type == 'fisher_jenks':
        classi = ps.Fisher_Jenks(series_to_plot, k=num_bins)
    elif type(split_type) == str:
        raise ValueError("Input parameter 'split_type' must be either 'quantiles', " +
                         "'equal_interval', or 'fisher_jenks'.")
    else:
        raise TypeError("Input parameter 'split_type' must be a string and either 'quantiles', " +
                        "'equal_interval, or 'fisher_jenks'.")
    # print the bins
    print(classi)

    # create figure and axis
    f, ax = plt.subplots(1, figsize=(9, 6))

    # plot KDE of the distribution
    sns.kdeplot(series_to_plot,
                shade=True,
                label='Distribution of counts of Teranet records per DA',
                bw=bw)

    # plot a rugplot
    sns.rugplot(series_to_plot, alpha=0.5)

    # plot the split of the distribution
    for classi_bin in classi.bins:
        ax.axvline(classi_bin, color='magenta', linewidth=1, linestyle='--')

    # plot the mean and the median
    ax.axvline(series_to_plot.mean(),
               color='deeppink',
               linestyle='--',
               linewidth=1)

    ax.text(series_to_plot.mean(),
            0,
            "Mean: {0:.2f}".format(series_to_plot.mean()),
            rotation=90)

    ax.axvline(series_to_plot.median(),
               color='coral',
               linestyle=':')

    ax.text(series_to_plot.median(),
            0,
            "Median: {0:.2f}".format(series_to_plot.median()),
            rotation=90)

    # configure axis parameters
    ax.set_title(plot_title,
                 fontdict={'fontsize': '18', 'fontweight': '3'})
    ax.set_xlabel(xlabel,
                  fontdict={'fontsize': '16', 'fontweight': '3'})
    ax.set_ylabel(ylabel,
                  fontdict={'fontsize': '16', 'fontweight': '3'})

    ax.legend(loc='best')

    plt.show()


def series_choropleth(column_to_plot, num_bins=7, split_type='quantiles', polygon=False,
                      minx_coef=1, maxx_coef=1, miny_coef=1, maxy_coef=1):
    """
    v1.0
    'column_choropleth' is a function that creates a choropleth map of column values
    based on the specified split type

    Plots data from the global variable (GeoDataFrame) 'teranet_da_gdf'

    ----------------
    Input arguments: column_to_plot -- string -- name of the column to be plotted

                     num_bins       -- int    -- number of bins to be used for the split of
                                                 the distribution (default=7)

                     split_type     -- str    -- type of the split of the distribution (default='quantiles')
                                                 must be either 'quantiles', 'equal_interval', or 'fisher_jenks'

                     polygon        -- bool   -- whether to plot the polygon of interest ('downtown_polygon_gdf')
                                                 (default=False)

                     minx_coef      -- float  -- min x coefficient to be used for zooming the map (default=1)
                     maxx_coef      -- float  -- max x coefficient to be used for zooming the map (default=1)
                     miny_coef      -- float  -- min y coefficient to be used for zooming the map (default=1)
                     maxy_coef      -- float  -- max y coefficient to be used for zooming the map (default=1)


    --------
    Returns:     None, plots a choropleth map

    -----------------
    Global variables:  teranet_da_gdf -- GeoDataFrame -- GeoDataFrame with dissimination areas
                                                         joined with Teranet aggregate
                                                         generated in step 1.2, 2.2, 3.2

    """
    # global variable used by the function: GeoDataFrame 'teranet_da_gdf'
    global teranet_da_gdf

    # check input given for split type (input parameter 'split_type')
    if split_type != 'quantiles' and split_type != 'equal_interval' and split_type != 'fisher_jenks':
        if type(split_type) == str:
            raise ValueError("Input parameter 'split_type' must be either 'quantiles', " +
                             "'equal_interval', or 'fisher_jenks'.")
        else:
            raise TypeError("Input parameter 'split_type' must be a string and either 'quantiles', " +
                            "'equal_interval, or 'fisher_jenks'.")

    # create figure and axis
    f, ax = plt.subplots(1, figsize=(12, 12))

    # create a choropleth map of column values based on the specified split
    teranet_da_gdf.to_crs(epsg=3857).plot(column=column_to_plot,
                                          scheme=split_type,
                                          ax=ax,
                                          legend=True,
                                          alpha=0.5)

    if polygon == True:
        global downtown_polygon_gdf
        downtown_polygon_gdf.to_crs(epsg=3857).plot(ax=ax, color='red', alpha=0.3)

    # zoom the map
    minx, miny, maxx, maxy = teranet_da_gdf.to_crs(epsg=3857).total_bounds
    minx = minx + (maxx - minx) / minx_coef
    maxx = maxx - (maxx - minx) / maxx_coef
    miny = miny + (maxy - miny) / miny_coef
    maxy = maxy - (maxy - miny) / maxy_coef

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # add basemap
    ctx.add_basemap(ax, url=ctx.sources.ST_TONER_BACKGROUND, alpha=0.5)

    # configure axis parameters
    ax.set_axis_off()
    ax.set_title("Choropleth of column '{0}' from Teranet records\n'{1}' split of the distribution\nfrom {2} to {3}"
                 .format(column_to_plot, split_type, start_date, end_date),
                 fontdict={'fontsize': '16', 'fontweight': '3'})

    plt.show()


def map_top_das(start_date, end_date,
                da_num=20, column='teranet_sales_count',
                minx_coef=1, maxx_coef=1, miny_coef=1, maxy_coef=1,
                display_pin_counts=False):
    """
    v1.0
    function that maps top 20 DAs by values in 'column' on a basemap
    plots a barchart of these values,
    and displays counts of Teranet records by 'pin' from these top DAs (optional)

    Plots data from the global variable (GeoDataFrame) 'teranet_da_gdf'

    uses another user-defined function 'unique_pins' to analyze 'pin's in the
    subset of Teranet records from top DAs

    ----------------
    Input arguments:  da_num     -- int    -- number of top DAs to be plotted (default=20)

                      column     -- string -- name of the column to be used for sorting
                                              (default='teranet_sales_count')

                      minx_coef  -- float  -- min x coefficient to be used for zooming the map (default=1)
                      maxx_coef  -- float  -- max x coefficient to be used for zooming the map (default=1)
                      miny_coef  -- float  -- min y coefficient to be used for zooming the map (default=1)
                      maxy_coef  -- float  -- max y coefficient to be used for zooming the map (default=1)

                      display_pin_counts -- bool -- option to display counts of Teranet records by pins
                                                    from the subset of Teranet records from top DAs
                                                    (default=False)
    --------
    Returns:     None, plots top 20 DAs with a basemap and a barchart

    -----------------
    Global variables: teranet_da_subset_df -- DataFrame    -- DataFrame with subset of Teranet sales
                                                              generated in steps 1.1, 2.1, 3.1

                      teranet_da_gdf       -- GeoDataFrame -- GeoDataFrame with dissimination areas
                                                              joined with Teranet aggregate
                                                              generated in step 1.2, 2.2, 3.2

                      unique_pins          -- user func    -- function 'unique_pins' used to display
                                                              counts of pins from subset of Teranet
                                                              records from top 'da_num' DAs
                                                              (used if 'display_pin_counts'=True)
    """
    # global variables
    global teranet_da_subset_df, teranet_da_gdf, unique_pins

    # map top 20 DAs by column values with a basemap

    # create a list of indexes of top DAs by 'column' values
    top_da_ids = teranet_da_gdf[column].sort_values(ascending=False)[:da_num].index

    # create figure and axis
    f, ax = plt.subplots(1, figsize=(12, 12))

    # plot top 20 DAs by count of Teranet sales for the time period
    teranet_da_gdf.to_crs(epsg=3857).loc[top_da_ids].plot(ax=ax)

    # plot counts of Teranet records for each of top 20 DAs
    for index, centroid in teranet_da_gdf.to_crs(epsg=3857).loc[top_da_ids].centroid.iteritems():
        x, y = centroid.coords[0]
        ax.text(x, y,
                "DA #" + str(index) + ": " + str(teranet_da_gdf.loc[index, column]) + " records")

    # add basemap
    ctx.add_basemap(ax, url=ctx.sources.ST_TONER_BACKGROUND, alpha=0.5)

    # zoom the map
    minx, miny, maxx, maxy = teranet_da_gdf.to_crs(epsg=3857).total_bounds
    minx = minx + (maxx - minx) / minx_coef
    maxx = maxx - (maxx - minx) / maxx_coef
    miny = miny + (maxy - miny) / miny_coef
    maxy = maxy - (maxy - miny) / maxy_coef

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # set axis parameters
    plt.axis('equal')
    ax.set_axis_off()
    ax.set_title("Top {0} DAs by '{1}' of Teranet records\nfrom {2} to {3}"
                 .format(da_num, column, start_date, end_date),
                 fontdict={'fontsize': '16', 'fontweight': '3'})

    plt.show()

    # plot the barchart with counts of top 20 DAs

    # create a new figure and axis
    f, ax = plt.subplots(1, figsize=(6, 8))

    # plot a horizontal barchart, same subset as above, inverse order, for highest values to show on the top
    teranet_da_gdf.loc[top_da_ids, column] \
        .reindex(index=teranet_da_gdf.loc[top_da_ids].index[::-1]) \
        .plot(kind='barh', color='gray', ax=ax)

    # plot the mean
    ax.axvline(teranet_da_gdf.loc[top_da_ids, column].mean(), linestyle='--', color='deeppink')
    ax.text(teranet_da_gdf.loc[top_da_ids, column].mean() * 0.9,
            10,
            "Mean of top {0} DAs: {1:.2f}"
            .format(da_num,
                    teranet_da_gdf.loc[top_da_ids, column].mean()),
            rotation=90,
            fontdict={'fontsize': '14', 'fontweight': '3'})

    # set axis parameters
    ax.set_title("Top {0} DAs by count of Teranet records\nfrom {1} to {2}".format(da_num, start_date, end_date),
                 fontdict={'fontsize': '16', 'fontweight': '3'})
    ax.set_ylabel("DA id", fontdict={'fontsize': '14', 'fontweight': '3'})
    ax.set_xlabel("Count of Teranet records", fontdict={'fontsize': '14', 'fontweight': '3'})

    plt.show()

    # display counts of Teranet records per pin from the top DAs (optional)
    if display_pin_counts == True:
        # create a mask to subset Teranet records -- all records with 'da_id' in the list of top_da_ids
        mask = teranet_da_subset_df['da_id'].isin(top_da_ids)

        # call function 'unique_pins' to display counts of unique pins in the subset of Teranet records from top DAs
        unique_pins(teranet_da_subset_df[mask])


def subset_teranet(start_date, end_date,
                   filter_top_outliers=False, max_price=None,
                   filter_bottom_outliers=False, min_price=None):
    """
    v1.0
    function that creates a subset from the full cleaned Teranet dataset

    Creates a subset from the global variable (DataFrame) 'teranet_da_df'
    from 'start_date' to 'end_date'

    (optionally) can filter outliers with price < 'min_price'
                                       or price > 'max_price'

    ----------------
    Input arguments:  start_date  -- string    -- string containing the start date for the subset
                      end_date    -- string    -- string containing the end date for the subset

                      filter_top_outliers -- bool  -- option to filter out all records
                                                      with 'consideration_amt' > max_price
                                                      (default=False)
                      max_price           -- float -- max price to use for filtering outliers
                                                      (default=None)


                      filter_bottom_outliers -- bool  -- option to filter out all records
                                                         with 'consideration_amt' < min_price
                                                         (default=False)
                      max_price              -- float -- min price to use for filtering outliers
                                                         (default=None)
    --------
    Returns:          teranet_da_subset_df -- DataFrame -- DataFrame with subset of (filtered)
                                                           Teranet records
    -----------------
    Global variables: teranet_da_df        -- DataFrame -- DataFrame with Teranet dataset
                                                           loaded from file in step (1.3)
    """
    # global variables
    global teranet_da_df

    # create the Teranet subset using the dates
    teranet_da_subset_df = teranet_da_df.set_index('registration_date') \
        .sort_index()

    teranet_da_subset_df = teranet_da_subset_df.loc[start_date:end_date] \
        .reset_index()

    # (optional) remove outliers above 'max_price'
    if filter_top_outliers == True:
        # remember lenth before filtering outliers
        old_len = len(teranet_da_subset_df)

        # filter Teranet records with 'consideration_amt' > max_price
        teranet_da_subset_df.query('consideration_amt <= @max_price', inplace=True)

        # print results of filtering
        print("{0:,}({1:.2f}% of the total {2:,} records) outliers with\
              \n'consideration_amt' > {3:,} have been removed from 'teranet_da_subset_df.'"
              .format(old_len - len(teranet_da_subset_df),
                      (old_len - len(teranet_da_subset_df)) / old_len * 100,
                      old_len,
                      max_price))

    # (optional) remove outliers below 'min_price'
    if filter_bottom_outliers == True:
        # remember lenth before filtering outliers
        old_len = len(teranet_da_subset_df)

        # filter Teranet records with 'consideration_amt' < max_price
        teranet_da_subset_df.query('consideration_amt >= @min_price', inplace=True)

        # print results of filtering
        print("\n{0:,}({1:.2f}% of the total {2:,} records) outliers with\
              \n'consideration_amt' < {3:,} have been removed from 'teranet_da_subset_df.'"
              .format(old_len - len(teranet_da_subset_df),
                      (old_len - len(teranet_da_subset_df)) / old_len * 100,
                      old_len,
                      min_price))

    print("\n---\nReturned DataFrame 'teranet_da_subset_df' with {0:,} records (out of the {1:,} total)"
          .format(len(teranet_da_subset_df),
                  len(teranet_da_df)))
    print("subset starting from '{0}' to '{1}'.\n"
          .format(start_date,
                  end_date))

    return teranet_da_subset_df


def join_teranet_da(agg_func='count',
                    teranet_column_name='registration_date',
                    agg_column_name='teranet_sales_count'):
    """
    v1.0
    function that joins Teranet aggregate with DA GeoDataFrame

    groups Teranet records by 'da_id' (Dissimination Area id)
    and generates an aggregate from Teranet records using the
    function supplied as 'agg_func'

    joins the aggregate with the GeoDataFrame on indexes

    returns the GeoDataFrame with a new column
    produced by aggregating Teranet records grouped by 'da_id'

    ----------------
    Input arguments:  agg_func  -- string   -- function to be used for aggregation
                                               of grouped Teranet records
                                               'count'/'min'/'max'/'mean'/'median'/'stdev', etc.
                                               see Pandas documentation for function df.agg()
                                               (default='count')

                      teranet_column_name   -- column from Teranet groups to apply the 'agg_func' to
                                               (default='registration_date')

                      agg_column_name       -- name of the column with aggregate from Teranet records
                                               to be renamed in the final joined GeoDataFrame
                                               (default='teranet_sales_count')

    --------
    Returns:          teranet_da_gdf -- GeoDataFrame -- GeoDataFrame with DA geometry
                                                        and a new column with aggregated
                                                        values from grouped Teranet records
                                                        and CRS taken from DA GeoDataFrame
    -----------------
    Global variables: teranet_da_subset_df  -- DataFrame -- DataFrame with Teranet subset
                                                            generated in (step 1.1)

                      da_income_gdf         -- GeoDataFrame -- GeoDataFrame with dissimination areas
                                                               loaded from Esri Open Data API in step (0.2)
    """
    # global variables (description in the docstring)
    global teranet_da_subset_df, da_income_gdf

    # group Teranet records by values in column 'group_col'
    da_groups = teranet_da_subset_df.groupby(by='da_id')

    # produce aggregated values for groups of Teranet records
    teranet_aggregate = pd.DataFrame(da_groups[teranet_column_name].agg(agg_func))
    print("Teranet records have been aggregated using function '{0}'".format(agg_func))
    print("on values in column '{0}', Teranet records grouped using values in column 'da_id'."
          .format(teranet_column_name))

    # save projection information from GeoDataFrame with DAs
    crs = da_income_gdf.crs
    print("\n---\nProjection of Dissimination Areas (will be passed on to the new GeoDataFrame 'teranet_da_gdf'):")
    print(crs)

    # JOIN Teranet aggregate to GeoDataFrame with DAs (produces a DataFrame)
    # joined on indexes ( !!! VALID ONLY FOR GROUPING MADE BY 'da_id' !!! )
    print("\nTeranet aggregate joined on:\n", teranet_aggregate.index)
    print("\nGeoDataFrame with DAs joined on:\n", da_income_gdf.index)
    teranet_da_gdf = pd.merge(teranet_aggregate,
                              da_income_gdf,
                              left_on=teranet_aggregate.index,
                              right_on=da_income_gdf.index)

    # rename the column containing aggregate Teranet information
    teranet_da_gdf.rename(columns={teranet_column_name: agg_column_name}, inplace=True)
    # rename the column used to perform the join (indexes of df and gdf) to 'da_id'
    teranet_da_gdf.rename(columns={'key_0': 'da_id'}, inplace=True)
    # set index of the new GeoDataFrame to 'da_id'
    teranet_da_gdf.set_index('da_id', inplace=True)

    # convert DataFrame to GeoDataFrame
    teranet_da_gdf = gpd.GeoDataFrame(teranet_da_gdf,
                                      geometry=teranet_da_gdf['geometry'])
    # add projection information to the new GeoDataFrame
    teranet_da_gdf.crs = crs

    print("\nTeranet data aggregate JOINED to GeoDataFrame with Dissimination Areas on indexes!")
    print("\n---\nReturning GeoDataFrame 'teranet_da_gdf' with DA geometry and a new column '{0}' with aggregate \
produced from Teranet records using function '{1}' on values in column '{2}'!\n"
          .format(agg_column_name, agg_func, teranet_column_name))

    return teranet_da_gdf


def da_polygon_sjoin(polygon_gdf,
                     annotate_column=False, ann_col_name=None, ann_label=None, start_date=None, end_date=None):
    """
    v1.0
    function performs the SPATIAL JOIN of GeoDataFrame with Dissimination Areas
    with the provided GeoDataFrame containing polygons of interest

    CRS of the polygon of interest needs to be specified as it will be converted
    to CRS of 'teranet_da_gdf' during the SPATIAL JOIN

    SPATIAL JOIN parameters -- how='inner', op='within'

    joins data from the global variable (GeoDataFrame) 'teranet_da_gdf'

    spatial join is performed on the GeoDataFrame with DA polygons and a column with Teranet aggregate,
    not on the actual points of Teranet records

    annotates the IDs (index) of the DAs from the subset on the map

    (optional) can add annotations with the values from column 'ann_col_name' in 'teranet_da_gdf'
    in this case, 'start_date' and 'end_date' need to be provided as well for the map title

    ----------------
    Input arguments: polygon_gdf    -- GeoDataFrame -- GeoDataFrame containing polygon(s) of interest
                                                       it will be used to subset GeoDataFrame of DAs
                                                       via a SPATIAL JOIN
                                                       CRS of the polygon needs to be specified and
                                                       will be converted to CRS of 'teranet_da_gdf'
                                                       during the SPATIAL JOIN

                     annotate_column -- bool   -- option to add annotations with values from a column
                                                  found in GeoDataFrame 'teranet_da_gdf'
                                                  to each DA on the map (e.g., counts of Teranet records)
                                                  (default=False)

                     ann_col_name    -- string -- column name from GeoDataFrame 'teranet_da_gdf'
                                                  to provide values if 'annotate_column' is True
                                                  Note: if true, 'start_date' and 'end_date' need to
                                                  be provided as well for map annotation
                                                  (default=None)

                     ann_label       -- string -- label to be used after the values from 'ann_col_name'
                                                  to annotate the map (e.g., 'records', '$ mean', etc.)
                                                  (default=None)

                     start_date      -- string -- start_date of Teranet subset, used for the map title
                                                  needed if 'ann_col_name' is True
                                                  (default=None)

                     end_date        -- string -- end_date of Teranet subset, used for the map title
                                                  needed if 'ann_col_name' is True
                                                  (default=None)
    --------
    Returns:     teranet_da_polygon_gdf -- GeoDataFrame -- subset of GeoDataFrame with DAs from
                                                           the SPATIAL JOIN with polygon of interest

    -----------------
    Global variables:  teranet_da_gdf -- GeoDataFrame -- GeoDataFrame with dissimination areas
                                                         joined with Teranet aggregate
                                                         generated in step 1.2, 2.2, 3.2


    """
    # global variables (see Docstring for description)
    global teranet_da_gdf
    # perform the SPATIAL JOIN between the Teranet DA GeoDataFrame and the polygon of interest
    teranet_da_polygon_gdf = gpd.sjoin(teranet_da_gdf, polygon_gdf.to_crs(teranet_da_gdf.crs),
                                       how='inner',
                                       op='within')

    # plot the GeoDataFrame with the polygon of interest
    # create figure and axis
    f, ax = plt.subplots(1, figsize=(12, 12))

    # plot subset of DAs
    teranet_da_polygon_gdf.to_crs(epsg=3857).plot(ax=ax, color='red', edgecolor='black', alpha=0.3)

    # plot the polygon of interest
    polygon_gdf.to_crs(epsg=3857).plot(ax=ax, alpha=0.3)

    # add basemap
    ctx.add_basemap(ax, url=ctx.sources.ST_TONER_BACKGROUND, alpha=0.1)

    # set axis parameters
    ax.set_title("{0} DAs selected within the polygon of interest".format(len(teranet_da_polygon_gdf)),
                 fontdict={'fontsize': '16', 'fontweight': '3'})
    ax.set_axis_off()

    # zoom the map to the polygon of interest
    minx, miny, maxx, maxy = downtown_polygon_gdf.to_crs(epsg=3857).total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # add annotation to each DA with (optional) values from 'ann_col_name' and 'ann_label'
    if annotate_column == True:
        # plot counts of Teranet records for each of top 20 DAs
        for index, centroid in teranet_da_polygon_gdf.to_crs(epsg=3857).centroid.iteritems():
            x, y = centroid.coords[0]
            ax.text(x,
                    y,
                    "DA #" + str(index) + ": \n" + str(teranet_da_gdf.loc[index, ann_col_name]) + ann_label,
                    verticalalignment='center',
                    horizontalalignment='center')
        # set plot title
        ax.set_title("{0} DAs selected within the polygon of interest\nfrom {1} to {2}"
                     .format(len(teranet_da_polygon_gdf),
                             start_date,
                             end_date),
                     fontdict={'fontsize': '16', 'fontweight': '3'})


    else:
        # plot IDs of DAs in the subset generated by the SPATIAL JOIN
        for index, centroid in teranet_da_polygon_gdf.to_crs(epsg=3857).centroid.iteritems():
            x, y = centroid.coords[0]
            ax.text(x,
                    y,
                    "DA #" + str(index),
                    verticalalignment='center',
                    horizontalalignment='center')
        # set plot title
        ax.set_title("{0} DAs selected within the polygon of interest".format(len(teranet_da_polygon_gdf)),
                     fontdict={'fontsize': '16', 'fontweight': '3'})

    plt.show()

    print("A subset with {0} DAs was created via a SPATIAL JOIN of GeoDataFrame 'teranet_da_gdf' \
and GeoDataFrame 'polygon_gdf'.\n---\n".format(len(teranet_da_polygon_gdf)))

    return teranet_da_polygon_gdf


def crop_teranet_daid(da_subset, start_date, end_date):
    """
    v1.0
    function that takes a (larger) subset of Teranet records
    from the global environment and "crops" it by 'da_id's from 'da_subset' list
    ----------------
    Input arguments: da_subset       -- list -- list with 'da_id's to subset Teranet records by
                                                and check unique 'pin' values

                     start_date      -- string -- start_date of Teranet subset

                     end_date        -- string -- end_date of Teranet subset
    --------
    Returns:      teranet_da_subset_df_cropped -- DataFrame -- DataFrame with a subset of Teranet records
                                                              "cropped" by 'da_id'

    -----------------
    Global variables:     teranet_da_subset_df -- DataFrame -- DataFrame with a subset of Teranet records
                                                               to be "cropped" by 'da_id'

    """
    # global variables (description in the Docstring)
    global teranet_da_subset_df

    # mask to be used to subset Teranet records
    mask = teranet_da_subset_df['da_id'].isin(da_subset)

    # crop the global DataFrame with Teranet records
    teranet_da_subset_df_cropped = teranet_da_subset_df.loc[mask]

    print('Teranet records "cropped" by Dissimination Area ids from the list provided.\n---')

    print("Out of total {0:,} Teranet records from {1} to {2}, {3:,} were found in {4} provided Dissimination Areas."
          .format(len(teranet_da_subset_df),
                  start_date,
                  end_date,
                  len(teranet_da_subset_df_cropped),
                  len(da_subset)))

    print(
        '\nReturned {0:,} ({1:.2f}%) records remaining out of {2:,} total from {3} to {4} after "cropping" by "da_id".'
        .format(len(teranet_da_subset_df_cropped),
                len(teranet_da_subset_df_cropped) / len(teranet_da_subset_df) * 100,
                len(teranet_da_subset_df),
                start_date,
                end_date))

    return teranet_da_subset_df_cropped


def unique_pins(teranet_subset):
    """
    v1.0
    function that displays unique 'pin's from a provided subset of Teranet records
    ----------------
    Input arguments: teranet_subset  -- DataFrame -- DataFrame with a subset of Teranet records
    --------
    Returns:         None, displays values counts for 'pin's in the provided subset
    """
    # print number of Teranet records with unique pins
    print("\n---\nOut of {0:,} provided Teranet records, {1:,} ({2:.2f}%) have unique pins."
          .format(len(teranet_subset),
                  len(teranet_subset['pin'].value_counts()),
                  len(teranet_subset['pin'].value_counts()) / len(teranet_subset) * 100))

    # print counts of Teranet records by 'pin'
    print("\nRecords per 'pin' in the provided Teranet subset:\n")
    print(teranet_subset['pin'].value_counts())
