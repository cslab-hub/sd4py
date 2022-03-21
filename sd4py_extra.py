import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from textwrap import wrap
import datetime
import networkx as nx
import sd4py


def bootstrapping(subgroups, data, metric_function, aggregation_function=None, ignore_defaults=False, number_simulations=100, frac=1/3, replace=True):
    '''
    Provides some estimate of variability for subgroups. Multiple samples (with replacement) are drawn from the data, 
    and subgroups are evaluated for each sample (using the metric_function).
    The aggregation function is then applied to this data (e.g. to select 0.05 and 0.95 quantiles) to give a final description for each subgroup. 
    
    Parameters
    ----------------
    subgroups: PySubgroup object or list of PySubgroup objects
        The subgroup(s) for which to perform bootstrapping.
    data: DataFrame
        The data to be used to evaluate the subgroups; bootstrapping works by drawing samples from this data using replacement.
    metric_function: function
        The function to use to evaluate the how well the subgroup is working on an individual sample. Must have the following parameters: (sample, subgroup_sample), where `sample` is a sample of the data, and `subgroup_sample` is the same but filtered to only include subgroup members. 
    aggregation_function: function, optional
        Used to aggregate across the samples. If not provided, the full list of scores (calculated by metric_function) over all the samples for each subgroup will be returned. 
    ignore_defaults: boolean, optional
        If True, then the first row in data will be treated as containing 'default values' to be ignored in the processing. 
    number_simulations: int, optional
        The number of samples to use. 
    frac: float, optional
        The size of each sample as a proportion of the length of data. Can be reduced to decrease computational cost. 
    replace: boolean, optional
        Set to False to override the default sampling with replacement strategy. 
    
    Returns
    -----------
    results
        A dict with subgroup names as keys and bootstrapping results as values, or, when using only one subgroup, just the bootstrapping results.
    aggregation
        A dict with subgroup names as keys and aggregated results as values (or an empty dict if there is no aggregation function), or when using only one subgroup, just the aggregation (or None if there is no aggregation function).
    '''
    
    ## This code is quite ugly but that was needed to speed things up. 

    samples = []
    
    for x in range(number_simulations):
        
        sample = data.sample(frac=frac, replace=False)
        
        if replace:
            sample = sample.sample(frac=1, replace=True)
            
        if ignore_defaults:
            sample = sample.replace(data.iloc[0,:], np.NaN)
            
        samples.append(sample.reset_index(drop=True))
        
    selectors = set()
    
    for subgroup in subgroups:
    
        for sel in subgroup.selectors:
        
            selectors.add(sel)
            
    selectors = list(selectors)
        
    if isinstance(subgroups, sd4py.PySubgroupResults):
        
        subgroups = subgroups.subgroups
    
    if isinstance(subgroups, list):
        
        def sample_indices(sample):
            
            def get_indices(sel):
            
                logical_indices = np.ones(sample.index.shape, dtype=bool)
            
                if isinstance(sel, sd4py.PyNumericSelector):
                    
                    if sel.include_lower_bound and sel.lower_bound != float("-inf"):
                        np.logical_and(logical_indices, sample[sel.attribute].values >= sel.lower_bound, out = logical_indices)  ## It's about x10 faster to use .values (i.e. numpy arrays and therefore numpy functions)
                    elif sel.lower_bound != float("-inf"):
                        np.logical_and(logical_indices, sample[sel.attribute].values > sel.lower_bound, out = logical_indices)
                    if sel.include_upper_bound and sel.upper_bound != float("inf"):
                        np.logical_and(logical_indices, sample[sel.attribute].values <= sel.upper_bound, out = logical_indices)
                    elif sel.upper_bound != float("inf"):
                        np.logical_and(logical_indices, sample[sel.attribute].values < sel.upper_bound, out = logical_indices)
                 
                if isinstance(sel, sd4py.PyNominalSelector):
                    
                    np.logical_and(logical_indices, sample[sel.attribute].astype(str).values == sel.value, out = logical_indices)
                    
                return logical_indices
            
            return dict(zip(map(str, selectors) , map(get_indices, selectors)))
        
        samples_indices = dict(zip(range(number_simulations), map(sample_indices, samples)))

        
        def process_subgroup(subgroup):
            
            def get_metric_values(args):
                
                idx, sample = args
                
                logical_indices = np.ones(sample.index.shape, dtype=bool)
                
                for sel in subgroup.selectors:
                    
                    np.logical_and(logical_indices, samples_indices[idx][str(sel)], out = logical_indices)
                    
                subgroup_sample = sample[logical_indices]

                return metric_function(sample, subgroup_sample)
            
            return list(map(get_metric_values, enumerate(samples)))
        
        results = dict(zip(map(str, subgroups), map(process_subgroup, subgroups)))
        
        
        if aggregation_function is not None:
                    
            aggregation = {key: aggregation_function(val) for key, val in results.items()}
        
        
        return results, aggregation
    
    else: ## Not a list 
    
        metric_values = []
        
        for sample in samples:
            
            subgroup_sample = subgroup.get_rows(sample)

            metric_values.append(metric_function(sample, subgroup_sample))
        
        if aggregation_function is None:

            return metric_values, None

        else:

            return metric_values, aggregation_function(metric_values)


def confidence_intervals(subgroups, data, ignore_defaults=False, number_simulations=100, frac=1/3, replace=True):
    '''
    Provides some estimate of variability of the target value for subgroups. Uses bootstrapping to achieve this. 
    The target value and the size of each subgroup is calculated across 100 samples of the data. The 0.05 and 0.95 quantiles are returned per subgroup. 
    For numeric target variables, the mean within subgroup members is used; for nominal targets, the proportion of subgroup members belonging to the 'positive' class is used. 
    
    Parameters
    ----------------
    subgroups: list of PySubgroup objects
        The subgroup(s) for which to estimate confidence intervals.
    data: DataFrame
        The data to be used to evaluate the subgroups; bootstrapping works by drawing samples from this data using replacement.
    ignore_defaults: boolean, optional
        If True, then the first row in data will be treated as containing 'default values' to be ignored in the processing. 
    number_simulations: int, optional
        The number of samples to use. 
    frac: float, optional
        The size of each sample as a proportion of the length of data. Can be reduced to decrease computational cost. 
    replace: boolean, optional
        Set to False to override the default sampling with replacement strategy. 

    Returns
    -----------
    bootstrapping_results: dict
        A dict with subgroup names as keys and bootstrapping results as values, or, when using only one subgroup, just the bootstrapping results.
    confidence_intervals: DataFrame
        A DataFrame with the estimated confidence intervals, indexed by subgroup name. 
    '''
    
    target = subgroups.target
    
    if data.loc[:,target].dtype == 'object' or data.loc[:,target].dtype == 'bool' or data.loc[:,target].dtype.name == 'category': ## if nominal
            
        def metric_function(sample, subgroup_sample):
            
            subgroup_sample = subgroup_sample.loc[:,target]
            sample = sample.loc[:,target]
            
            population_share = subgroup_sample.count() / sample.count()
            target_proportion = subgroup_sample.eq(subgroups.target_value).sum() / subgroup_sample.count()  ## what proportion of values is equal to the target value
            
            return population_share, target_proportion
        
    else: ## if numeric
        
        def metric_function(sample, subgroup_sample):
            
            subgroup_sample = subgroup_sample.loc[:,target]
            sample = sample.loc[:,target]
            
            population_share = subgroup_sample.count() / sample.count()
            average = subgroup_sample.mean()
            
            return population_share, average
        
    def aggregation_function(subgroup_values):
        
        out = {
            'proportion_lower': np.nanquantile([val[0] for val in subgroup_values], 0.05),
            'proportion_upper': np.nanquantile([val[0] for val in subgroup_values], 0.95),
            'target_lower': np.nanquantile([val[1] for val in subgroup_values], 0.05),
            'target_upper': np.nanquantile([val[1] for val in subgroup_values], 0.95)
        }
        
        return out 
    
    bootstrapping_results, confidence_intervals = bootstrapping(subgroups, data, metric_function, aggregation_function,  
                ignore_defaults=ignore_defaults, number_simulations=number_simulations, frac=frac, replace=replace)
    
    return bootstrapping_results, pd.DataFrame({'pattern':str(subgroup), **values} for subgroup, values in confidence_intervals.items())


def confidence_intervals_to_boxplots(bootstrapping_results_list, labels):
    '''
    Takes the outputs of the `confidence_intervals` function and creates a boxplot showing the distribution of the target value, 
    with the width of boxes indicating the relative sizes of the subgroups on average. 
    
    Parameters
    ----------------
    bootstrapping_results_list: list
        A list with subgroup bootstrapping results as values. 
    labels: list
        The label to use for each subgroup. 

    Returns
    -----------
    fig: Figure
        The matplotlib Figure of the boxplots
    '''
    
    averages = np.stack([np.array(x)[:,1] for x in bootstrapping_results_list])

    for idx, row in enumerate(averages): 
        averages[idx][np.isnan(row)] = row[~np.isnan(row)].mean() # remove nan

    widths = [np.array(x)[:,0].mean() for x in bootstrapping_results_list]
    widths = 0.9 * np.array(widths) / np.max(widths)  ## Box thickness relative to the maximum shown. Adjusted by 0.9 to avoid overlap

    plt.boxplot(averages.T, vert=False, widths=widths, labels=labels)
    plt.gca().xaxis.grid(True, linestyle='--')
    
    return plt.gcf()


def confidence_precision_recall_f1(subgroups, data, ignore_defaults=False, number_simulations=100, frac=1/3, replace=True):
    '''
    Used to provide an estimate of how variable the performance of each subgroup is. 
    Applies to nominal variables, where the precision, recall and $F_1$ score are used to quantify how well a subgroup performs. 
    
    Parameters
    ----------------
    subgroups: list of PySubgroup objects
        The subgroup(s) for which to estimate confidence intervals.
    data: DataFrame
        The data to be used to evaluate the subgroups; bootstrapping works by drawing samples from this data using replacement.
    ignore_defaults: boolean, optional
        If True, then the first row in data will be treated as containing 'default values' to be ignored in the processing. 
    number_simulations: int, optional
        The number of samples to use. 
    frac: float, optional
        The size of each sample as a proportion of the length of data. Can be reduced to decrease computational cost. 
    replace: boolean, optional
        Set to False to override the default sampling with replacement strategy. 

    Returns
    -----------
    bootstrapping_results: dict
        A dict with subgroup names as keys and bootstrapping results as values, or, when using only one subgroup, just the bootstrapping results.
    precision_recall_f1: DataFrame
        A DataFrame with the estimated confidence intervals (0.05 and 0.95 quantiles from bootstrapping) on each of precision, recall and $F_1$, indexed by subgroup name. 
    '''
    
    target = subgroups.target
    target_value = subgroups.target_value
    
    def metric_function(sample, subgroup_sample):

        subgroup_sample = subgroup_sample.loc[:,target]
        sample = sample.loc[:,target]

        precision = subgroup_sample.values.__eq__(target_value).sum() / subgroup_sample.count()    ## Use numpy arrays to check for equality since they're much faster 
        recall = subgroup_sample.values.__eq__(target_value).sum() / sample.values.__eq__(target_value).sum()    ## Use numpy arrays to check for equality since they're much faster 
        f1 = (2 * precision * recall) / (precision + recall)

        return precision, recall, f1
    
    def aggregation_function(subgroup_values):
        
        out = {
            'precision_lower': np.nanquantile([val[0] for val in subgroup_values], 0.05),
            'precision_upper': np.nanquantile([val[0] for val in subgroup_values], 0.95),
            'recall_lower': np.nanquantile([val[1] for val in subgroup_values], 0.05),
            'recall_upper': np.nanquantile([val[1] for val in subgroup_values], 0.95),
            'f1_lower': np.nanquantile([val[2] for val in subgroup_values], 0.05),
            'f1_upper': np.nanquantile([val[2] for val in subgroup_values], 0.95)
        }
        
        return out 
    
    bootstrapping_results, aggregation = bootstrapping(subgroups, data, metric_function, aggregation_function, 
                ignore_defaults=ignore_defaults, number_simulations=number_simulations, frac=frac, replace=replace)
    
    return bootstrapping_results, pd.DataFrame({'pattern':str(subgroup), **values} for subgroup, values in aggregation.items())


def corrected_hedges_g(sample1, sample2):
    '''
    Estimates the effect size between two samples of a numeric variable. 
    This is the corrected Hedge's G; see <https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm>. 
    
    Parameters
    ----------------
    sample1: array
        The first sample of values. 
    sample2: array
        The second sample of values. 

    Returns
    -----------
    corrected_hedges_g: float
        The estimated effect size. 
    '''
    
    n_1 = sample1.count()
    n_2 = sample2.count()
    
    pooled_sd = np.sqrt((((n_1-1) * sample1.var()) + ((n_2-1) * sample2.var())) / (n_1 + n_2 - 2))
    
    n =  n_1 + n_2
    bias_correction = ((n-3)/(n-2.25)) * np.sqrt((n - 2) / n)
    
    return bias_correction * (sample1.mean() - sample2.mean()) / pooled_sd


def confidence_hedges_g(subgroups, data, ignore_defaults=False, number_simulations=100, frac=1/3, replace=True):
    '''
    Used to provide an estimate of the effect size for different subgroups when the target variable is numeric. 
    
    Parameters
    ----------------
    subgroups: list of PySubgroup objects
        The subgroup(s) for which to estimate confidence intervals.
    data: DataFrame
        The data to be used to evaluate the subgroups; bootstrapping works by drawing samples from this data using replacement.
    target: string
        The name of the target variable. 
    value: object, optional
        For nominal target variables only. The value of the target variable that counts as the 'positive' class. 
    ignore_defaults: boolean, optional
        If True, then the first row in data will be treated as containing 'default values' to be ignored in the processing. 
    number_simulations: int, optional
        The number of samples to use. 
    frac: float, optional
        The size of each sample as a proportion of the length of data. Can be reduced to decrease computational cost. 
    replace: boolean, optional
        Set to False to override the default sampling with replacement strategy. 

    Returns
    -----------
    bootstrapping_results: dict
        A dict with subgroup names as keys and bootstrapping results as values, or, when using only one subgroup, just the bootstrapping results.
    confidence_hedges_g: DataFrame
        A DataFrame with the estimated confidence intervals (0.05 and 0.95 quantiles from bootstrapping) on the effect size, indexed by subgroup name. 
    '''
    
    target = subgroups.target
    
    def metric_function(sample, subgroup_sample):

        subgroup_sample = subgroup_sample.loc[:,target]
        sample = sample.loc[:,target]
        complement = sample[~sample.index.isin(subgroup_sample.index)]

        proportion = subgroup_sample.count() / sample.count()
        hedges_g = corrected_hedges_g(subgroup_sample, complement)

        return proportion, hedges_g
    
    def aggregation_function(subgroup_values):
        
        out = {
            'proportion_lower': np.nanquantile([val[0] for val in subgroup_values], 0.05),
            'proportion_upper': np.nanquantile([val[0] for val in subgroup_values], 0.95),
            'hedges_g_lower': np.nanquantile([val[1] for val in subgroup_values], 0.05),
            'hedges_g_upper': np.nanquantile([val[1] for val in subgroup_values], 0.95)
        }
        
        return out 
    
    bootstrapping_results, aggregation = bootstrapping(subgroups, data, metric_function, aggregation_function, 
                ignore_defaults=ignore_defaults, number_simulations=number_simulations, frac=frac, replace=replace)
    
    return bootstrapping_results, pd.DataFrame({'pattern':str(subgroup), **values} for subgroup, values in aggregation.items())


def odds_ratio_ci(sample1, sample2):
    '''
    Estimates the effect size between two samples of a binary nominal variable. 
    This is the odds ratio, which allows us to estimate confidence intervals directly from the confusion matrix. 
    
    Parameters
    ----------------
    sample1: array
        The first sample of values. 
    sample2: array
        The second sample of values. 

    Returns
    -----------
    odds_ratio: float
        The estimated effect size. 
    lower: float
        Lower confidence interval on the estimated effect size. 
    upper: float
        Upper confidence interval on the estimated effect size. 
    '''
    
    a = sample1.eq(True).sum() # subgroup == True and column == value
    b = sample1.eq(False).sum() # subgroup == True and column != value
    c = sample2.eq(True).sum() # subgroup == False and column == value
    d = sample2.eq(False).sum()  # subgroup == False and column != value
    
    if min(a,b,c,d) == 0:
        
        return np.NaN, np.NaN, np.NaN
    
    odds_ratio = (a * d) / (b * c)
    
    lower = np.exp(np.log(odds_ratio) - (1.96 * np.sqrt((1/a) + (1/b) + (1/c) + (1/d))))
    upper = np.exp(np.log(odds_ratio) + (1.96 * np.sqrt((1/a) + (1/b) + (1/c) + (1/d))))
    
    return odds_ratio, lower, upper


def find_interesting_columns(subgroup, data, use_complement = True, ignore_defaults = False, columns_to_ignore=[]):
    '''
    Makes it easier to find 'interesting' columns for particular subgroup by returning the estimated effect size for each variable in the dataset 
    (i.e., if a variable has a large effect size then the subgroup is extreme with respect to that variable).
    Provides both interesting numeric and interesting nominal columns. 
    Corrected Hedge's G is used to estimate effect size on numeric variables, and the odds ratio (and its confidence intervals) is used for nominal variables. 
    
    Parameters
    ----------------
    subgroup: PySubgroup
        The subgroup for which to find interesting columns.
    data: DataFrame
        The data to be used to evaluate the subgroups; bootstrapping works by drawing samples from this data using replacement.
    use_complement: boolean, optional
        If True, subgroup members will be compared to non-subgroup members. Otherwise, subgroup members will be compared to the full dataset (including subgroup members).  
    ignore_defaults: boolean, optional
        If True, then the first row in data will be treated as containing 'default values' to be ignored in the processing. 
    columns_to_ignore: list, optional
        A list of columns to ignore, for example these could be the target variable and/or selector variables since they are already known to be interesting. 

    Returns
    -----------
    numeric_columns: dict
        A dictionary with variable names as keys and estimated effect sizes as values. 
    nominal_columns: dict
        A dictionary with variable names as keys and estimated effect sizes as values, for nominal variables. 
    '''
    
    numeric_columns = {}
    nominal_columns = {}
    
    subgroup_indices = subgroup.get_indices(data)
        
    if ignore_defaults:

        data = data.replace(data.iloc[0,:], np.NaN)
    
    for column in data:
        
        if column in columns_to_ignore:
            
            continue
        
        column = data[column]
        
        if (np.issubdtype(column.dtype, np.datetime64) or np.issubdtype(column.dtype, np.timedelta64)):  ## these need to be converted to a straightforward numeric format
            
            column = pd.to_numeric(column) 
            column = (column - column.mean()) / column.std() ## Just to get to a reasonable timescale, otherwise it's nanoseconds or something like that
        
        subgroup_rows = column.loc[subgroup_indices]

        if use_complement:
            population_rows = column.drop(subgroup_indices, axis=0)
        else:
            population_rows = column
        
        if column.dtype == 'object' or column.dtype == 'bool' or column.dtype.name == 'category':  ## nominals

            vals, counts = np.unique(column, return_counts=True)

            for value in vals[np.argsort(-counts)][:5]: ## 5 most common values for this variable; each providing a feature-value pair to investigate
                
                nominal_columns[(column.name, value)] = odds_ratio_ci(subgroup_rows == value, population_rows == value)

        else:   ## numerics
            
            numeric_columns[column.name] = corrected_hedges_g(subgroup_rows, population_rows)

    return numeric_columns, nominal_columns 


def most_interesting_columns(subgroup, data, columns_to_ignore=[]):
    '''
    To support visualisation of a single subgroup, uses the `find_interesting_columns` function to pick the 10 most numeric and 10 most interesting nominal values for a subgroup.
    Corrected Hedge's G is used to estimate effect size on numeric variables, and lower confidence on the odds ratio is used for nominal variables.
    
    
    Parameters
    ----------------
    subgroup: PySubgroup
        The subgroup for which to find interesting columns.
    data: DataFrame
        The data to be used to evaluate the subgroups; bootstrapping works by drawing samples from this data using replacement.
    columns_to_ignore: list, optional
        A list of columns to ignore, for example these could be the target variable and/or selector variables since they are already known to be interesting. 

    Returns
    -----------
    most_interesting_numeric: DataFrame
        A pandas DataFrame with variable names as index and estimated effect sizes as values. 
    most_interesting_nominal: DataFrame
        A pandas DataFrame with variable names as index and estimated effect sizes as values, for nominal variables. 
    '''
    
    interesting_numeric, interesting_nominal = find_interesting_columns(subgroup, data, columns_to_ignore=columns_to_ignore)
    
    interesting_numeric = pd.DataFrame(interesting_numeric.values(), index=interesting_numeric.keys())
    interesting_numeric = interesting_numeric.dropna()
    interesting_nominal = pd.DataFrame(interesting_nominal.values(), index=interesting_nominal.keys())
    interesting_nominal = interesting_nominal.dropna()
    
    if len(interesting_numeric) > 0:

        interesting_numeric = interesting_numeric.dropna()
        most_interesting_numeric = interesting_numeric.iloc[interesting_numeric[0].abs().argsort()][::-1][0].iloc[:10] ## Find the 10 most interesting by effect size

    else:
        
        most_interesting_numeric = interesting_numeric
    
    if len(interesting_nominal) > 0:
        
        max_lower = interesting_nominal.loc[interesting_nominal[1].abs().groupby(level=0).idxmax()][1] ## Maximum lower confidence interval
        max_lower = max_lower.iloc[max_lower.abs().argsort()][::-1]
        max_lower

        min_upper = interesting_nominal.loc[interesting_nominal[2].abs().groupby(level=0).idxmin()][2]
        min_upper = min_upper[interesting_nominal.groupby(level=0).count()[2].values > 2]
        min_upper = min_upper[min_upper > 0]
        min_upper = (1 / min_upper)
        min_upper = min_upper.iloc[min_upper.abs().argsort()][::-1]
        min_upper

        most_interesting_nominal = pd.concat([max_lower, min_upper]).sort_values(ascending=False).iloc[:10] ## 10 most interesting by having especially high or especially low odds ratio
    
    else:
        
        most_interesting_nominal = interesting_nominal
    
    return most_interesting_numeric, most_interesting_nominal


def radar_plot(data, prop_scale=3, subplot=111, text_size = 10, axis_padding = 15, ymins = None, ymaxes = None):
    '''
    Creates a custom radar plot, where axis names and units can vary. Note that radar plots are poorly-supported by matplotlib and things like tight_layout will not work. 
    
    Parameters
    ----------------
    data: DataFrame
        A dataframe where columns are variables and rows are the groups (each group will become a polygon).
    prop_scale: float, optional
        Used to control the size of the innermost circle (where axes begin) compared to the rest of the plot. 
    subplot: int, optional
        Used to determine which subplot to draw the radar plot onto.  
    text_size: int, optional
        Used to modify the text size of axis labels. 
    axis_padding: int, optional
        Used to modify the padding around axis names (to prevent them overlapping with axis tick labels). Modified by position, so more horizontal axes get more padding (since the names are more likely to overlap with the ticks). 
    ymins: list, optional
        Used to set the beginning of each axis
    ymaxes: list, optional
        Used to set the end of each axis

    Returns
    -----------
    ax: Axis
        The matplotlib Axis of the radar plot
    '''
    
    num_variables = len(data.columns)  ## Number of columns/variables
    
    if num_variables < 3:
        
        num_variables = 3  ## So that we always have a shape with an area 
    
    angles = [n / float(num_variables) * 2 * np.pi for n in range(num_variables)]
    angles += angles[:1] # And back to the first position

    # Initialise the radar plot
    ax = plt.subplot(subplot, polar=True)

    # To put the first axis on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axis per variable + add names
    x_ticks = plt.xticks(angles[:len(data.columns)], data.columns, size=text_size+1)
    
    ax.spines['polar'].set_color('grey')
    
    ax.yaxis.set_visible(False) # Axes and appropriate scales will be drawn later, using polar_twin
    
    ax.grid(False) # Axes and appropriate scales will be drawn later, using polar_twin
    
    #ax.tick_params(axis='x', which='major', pad=axis_padding)          #Space the axis labels a bit
    
    for idx, x_tick in enumerate(x_ticks[0]):

        x_tick.set_pad((axis_padding//5) + (axis_padding * np.abs(np.sin(angles[idx]) ** 2)))  ## This seems to give slightly better padding than the previous attempt
    
    def polar_twin(ax, ymin, ymax, angle=20): # function to make a new axis with appropriate tick marks
        
        ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                                 label='twin', frameon=False,
                                 theta_direction=ax.get_theta_direction(),
                                 theta_offset=ax.get_theta_offset())
        ax2.xaxis.set_visible(False)
        
        #labels = [ymin, ymax]
        
        labels = [ymin, ymin + ((ymax-ymin) * 1/3), ymin + ((ymax-ymin) * 2/3), ymax]

        if hasattr(ymin, 'strftime'):
            labels = [item.strftime('%Y-%m-%d\n%H:%M:%S') for item in labels]
        else:
            try:
                labels = ["{:.2f}".format(float(item)) for item in labels]

            except:
                labels = ["{0} days\n{1:02d}:{2:02d}:{3:02d}".format(*item.components) for item in labels]

        ax2.set_ylim(0, 1+prop_scale)
        ax2.set_rgrids([1,1+(prop_scale/3),1+(2*prop_scale/3),1+prop_scale], labels, angle, size=text_size, ha="center", va="center")

        # To ensure that the original axes tick labels are on top of
        # whatever is plotted in the twinned axes. Tick labels will be drawn twice.
        for label in ax.get_yticklabels():
            ax.figure.texts.append(label)
        
        ax2.grid(False)
        
        return ax2
    
    if ymins is None:
        ymins = data.min()

    if ymaxes is None:
        ymaxes = data.max()
    
    for idx, colname in enumerate(data):
        
        col = data.loc[:,colname]
        
        ymin = ymins[colname]
        ymax = ymaxes[colname]
        
        angle = idx * 360 / num_variables
        
        ax_latest = polar_twin(ax, ymin, ymax, angle)
        
        ax_latest.set_zorder(100)  # so axis grid doesn't appear in front of other content
    
    ax_latest.grid(True)
    ax_latest.set_zorder(10)  # so axis grid doesn't appear in front of other content
    
    def plot_polygon(row, angles, colour, label, linestyle):

        # Draws the polygon for one subgroup onto the radar plot 
        
        values=row.flatten().tolist()
        if len(values) < 3:
            values += np.ones(3 - len(values)).tolist()
        values += values[:1]  ## To go back to the start
        ax.set_ylim(0, 1+prop_scale)
        
        ax.plot(angles, values, linewidth=2, linestyle=linestyle, color=colour, label=label)
        ax.fill(angles, values, colour, alpha=0.1)
    
    data_norm = 1 + (prop_scale * (data - ymins) / (ymaxes - ymins))   # Scale the data to match to the labels
    
    for idx, row in enumerate(data_norm.values):
        
        # We use standard 'tableau' colours from matplotlib, and varying linestyle
        
        plot_polygon(row, angles, list(mcolors.TABLEAU_COLORS)[idx % len(mcolors.TABLEAU_COLORS)], 
                     label=str(data_norm.index[idx]), linestyle=['solid','dashed','dotted','dashdot'][idx%4])
        
    # Draw a legend now that the polygons have been plotted
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    ax.set_zorder(100)  # so axis grid doesn't appear in front of other content
    ax.patch.set_visible(False)  # so axis grid doesn't appear in front of other content
    
    return ax


def subgroup_overview(subgroup, selection_data, visualisation_data=None, use_complement=True, axis_padding = 15):
    '''
    Creates a four-panel matplotlib visualisation for a single subgroup. 
    From left to right, top to bottom, this shows: 
    (i) the distribution of target values for the subgroup and its complement, 
    (ii) the selector variable average values, 
    (iii) average values for other numeric variables, and 
    (iv) relative frequency of certain (variable, value) pairs for other nominal variables.} 
    Note that radar plots are poorly-supported by matplotlib and things like tight_layout will not work. 
    
    Parameters
    ----------------
    subgroup: PySubgroup
        The subgroup to be visualised.
    selection_data: DataFrame
        The subgroup will be applied to this data, to select subgroup members. From this, the most interesting columns to visualise will be chosen. If visualisation_data is not provided, this will also be the data used to compute the values that are visualised.
    visualisation_data: DataFrame
        If desired, a second dataset can be used to provide the data that is visualised (but not used to select the 'most interesting columns'). 
    use_complement: boolean, optional
        If True, subgroup members will be compared to non-subgroup members. Otherwise, subgroup members will be compared to the full dataset (including subgroup members).  

    Returns
    -----------
    fig: Figure
        The matplotlib Figure of the subgroup overview.
    '''
    
    target = subgroup.target
    
    if visualisation_data is None:
        
        visualisation_data = selection_data
    
    def visualise_columns(numeric_columns=None, nominal_columns=None, nominal_values=None, prop_scale=2.5, subplot=111):
        
        ## This function finds appropriate ymins and ymaxes for plotting each axis, and then calls the radar_plot function
        
        means = pd.DataFrame()
        proportions = pd.DataFrame()
        numeric_ymins = pd.Series(dtype=object)
        numeric_ymaxes = pd.Series(dtype=object)
        nominal_ymins = pd.Series(dtype=object)
        nominal_ymaxes = pd.Series(dtype=object)
        
        subgroup_indices = subgroup.get_indices(visualisation_data)
        
        ## Numerics
        
        if numeric_columns is not None:

            subgroup_means = visualisation_data.loc[subgroup_indices][numeric_columns].mean(numeric_only=False)

            if use_complement:
                means = pd.concat([
                            subgroup_means, 
                            visualisation_data.drop(subgroup_indices, axis=0)[numeric_columns].mean(numeric_only=False)
                        ], axis=1).T.set_index([['Subgroup', 'Complement']])

            else:
                means = pd.concat([
                            subgroup_means, 
                            visualisation_data[numeric_columns].mean(numeric_only=False)
                        ], axis=1).T.set_index([['Subgroup', 'Population']])

            vis_data_numerics = visualisation_data[numeric_columns]

            numeric_ymins = vis_data_numerics.mean(numeric_only=False) - (vis_data_numerics.std(numeric_only=False))

            numeric_ymins = pd.concat([
                numeric_ymins,
                subgroup_means
            ],axis=1).T.min(numeric_only=False)  ## Minimum of (complement - 1 std) and (subgroup_mean)

            numeric_ymaxes = vis_data_numerics.mean(numeric_only=False) + (vis_data_numerics.std(numeric_only=False))

            numeric_ymaxes = pd.concat([
                numeric_ymaxes,
                subgroup_means
            ],axis=1).T.max(numeric_only=False) ## Maximum of (complement + 1 std) and (subgroup_mean)

        ## Now the nominals
        
        if nominal_columns is not None:
            
            nominal_data = visualisation_data.loc[:,nominal_columns].astype(str)

            subgroup_proportions = nominal_data.loc[subgroup_indices, :].eq(nominal_values).sum() / nominal_data.loc[subgroup_indices, :].count()
            
            if use_complement:

                proportions = pd.concat([
                            subgroup_proportions, 
                            nominal_data.drop(subgroup_indices, axis=0).eq(nominal_values).sum()  / nominal_data.drop(subgroup_indices, axis=0).count()
                        ], axis=1).T.set_index([['Subgroup', 'Complement']])

            else:

                proportions = pd.concat([
                            subgroup_proportions, 
                            nominal_data.eq(nominal_values).sum() / nominal_data.count()
                        ], axis=1).T.set_index([['Subgroup', 'Population']])

            nominal_ymins = ((2* proportions) - 1).min() ## same distance below the proportion as above it (up to 1), minimum across subgroup and complement
            nominal_ymins = pd.concat([nominal_ymins, pd.Series(0, index=nominal_ymins.index)], axis=1).T.max() ## set to zero if currently below zero

            nominal_ymaxes = (2* proportions).max() ## same distance above the proportion as below it (down to 1), maximum across subgroup and complement
            nominal_ymaxes = pd.concat([nominal_ymaxes, pd.Series(1, index=nominal_ymaxes.index)], axis=1).T.min() ## set to 1 if currently above 1

            nominal_ymins.index = ["{0} == {1}".format(*x) for x in zip(nominal_columns, nominal_values)]
            nominal_ymaxes.index = ["{0} == {1}".format(*x) for x in zip(nominal_columns, nominal_values)]
            proportions.columns = ["{0} == {1}".format(*x) for x in zip(nominal_columns, nominal_values)]

        total = pd.concat([means, proportions], axis=1)
        ymins = pd.concat([numeric_ymins, nominal_ymins])
        ymaxes = pd.concat([numeric_ymaxes, nominal_ymaxes])
        
        return radar_plot(total, prop_scale=prop_scale, ymins=ymins, ymaxes=ymaxes, subplot=subplot, axis_padding=axis_padding)
        
    ## Target
    
    ax = plt.subplot(221)
    subgroup_indices = subgroup.get_indices(visualisation_data)
    
    if visualisation_data[target].dtype == 'object' or visualisation_data[target].dtype == 'bool' or visualisation_data[target].dtype.name == 'category':
        
        ## For nominal target, use a stacked barchart to visualise distribution

        if use_complement:
            pd.concat([
                pd.Series(*np.unique(visualisation_data.loc[subgroup_indices][target], return_counts=True)[::-1], name='Subgroup') \
                                / visualisation_data.loc[subgroup_indices][target].count(),
                pd.Series(*np.unique(visualisation_data.drop(subgroup_indices, axis=0)[target], return_counts=True)[::-1], name='Complement') \
                                / visualisation_data.drop(subgroup_indices, axis=0)[target].count()
            ],axis=1).T.plot(kind='barh', stacked=True, cmap=plt.get_cmap('Set2'), ax=ax)
                        
            for container in ax.containers:
                ax.bar_label(container, label_type='center', fmt="%.2f")
            ax.legend()
        
        else:
            pd.concat([
                pd.Series(*np.unique(visualisation_data.loc[subgroup_indices][target], return_counts=True)[::-1], name='Subgroup') \
                                / visualisation_data.loc[subgroup_indices][target].count(),
                pd.Series(*np.unique(visualisation_data[target], return_counts=True)[::-1], name='Complement') \
                                / visualisation_data[target].count()
            ],axis=1).T.plot(kind='barh', stacked=True, cmap=plt.get_cmap('Set2'), ax=ax)
            
            for container in ax.containers:
                ax.bar_label(container, label_type='center', fmt="%.2f")
            ax.legend() 
            
    else:
        
        ## For numeric target, use an estimated probability density plot
        
        if use_complement:
            sns.kdeplot(visualisation_data.loc[subgroup_indices][target], linewidth=2, label='Subgroup')
            sns.kdeplot(visualisation_data.drop(subgroup_indices, axis=0)[target], linewidth=2, label='Complement', linestyle='dashed')
            ax.legend()
        
        else:
            sns.kdeplot(visualisation_data.loc[subgroup_indices][target], linewidth=2, label='Subgroup')
            sns.kdeplot(visualisation_data[target], linewidth=2, label='Population', linestyle='dashed')
            ax.legend()
            
    ax.set_title('Target', pad =20)
    
    ## Selectors
    
    numeric_selectors = []
    nominal_selectors = []
    nominal_selector_values = []
    
    for selector in subgroup.selectors:
        
        if isinstance(selector, sd4py.PyNumericSelector):
            
            numeric_selectors.append(selector.attribute)
        
        else:
            
            nominal_selectors.append(selector.attribute)
            nominal_selector_values.append(str(selector.value))
    
    if len(numeric_selectors) == 0:
        numeric_selectors = None
    
    if len(nominal_selectors) == 0:
        nominal_selectors = None
    
    ax = visualise_columns(numeric_columns=numeric_selectors, nominal_columns=nominal_selectors, nominal_values=nominal_selector_values, subplot=222)
    ax.set_title('Selectors', pad =20)
    
    
    ## Additional variables
    
    columns_to_ignore = [s.attribute for s in subgroup.selectors] ## Selectors will already be visualised
    columns_to_ignore += [target] ## Target will already be visualised
    
    most_interesting_numeric, most_interesting_nominal = most_interesting_columns(subgroup, selection_data, columns_to_ignore=columns_to_ignore)
    
    ## Numeric
    
    if len(most_interesting_numeric) > 0:
        
        ax = visualise_columns(numeric_columns=most_interesting_numeric.index.tolist(), subplot=223)
        ax.set_title('Additional Numeric Variables', pad =20)
    
    ## Nominals
    
    if len(most_interesting_nominal) > 0:
    
        columns = [x[0] for x in most_interesting_nominal.index]
        values = [str(x[1]) for x in most_interesting_nominal.index]

        ax = visualise_columns(nominal_columns=columns, nominal_values=values, subplot=224)
        ax.set_title('Additional Nominal Variables', pad =20)
        
    return plt.gcf()


def jaccard_visualisation(subgroups, data, minimum_jaccard=0, labels=None):
    '''
    Shows the similarity between a selection of subgroups. Uses the Jaccard similarity between each pair of subgroups to construct edges in a network diagram. 
    
    Parameters
    ----------------
    subgroups: list of PySubgroup objects
        The subgroups to visualise.
    data: DataFrame
        The data to be used to evaluate the similarity between pairs of subgroups.
    minimum_jaccard: float
        An edge will only be drawn between two subgroups if their Jaccard similarity is above this value. 
    labels: list
        The label to use for each subgroup. 

    Returns
    -----------
    fig: Figure
        The matplotlib Figure of the boxplots
    '''
    
    if labels is None:
        
        labels = [str(sg) for sg in subgroups]
    
    adjacency = np.zeros((len(subgroups), len(subgroups)))

    for idx1, subgroup1 in enumerate(subgroups):

        for idx2, subgroup2 in enumerate(subgroups):

            if idx1 < idx2:

                indices1 = subgroup1.get_indices(data)
                indices2 = subgroup2.get_indices(data)

                adjacency[idx1, idx2] = indices1.intersection(indices2).size / indices1.union(indices2).size
                
    G = nx.from_numpy_matrix(adjacency * (adjacency > minimum_jaccard))

    G = nx.relabel_nodes(G, mapping={idx:sg for idx, sg in enumerate(labels)})

    pos = nx.spring_layout(G, seed=10)  # seed so the results are consistent each time

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.5)

    # edges
    nx.draw_networkx_edges(
        G, pos, alpha=0.2,
        width = [7.5 * x for x in nx.get_edge_attributes(G,'weight').values()]
    )

    # labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    
    return plt.gca()


def timeseries_pad(*series):
    '''
    Helper function to pad timeseries that are of different lengths so that they are all the same length. Inputs can be Series or DataFrame objects.
    Note that the original index of each timeseries will be dropped. Returns the padded timeseries as a list. 
    '''
    
    aligned_timeseries = []
    
    max_length = np.max([len(x) for x in series])
    
    for x in series:
        
        if len(x) < max_length:
            
            pre_pad = (max_length - len(x)) // 2
            post_pad = max_length - (len(x) + pre_pad)
            
            x = x.reset_index(drop=True).reindex(np.arange(-pre_pad,len(x)+post_pad))

        x = x.reset_index(drop=True)
        
        aligned_timeseries.append(x)
    
    return aligned_timeseries


def subgroup_background_rectangle(ax, subgroup, features, window_size):
    '''
    Helper function to draw background rectangles on a matplotlib plot to show when subgroup members have occurred over time. 
    '''
    
    member_indices = subgroup.get_indices(features)
    
    for i, index_value in enumerate(features.index):

        value = int(index_value in member_indices)
        start = (i * window_size) - 1/2 ## -1/2 so first point of the window is included in the rectangle 
        end = ((i + 1) * window_size) - 1/2  ## -1/2, otherwise an extra point is included in the rectangle 

        tmp = plt.axvspan(xmin=start, xmax=end, color='tab:red', lw=0, alpha=value / 4)


def time_plot(subgroup, features, *series, window_size, use_start = False):
    '''
    Suitable when subgroup discovery was used to analyse windows of timeseries. 
    Shows multiple variables over time, and which windows of the time correspond to subgroup members (from data in the format originally used to discover subgroups).
    Windows that are subgroup members are represented by red rectangles in the background. 
    An arbitrary number of other variables are also visualised, and they can have a different sampling frequency to the data originally used to discover subgroups. 
    In summary, this shows how multiple variables progress over time, inside and outside of windows that are subgroup members. 
    
    Parameters
    ----------------
    subgroup: PySubgroup object
        The subgroup to visualise.
    features: DataFrame
        The data to be used to determine which windows of time are subgroup members. Should be (a subselection of) the data used first to discover subgroups. 
    n objects containing timeseries: Series or DataFrame objects
        Multiple arguments can be passed; each one should be a Series or DataFrame object with the same sampling frequency and duration as each other.
    timestep_delta: Timedelta
        Used to label the x-axis. Should be the Timedelta corresponding to one step in the x direction. 

    Returns
    -----------
    fig: Figure
        The matplotlib Figure showing multiple variables over time.
    '''
    
    timestep_timedelta = series[0].index[1] - series[0].index[0]
    
    start = series[0].index[0]
    
    def get_xtick_formatter_function(largest_timedelta, smallest_timedelta):
        
        ## Matplotlib does not format Timedeltas well by default, so use this custom function to get xtick labels

        largest_component = np.arange(7)[np.array(largest_timedelta.components, dtype=bool)][0]
        smallest_component = np.arange(7)[np.array(smallest_timedelta.components, dtype=bool)][-1]

        format_string_pieces = []
        components_used = []

        if largest_component == 0:

            format_string_pieces.append('{}d ')
            components_used.append(0)

        if largest_component < 4:

            format_string_pieces.append('{:02}:{:02}:{:02}')
            components_used.extend([1,2,3])

        if largest_component >= 4:

            format_string_pieces.append('0')

        if smallest_component >= 4:

            format_string_pieces.append('.{:03}')
            components_used.append(4)

        if smallest_component >= 5:

            format_string_pieces.append('{:03}')
            components_used.append(5)

        if smallest_component >= 6:

            format_string_pieces.append('{:03}')
            components_used.append(6)

        format_string = ''.join(format_string_pieces)

        return lambda x: format_string.format(*np.array(x.components)[components_used])
    
    assert [len(x) for x in series].count(len(series[0])) == len(series), "Input series are not the same length" ## Check all input series are the same length

    num_subplots = len(series)
    if not use_start:
        smallest_timedelta = timestep_timedelta
        largest_timedelta = timestep_timedelta * len(series[0])
        formatter = get_xtick_formatter_function(largest_timedelta, smallest_timedelta)
    
    for idx, series in enumerate(series):
        
        ax = plt.subplot(num_subplots,1,idx+1)
        
        if timestep_timedelta is not None:
            tmp = ax.plot(series.values)
        else:
            tmp = ax.plot(series)

        if isinstance(series, pd.DataFrame):
            name = series.columns.name  ## This can be set with df.rename_axis('New Name', axis=1)
        else:
            name = series.name
            
        ax.annotate(name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 10, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=14, ha='right', va='center')
                    
                    
        ax.set_xticks(ax.get_xticks().tolist()[1:-1])  # First and last xticks always seem irrelevant

        subgroup_background_rectangle(ax, subgroup, features, window_size)
        
        if use_start:
            ax.set_xticklabels([((x * timestep_timedelta) + start) for x in ax.get_xticks().tolist()], size=12)
        else:
            ax.set_xticklabels([formatter(x * timestep_timedelta) for x in ax.get_xticks().tolist()], size=12)
            
    return plt.gcf()


def first_k_non_overlapping(data, subgroups, jaccard_threshold, k=10):
    '''
    Finds the first k subgroups that do not overlap (i.e., do not have a Jaccard similarity above the threshold) with earlier subgroups. The orderring of the subgroups is of course important for determining the results. 
    
    Parameters
    ----------------
    data: DataFrame
        Data with which to evaluate the overlap between subgroups.
    subgroups: PySubgroupResults or list
        The subgroups. 
    jaccard_threshold: float
        The threshold for the Jaccard similarity that decides if a subgroup is overlapping with any subgroups already processed. The results will all have a Jaccard similarity with each other that is lower than this threshold. 
    k: int
        How many subgroups to look for. 

    Returns
    -----------
    non_overlapping: PySubgroupResults or list
        A PySubgroupResults object or list (whichever was passaed as an argument to `subgroups`) containing the first k subgroups where the Jaccard similarity between them is below the threshold provided. 
    '''

    non_overlapping = []

    if jaccard_threshold < 1.0:

        for idx1, sg1 in enumerate(subgroups):

            if len(non_overlapping) == 0:

                non_overlapping.append(idx1)

                continue

            overlapping = False
            
            indices1 = sg1.get_indices(data)

            for idx2 in non_overlapping:
                    
                indices2 = subgroups[idx2].get_indices(data)
                
                if (indices1.intersection(indices2).size / indices1.union(indices2).size) > jaccard_threshold:

                    overlapping = True

            if overlapping:

                continue
                
            non_overlapping.append(idx1)

            if len(non_overlapping) == k:

                return subgroups[non_overlapping]

        return subgroups[non_overlapping]
    
    return subgroups[:k]