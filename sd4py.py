'''
sd4py is a package that makes it easy to perform subgroup discovery on tabular data. It is extremely simple to use. Call the `sd4py.discover_subgroups()` function on a pandas dataframe and a collection of subgroups will be returned. 

This package provides a Python interface for using the Java application VIKAMINE. 

Subgroup discovery is based on finding patterns within some (explanatory) columns of data that then help to explain another (target) column of data. 
The goal of the subgroup discovery process will be to understand in what circumstances the target is extreme. With a numeric target, this means finding circumstances in which the value is exceptionally high (or exceptionally low) on average.
For a non-numeric target, this means looking for circumstances when a particular value is especially likely to occur.
One of the key benefits of this approach is that the outputs are interpretable, being expressed as a readable combination of rules like (e.g.)  "'Temperature'=high AND 'Pressure'=low". 

The package contains a `discover_subgroups()` function that finds subgroups based on a pandas `DataFrame` and a specifed target column. The package also includes custom python objects for holding the results. 
'''

import jpype
import jpype.imports
from jpype.types import *
jpype.startJVM(classpath=['vikamine_kernel.jar'])
import java.util.HashSet

from org.vikamine.kernel._examples import *
from org.vikamine.kernel.subgroup.selectors import *

import pandas as pd
import numpy as np

import copy

class PyOntology:
    '''
    Puts data into a Java `Ontology` object for use with the underlying Java subgroup discovery application. 
    
    It is not necessary to use this class explicitly; pandas dataframes will automatically be converted into a `PyOntology` object when passed into `discover_subgroups()`. 
    However, if the dataset is large, and subgroup discovery will be performed multiple times, then the user may opt to convert the dataset into a `PyOntology` to pass into `discover_subgroups()` for the sake of performance. 
    
    Attributes
    --------------
    The only attribute of the class is `ontology`, created during initialisation, which is bound to an `Ontology` object in the Java runtime. 
    '''
    
    def __init__(self, df):
        
        #self.df = df.copy(deep=False)
        #self.index = df.index.copy(deep=False) 
        self.column_names = df.columns.to_list()
        
        self.column_types = []
        self.datetime_columns = {}
        self.timedelta_columns = []
        
        numeric_arrays = []
        nominal_arrays = []
        
        for name, x in df.iteritems():
            
            if x.dtype == 'object' or x.dtype == 'bool' or x.dtype.name == 'category': # category depends on whether it's ordered?

                nominal_arrays.append(
                    JArray(JString)(x.astype(str))
                )
                self.column_types.append("nominal")
                
            elif np.issubdtype(x.dtype, np.datetime64):
                
                numeric_arrays.append(
                    JArray(JDouble)(pd.to_numeric(x))
                )
                self.column_types.append("numeric")
                self.datetime_columns[name] = x.dt.tz
                
            elif np.issubdtype(x.dtype, np.timedelta64): 
                
                numeric_arrays.append(
                    JArray(JDouble)(pd.to_numeric(x))
                )
                self.column_types.append("numeric")
                self.timedelta_columns.append(name)
                
            elif np.issubdtype(x.dtype, np.number):

                numeric_arrays.append(
                    JArray(JDouble)(pd.to_numeric(x))
                )
                self.column_types.append("numeric")

            else:

                raise ValueError("Unrecognised pandas dtype for :{0}".format(x))
                
        numeric_arrays = JArray(JArray(JDouble))(numeric_arrays)
        nominal_arrays = JArray(JArray(JString))(nominal_arrays)
        
        self.ontology = PythonOntologyCreator(JArray(JString)(self.column_names), 
                                              JArray(JString)(self.column_types),
                                              numeric_arrays,
                                              nominal_arrays).ontology


class PyNumericSelector:
    '''
    Represents a rule to select a subset of data, which combines with other selectors to form the subgroup/pattern definition. 
    The relevant attribute name in the data is stored in `attribute`. 
    This contains a `numeric lower_bound`, `upper_bound`, plus booleans `include_lower_bound` and `include_upper_bound` to decide whether border values are included in the selection. 
    
    Note that this is detached from the Java runtime, and so is a plain python object. 
    '''
    
    def __init__(self, attribute, lower_bound, upper_bound, include_lower_bound, include_upper_bound):
        
        self.attribute = attribute
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_lower_bound = include_lower_bound
        self.include_upper_bound = include_upper_bound
        
    def __str__(self):
        
        out_string = ""

        if isinstance(self.lower_bound, float) and isinstance(self.upper_bound, float):
            
            if self.lower_bound != float("-inf"):
                if self.include_lower_bound:
                    out_string += "{0:.2f} <= ".format(self.lower_bound)
                else:
                    out_string += "{0:.2f} < ".format(self.lower_bound)
            
            out_string += self.attribute
            
            if self.upper_bound != float("inf"):
                if self.include_upper_bound:
                    out_string += " <= {0:.2f}".format(self.upper_bound)
                else:
                    out_string += " < {0:.2f}".format(self.upper_bound)
        
            return out_string
        
        else: ## For datetimes and so on
            
            if self.lower_bound != float("-inf"):
                if self.include_lower_bound:
                    out_string += "{0} <= ".format(self.lower_bound)
                else:
                    out_string += "{0} < ".format(self.lower_bound)
            
            out_string += self.attribute
            
            if self.upper_bound != float("inf"):
                if self.include_upper_bound:
                    out_string += " <= {0}".format(self.upper_bound)
                else:
                    out_string += " < {0}".format(self.upper_bound)
            
            return out_string


class PyNominalSelector:
    '''
    Represents a rule to select a subset of data, which combines with other selectors to form the subgroup/pattern definition. 
    It indicates an attribute-value pair through `attribute` and `value`.
    
    Note that this is detached from the Java runtime, and so is a plain python object. 
    '''
    
    def __init__(self, attribute, value):
        
        self.attribute = attribute
        self.value = value
        
    def __str__(self):
        
        return "{0} = {1}".format(self.attribute, self.value)


class PySubgroup:
    '''
    Represents a subgroup in terms of its selectors, target evaluation, size and quality. 
    
    Note that this is detached from the Java runtime, and so is a plain python object. 
    
    Attributes
    --------------
    selectors: list
        A list of `PySelector` objects representing the rules constituting the subgroup/pattern. 
    target_evaluation: float
        The value of the target variable for this subgroup (when evaluated against the dataset originally used for subgroup discovery). 
    size: int
        The number of members in this subgroup (when evaluated against the dataset originally used for subgroup discovery). 
    quality: float
        The quality of this subgroup (when applying the quality function to the dataset originally used for subgroup discovery). 
    target: string
        The name of the target column.
    target_value: object
        The value of the target variable that counts as the 'positive' class. 
    '''
    
    def __init__(self, selectors, target_evaluation, size, quality, target, target_value):
        
        self.selectors = selectors
        self.target_evaluation = target_evaluation
        self.size = size
        self.quality = quality
        self.target = target
        self.target_value = target_value
        
    def __str__(self):
        
        return " AND ".join([str(sel) for sel in self.selectors]).strip()
    
    def get_indices(self, data):
        '''
        Get the indices of rows that meet the subgroup definition for a specified dataset. 
        
        Parameters
        ----------------
        data: pandas DataFrame
            The dataset in which to look for (the indices of) rows that match the subgroup definition. 
            
        Returns
        -----------
        index: pandas Index
            The index identifying rows that meet the subgroup definition in the dataset provided. 
        '''
        
        logical_indices = np.ones(data.index.shape, dtype=bool)
        
        for sel in self.selectors:
            
            if isinstance(sel, PyNumericSelector):
                
                if sel.include_lower_bound and sel.lower_bound != float("-inf"):
                    logical_indices = logical_indices & (data[sel.attribute].values >= sel.lower_bound)
                elif sel.lower_bound != float("-inf"):
                    logical_indices = logical_indices & (data[sel.attribute].values > sel.lower_bound)
                if sel.include_upper_bound and sel.upper_bound != float("inf"):
                    logical_indices = logical_indices & (data[sel.attribute].values <= sel.upper_bound)
                elif sel.upper_bound != float("inf"):
                    logical_indices = logical_indices & (data[sel.attribute].values < sel.upper_bound)
            
            if isinstance(sel, PyNominalSelector):
                
                logical_indices = logical_indices & (data[sel.attribute].astype(str).values == sel.value)
                    
        return data.index[logical_indices]
    
    def get_rows(self,data):
        '''
        Get the rows that meet the subgroup definition for a specified dataset. 
        
        Parameters
        ----------------
        data: pandas DataFrame
            The dataset in which to look for rows that match the subgroup definition. 
            
        Returns
        -----------
        rows: pandas DataFrame
            A selection of rows in the provided dataset that meet the subgroup definition. 
        '''
        
        return data.loc[self.get_indices(data)]



class PySubgroupResults:
    '''
    A collection of subgroups, returned as a result of performing subgroup discovery.
    
    Note that this is detached from the Java runtime, and so is a plain python object. 
    
    Attributes
    --------------
    subgroups: list
        A list of `PySubgroup` objects. 
    population_evaluation: float
        The value of the target variable across the entire dataset originally used for subgroup discovery. 
    population_size: int
        The number of rows in the dataset originally used for subgroup discovery. 
    target: string
        The name of the target column.
    target_value: object
        The value of the target variable that counts as the 'positive' class. 
    '''
    
    def __init__(self, subgroups, population_evaluation, population_size, target, target_value):
        
        self.subgroups = subgroups
        self.population_evaluation = population_evaluation
        self.population_size = population_size
        self.target = target
        self.target_value = target_value
    
    def __len__(self):
        
        return len(self.subgroups)
    
    def __iter__(self):
    
        return self.subgroups.__iter__()
    
    def __getitem__(self, selection):
    
        if hasattr(selection, '__iter__'):
        
            subgroups = [self.subgroups[i] for i in selection]
            
            out = copy.copy(self)
            out.subgroups = subgroups
            
            return out
    
        if isinstance(selection, slice):

            out = copy.copy(self)
            out.subgroups = self.subgroups.__getitem__(selection)
            
            return out 
        
        return self.subgroups[selection]
        
    
    def to_df(self):
        '''
        Convert the subgroups included in this object into an easy-to-read pandas dataframe for viewing. 
        
        Returns
        -----------
        subgroups_df: pandas DataFrame
            A table showing the subgroup definitions and associated important values like size, target value, and quality. 
        '''
        
        return pd.DataFrame([{"pattern":str(sg),"target_evaluation":sg.target_evaluation,"size":sg.size,"quality":sg.quality} for sg in self.subgroups])


def discover_subgroups(
    ontology,
    target,
    target_value=None,
    included_attributes=None,
#    discretise=True,
    nbins=3,
    method="sdmap",
    qf="ps",
    k=20,
    minqual=0,
    minsize=0,
    mintp=0,
    max_selectors=3,
    ignore_defaults=False,
    filter_irrelevant=False,
    postfilter="",
    postfilter_param=0.00 ## Must be provided for most postfiltering types
):
    '''
    Search for interesting subgroups within a dataset. 
    
    Parameters
    ----------------
    ontology: pandas DataFrame or PyOntology object. 
        The data to use to peform subgroup discovery. Can be a pandas DataFrame, or a PyOntology object. 
    target: string
        The name of the column to be used as the target.
     target_value: object, optional
        The value of the target variable that counts as the 'positive' class. Not needed for a numeric target, in which case the mean of the target variable will be used for subgroup discovery.
    included_attributes: list, optional
        A list of strings containing the names of columns to use. If not specified, all columns of the data will be used. 
    nbins: int, optional
        The number of bins to use when discretising numeric variables. Default value is 3. 
    method: string, optional
        Used to decide which algorithm to use. Must be one of Beam-Search `beam`, BSD `bsd`, SD-Map `sdmap`, SD-Map enabling internal disjunctions `sdmap-dis`. The default is `sdmap`.
    qf: string, optional
        Used to decide which algorithm to use. Must be one of Adjusted Residuals `ares`, Binomial Test `bin`, Chi-Square Test `chi2`, Gain `gain`, Lift `lift`, Piatetsky-Shapiro `ps`, Relative Gain `relgain`, Weighted Relative Accuracy `wracc`. The default is qf = `ps`.
    k: int, optional
        Maximum number (top-k) of patterns to discover, i.e., the best k patterns according to the selected quality function. The default is 20. 
    minqual: float, optional
        The minimal quality. Defaults to 0, meaning there is no minimum.
    minsize: int, optional
        The minimum size of a subgroup in order for it to be included in the results. Defaults to 0, meaning there is no minimum. 
    mintp: int, optional
        The minimum number of true positives in a subgroup (relevant for binary target concepts only). Defaults to 0, meaning there is no minimum
    max_selectors: int, optional
        The maximum number of selectors/rules included in a subgroup. The default is 3.
    ignore_defaults: bool, optional
        If set to True , the values in the first row of data will be considered ‘default values’, and the same values will be ignored when searching for subgroups. Defaults to False. 
    filter_irrelevant: bool, optional
        Whether irrelevant patterns are filtered out. Note that this negatively impacts performance. Defaults to False. 
    postfilter: string, optional
        Which post-processing filter is applied. 
        Can be one of: 
         * Minimum Improvement (Global) `min_improve_global`, which checks the patterns against all possible generalisations; 
         * Minimum Improvement (Pattern Set) `min_improve_set`, checks the patterns against all their generalisations in the result set, 
         * Relevancy Filter `relevancy`, removes patterns that are strictly irrelevant, 
         * Significant Improvement (Global) `sig_improve_global`, removes patterns that do not significantly improve (default 0.01 level, can be overridden with postfilter_param) with respect to all their possible generalizations, 
         * Significant Improvement (Set) `sig_improve_set`, removes patterns that do not significantly improve (default 0.01 level, can be overridden with postfilter_param) with respect to all generalizations in the result set,
         * Weighted Covering `weighted_covering`, performs weighted covering on the data in order to select a covering set of subgroups while reducing the overlap on the data. 
        By default, no postfilter is set, i.e., postfilter = "".
    postfilter_param: float, optional
        Provides the corresponding parameter value for the filtering chosen in postfilter. Must be provided for most postfiltering types
    
    Returns
    -----------
    subgroups: PySubgroupResults
        The discovered subgroups. 
    '''
    
    if target_value is not None :
    
        if isinstance(ontology, PyOntology):
        
            raise ValueError("target_value cannot be provided when passing in a PyOntology instead of a pands DataFrame.")
        
        target_bool = ontology[target] == target_value
        
        ontology = ontology.drop(columns=target).join(target_bool)
    
    elif ontology[target].dtype == 'object' or ontology[target].dtype == 'bool' or ontology[target].dtype.name == 'category':
        
        target_value = ontology[target].iloc[0]
    
    if not isinstance(ontology, PyOntology):
    
        ontology = PyOntology(ontology.reset_index(drop=True)) ## Reset index because pandas seems to confuse itself when there is a MultiIndex!!!
    
    ont = ontology.ontology
    
    includedAttributes = java.util.HashSet()
    if included_attributes:
        includedAttributes.addAll(included_attributes)
    else:
        includedAttributes.addAll(ontology.column_names)
        
    if target not in includedAttributes:
        includedAttributes.add(target)
    
    subgroups = PythonDiscoverSubgroups.discoverSubgroups(
        ont,
        JString(target),
        includedAttributes,
        JInt(nbins),
        JString(method),
        JString(qf),
        JInt(k),
        JDouble(minqual),
        JInt(minsize),
        JInt(mintp),
        JInt(max_selectors),
        JBoolean(ignore_defaults),
        JBoolean(filter_irrelevant),
        JString(postfilter),
        JDouble(postfilter_param)
    )
    
    py_subgroups = []
    
    population_value = None
    population_size = None
    
    for sg in subgroups.sortSubgroupsByQualityDescending():
        
        py_selectors = []
        
        for selector in sg.getDescription():
            
            if isinstance(selector, DefaultSGSelector):
                
                py_selectors.append(
                    PyNominalSelector(
                        str(selector.getAttribute().getId()),
                        str(list(selector.getValues())[0])
                    )
                )
            
            if isinstance(selector, NumericSelector):
                
                lb = selector.getLowerBound()
                ub = selector.getUpperBound()
                
                if str(selector.getAttribute().getId()) in ontology.datetime_columns:
                    
                    try: ## convert back to datetime
                        lb = pd.to_datetime(int(lb)).tz_localize("GMT").tz_convert(
                            ontology.datetime_columns[str(selector.getAttribute().getId())])
                    except OverflowError: ## if 'inf'
                        lb = float("-inf")
                
                    try:
                        ub = pd.to_datetime(int(ub)).tz_localize("GMT").tz_convert(
                            ontology.datetime_columns[str(selector.getAttribute().getId())])
                    except OverflowError:
                        ub = float("inf")
                
                elif str(selector.getAttribute().getId()) in ontology.timedelta_columns:
                    
                    try: ## convert back to timedelta
                        lb = pd.to_timedelta(int(lb))
                    except OverflowError: ## if 'inf'
                        lb = float("-inf")
                    
                    try:
                        ub = pd.to_timedelta(int(ub))
                    except OverflowError:
                        ub = float("inf")
                    
                else:
                    
                    lb = float(lb)
                    ub = float(ub)
                    
                
                py_selectors.append(
                    PyNumericSelector(
                        str(selector.getAttribute().getId()),
                        lb,
                        ub,
                        bool(selector.isIncludeLowerBound()),
                        bool(selector.isIncludeUpperBound())
                    )
                )  
        
        stats = sg.getStatistics()
        population_value = float(stats.getTargetQuantityPopulation())
        population_size = float(stats.getDefinedPopulationCount())
        
        subgroup_value = float(stats.getTargetQuantitySG())
        subgroup_size = float(stats.getSubgroupSize())
        subgroup_quality = float(sg.getQuality())
        
        py_subgroups.append(
            PySubgroup(
                py_selectors,
                subgroup_value,
                subgroup_size,
                subgroup_quality,
                target,
                target_value
            )
        )
    
    return PySubgroupResults(py_subgroups, population_value, population_size, target, target_value)

