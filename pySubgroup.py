import jpype
import jpype.imports
from jpype.types import *
jpype.startJVM(classpath=['vikamine_kernel.jar'])
import java.util.HashSet

from org.vikamine.kernel._examples import *
from org.vikamine.kernel.subgroup.selectors import *

import pandas as pd
import numpy as np

class PyOntology:
    
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
            
            if x.dtype == 'object' or x.dtype == 'bool' or x.dtype.name == 'category': # category depends on whether its ordered?

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
    
    def __init__(self, attribute, value):
        
        self.attribute = attribute
        self.value = value
        
    def __str__(self):
        
        return "{0} = {1}".format(self.attribute, self.value)


class PySubgroup:
    
    def __init__(self, selectors, target_value, size, quality):
        
        self.selectors = selectors
        self.target_value = target_value
        self.size = size
        self.quality = quality
        
    def __str__(self):
        
        return " AND ".join([str(sel) for sel in self.selectors]).strip()
    
    def get_indices(self, data):
        
        logical_indices = np.ones(data.index.shape, dtype=bool)
        
        for sel in self.selectors:
            
            if isinstance(sel, PyNumericSelector):
                
                if sel.include_lower_bound and sel.lower_bound != float("-inf"):
                    logical_indices = logical_indices & (data[sel.attribute] >= sel.lower_bound)
                elif sel.lower_bound != float("-inf"):
                    logical_indices = logical_indices & (data[sel.attribute] > sel.lower_bound)
                if sel.include_upper_bound and sel.upper_bound != float("inf"):
                    logical_indices = logical_indices & (data[sel.attribute] <= sel.upper_bound)
                elif sel.upper_bound != float("inf"):
                    logical_indices = logical_indices & (data[sel.attribute] < sel.upper_bound)
            
            if isinstance(sel, PyNominalSelector):
                
                logical_indices = logical_indices & (data[sel.attribute].astype(str) == sel.value)
                    
        return data.index[logical_indices]
    
    def get_rows(self,data):
        
        return data.loc[self.get_indices(data)]



class PySubgroupResults:
    
    def __init__(self, subgroups, population_value, population_size):
        
        self.subgroups = subgroups
        self.population_value = population_value
        self.population_size = population_size
    
    def to_df(self):
        
        return pd.DataFrame([{"pattern":str(sg),"target_quantity":sg.target_value,"size":sg.size,"quality":sg.quality} for sg in self.subgroups])


def discover_subgroups(
    ontology,
    target,
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
    
    if not isinstance(ontology, PyOntology):
    
        ontology = PyOntology(ontology)
    
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
                subgroup_quality
            )
        )
    
    return PySubgroupResults(py_subgroups, population_value, population_size)

