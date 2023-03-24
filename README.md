# SD4Py

* License: LGPL (GNU Lesser General Public License v3.0)

SD4Py is a package that makes it easy to perform subgroup discovery on tabular data. It is extremely simple to use. Call the `sd4py.discover_subgroups()` function on a Pandas dataframe and a collection of subgroups will be returned. 

This package provides a Python interface for using the Java application VIKAMINE. 

## Quick overview of subgroup discovery

Subgroup discovery is based on finding patterns within some (explanatory) columns of data that then help to explain another (target) column of data. 
The goal of the subgroup discovery process will be to understand in what circumstances the target is extreme. With a numeric target, this means finding circumstances in which the value is exceptionally high (or exceptionally low) on average.
For a non-numeric target, this means looking for circumstances when a particular value is especially likely to occur.
One of the key benefits of this approach is that the outputs are interpretable, being expressed as a readable combination of rules like (e.g.)  "'Temperature'=high AND 'Pressure'=low". 

Alongside the target variable, it is crucial to have other variables that potentially will help to explain the target variable. These should be measurements that naturally accompany the target, i.e., variables that seem irrelevant to the target should be excluded. 
Once this has been decided and the data has been prepared, the user is ready to use SD4Py. 

One thing to note when performing the analysis is that subgroup discovery supports an iterative approach. This means that the subgroup discovery process can be run to obtain initial subgroups, 
which then might suggest changes like adding or removing variables or refining the search parameters, before re-running the process. 

The outputs of the process are discovered subgroups that help to explain the target. 

Before moving on, it is worth defining some relevant terminology:
 * A *rule* is a condition in the form “Property_A = True” that combines with other rules to form a pattern
 * A *subgroup* is the subset of data, defined through a combination of *rules*
 * The *target* is the variable that patterns try to explain, by looking for subgroups with an extreme target value on average
 * The *quality* of a pattern is a number that quantifies how well the pattern identifies a subgroup with an extreme target value

## Important note on dependencies 

There are two important dependencies for SD4Py. The first is JPype, which is used to run the JRE behind the scenes (although the user does not need to interact with JPype). 
The second is Pandas, which is used to store and manipulate tabular data. The data you want to use must be in a Pandas dataframe. 

## Basic usage example 

Please see the basic usage example provided as a notebook on the SD4Py GitHub page to see how to get started quickly. 
