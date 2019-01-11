# Crime, Weather & Climate Change

Reproducing "Crime, Weather & Climate Change" by Matthew Ranson (2014) — Group 8

## Purpose 

- The purpose of the code and analyses in this repository is to reproduce the calculations done by Matthew Ranson in his paper "Climate, Weather, and Climate Change" (2014), using data from Alameda County. 
- Ranson "estimates the impact of climate change on the prevalence of criminal activity in the United States" (Ranson 2014). He does this by analyzing 30 years worth of monthly data regarding crime and weather in 2997 counties in the US and eventually identifies the relationship between the two. Ranson came to the conclusion that “temperature has a strong positive effect on criminal behavior” (Ranson 2014).

## Repository Organization 

- **Analysis** - The analysis for all parts of this group assignment is contained in a single Jupyter notebook called Analysis.ipynb, accompanied with its exported markdown format. Each part of the assignment is ordered in sequential order, accompanied with prompt, code, explanation and visualization.
- **Code** - All python functions that are used in the analysis file are included in separate folders with name "assignmentX". Each folder is am independent python module, providing all functionality that are required for corresponding assignments. Each module also contains a unit_test, and an environment file, which is the conda environment configuration required for this assignment.
- **Data** - All data files downloaded from OSF will be stored in the folder called data. However, due to the size of the data, they are not included within this Github repo. Instead, all files will be downloaded by python functions when the first time they are used. Please refer to the readme file within data folder to see the complete list of files and their download
- **Literature** - All related literature are contained in lit folder.
- **Images** - All related images are contained in images folder.

## Analysis

The following section contains links to the analysis for each group assignment.

* [Group Assignment 1](https://github.berkeley.edu/stat-159-259-f18/cwcc-g8/blob/master/Analysis.md#group-assignment-1)
* [Group Assignment 2](https://github.berkeley.edu/stat-159-259-f18/cwcc-g8/blob/master/Analysis.md#group-assignment-2)
* [Group Assignment 3](https://github.berkeley.edu/stat-159-259-f18/cwcc-g8/blob/master/Analysis.md#group-assignment-3)
* [Group Assignment 4](https://github.berkeley.edu/stat-159-259-f18/cwcc-g8/blob/master/Analysis.md#group-assignment-4)
* [Group Assignment 5](https://github.berkeley.edu/stat-159-259-f18/cwcc-g8/blob/master/Analysis.md#group-assignment-5)
* [Group Assignment 6](https://github.berkeley.edu/stat-159-259-f18/cwcc-g8/blob/master/Analysis.md#group-assignment-6)
* [Final Presentation Video](http://inst.eecs.berkeley.edu/~cs199-dnd/FinalPresentation.mp4)

## Environment Setup

Environment configuration fies are located within each assignment folder named assignmentX. Please use the corresponding environment file to run functions or unit tests for any specific assignment. For the Analysis Jupyter Notebook, please use the environment file from the latest assignment folder.


## Makefile

* `make output` - Generate md and pdf for ipython notebook
* `make test` - Run all unit tests
* `make coverage` - Compute unit tests coverage rate (using [Coverage.py](https://coverage.readthedocs.io))
* `make clean` - Clean up everything

## Works Cited

- Ranson, M. (2014). Crime, weather, and climate change. Journal of Environmental Economics and Management, 67(3), 274–302. https://doi.org/10.1016/j.jeem.2013.11.008
- Stark, Philip B. pbstark/S159-f18. (n.d.). Retrieved October 31, 2018, from https://github.berkeley.edu/pbstark/S159-f18
- Kaplan, Jacob. Uniform Crime Reporting Program Data: Offenses Known and Clearances by Arrest, 1960-2016. Ann Arbor, MI: Inter-university Consortium for Political and Social Research [distributor], 2018-08-21. https://doi.org/10.3886/E100707V7
- Negus, Mitch. “Crime, Weather, and Climate Change in Alameda County.” OSF, 26 Oct. 2018. Web. https://osf.io/q8hgm/

## Authors
- Junfang Jiang (junfang@berkeley.edu)
- Kaiqian Zhu (tim3212008@berkeley.edu)
- Samuel Ouyang (souyang@berkeley.edu)
- Rowan Cassius (rocassius@berkeley.edu)
- Rachel Henry (rachelhenry@berkeley.edu)
