# kmml-data-challenge

Repository for the [Data Challenge](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021) of the [Machine learning with Kernel methods](http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2021mva/index.html) class (Spring 2021) taught by [Julien Mairal](https://lear.inrialpes.fr/people/mairal/) and [Jean-Philippe Vert](http://members.cbio.mines-paristech.fr/~jvert/) as part of the [Master MVA](https://www.master-mva.com) at [École Normale Supérieure Paris-Saclay](https://ens-paris-saclay.fr/en).

## Usage
Run `start.py` with python3 and the dependencies to reproduce the Kaggle submission.

## Code organization

### Main files
* `models.py` contains the different kernel methods
* `kernels.py` contains the different kernels on R^d
* `kmer_procesor.py` contains the functions and class used to process and embed the data in R^d
* `utils.py` contains utils functions used in the repository

### Auxiliary files
* `benchmark.py` contains code used to benchmark differnent methods with use of optionnal cross-validation
* `majority_voting.py` contains code used to combine submission with majority voting (although we did not use it to produce our final submission) as we wanted it to be easily reproducible
* `submission_analysis.py` contains code used to analyze and compare different submissions

### Folders
* `experiments/` contains experimental code that we used to conduct our research
* `data/` contains the data for the data challenge
* `test/` contains unit tests for some functions and methods in the Main files
* `submissions/` contains the submissions we uploaded to Kaggle
