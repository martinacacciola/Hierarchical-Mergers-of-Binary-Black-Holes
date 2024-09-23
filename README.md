# Hierarchical Mergers of Binary Black Holes
This repository contains the material related to the project "Hierarchical Mergers of Binary Black Holes" carried out during the Laboratory of Computational Physics (Mod. A) course at the University of Padua.
The supervisor is Dr. Giuliano Iorio from the DEMOBLACK group, whose PI is Prof. Michela Mapelli.

## The project

A binary black hole (BBH) can form via close encounters with black holes (BHs) in a dense stellar environment, such as a nuclear star cluster (NSC), a globular cluster (GC) or a young star cluster (YSC). NSCs are very massive (~ $10^5 - 10^8 \, M_\odot$) star clusters lying at the center of some galaxies, including the Milky Way. GCs are old (~ 12 Gyr) massive (~ $10^4 - 10^6 M_\odot$) stellar clusters lying in the halo of galaxies. YSC are young (< 100 Myr) stellar clusters forming mostly in the disk of a galaxy.  

Several channels can lead to the formation of BBHs. But the distinctive signature of the dynamical scenario is the formation of hierarchical mergers (IMs), i.e. repeated mergers of stellar-origin BHs that build up more massive ones. This process is possible only in dense star clusters, where the merger remnant, which is initially a single BH, can pair up by dynamical exchanges or three-body encounters. The main obstacle to the formation of second-generation BHs via hierarchical mergers is the relativistic kick that the merger remnant receives at birth. This kick can be up to several thousand km/s. Hence, the interplay between the properties of the host star cluster (e.g., its escape velocity), those of the first-generation BBH population and the magnitude of the kick decides the maximum mass of a merger remnant in a given environment.  

A property that is being studied is that IM can build up IMBHs and also partially fill the pair-instability (PI) mass gap between ~60 and ~120 $M_\odot$, explaining the formation of BBHs like GW190521.

#### Hierarchical mergers
When two stellar-born BHs merge via GW emission, their merger remnant is called second-generation (2g) BH. The 2g BH is a single object at birth. However, if it is retained inside its host star cluster, it may pair up dynamically with another BH. This gives birth to what we call a second-generation (2g) BBH, i.e. a binary black hole that hosts a 2g black hole. If a 2g binary black hole merges again, it gives birth to a third-generation (3g) BH, and so on. In this way, repeated black hole mergers in star clusters can give birth to hierarchical chains of mergers, leading to the formation of more and more massive black holes.

## Goal of the project
Understand the differences between hierarchical binary black hole mergers in NSCs, GCc and YSc, by looking at a set of simulated BBHs. 
Our analysis will be carried out with classification ML algorithms, such as Random Forest and XGBoost. We will then proceed to analyze the importance of features to understand the properties of systems of BBHs.   

The idea is to split the analysis into two parts:

- Based on the features of the BBHs systems, figure out to which host star cluster these systems belong. 
  It's a classification problem with labeling the `label` column of the dataset corresponding to `0 -> GC, 1 -> NSC, 2 -> YSC`. Feature importance analysis will tell us which features are most important to understand which system belongs to which host stellar cluster.

- Analyze each stellar cluster independently. To do this we added a new label column `label_ngen`: `0` if the system has no other mergers beyond the 2nd generation; `1` if the system evolves beyond the 2nd generation.
  This is still a classification problem, this time with respect to `label_ngen`. Analysis of features importance will tell us which features are most important that lead systems to evolve and which do not.

To do this in a more detailed way, both globally (all labels together) and locally (single label), we will use `SHAP` values (Section 6).  

To have a cleaner notebook we created a file (`hmbh.py`) containing the functions needed to create the dataset, to train the ML models and to plot the results.

## Conclusions and future work
We investigated the differences between hierarchical binary black hole mergers in NSCs, GCc and YSc and HMs efficiency. 
We considered two different machine learning algorithms: a `RandomForestClassifier` and `XGBoostClassifier`. Both models are based on tree ensemble, but the optimization is different.  
Before training the models we studied which dataset setup would lead to better results, using a simple-as-possible RandomForest model. The benchmarking led us to the conclusion that a balanced (label-wise) dataset is preferable and that we could have reached high performance using a reduced dataset, gaining computational efficiency (i.e. faster training).  

To retrieve the best model, we performed a grid search on a grid of different hyperparameters. The results of the two models for the two tasks were very close, thus we decided to use only the `RandomForestClassifier` model in our final analysis. Both classifiers performed greatly on the task concerning the classification of the stellar clusters, reaching both high accuracy and high classification scores. 
In the task concerning the classification with labeling `label_ngen`, two different models struggle to classify approximately 25%/30% of label `1`, despite the dataset being balanced and the models not exhibiting strong overfitting. There can be multiple causes, for example: data might contain patterns that are inherently difficult to capture or the models used are too complex or too simple for the given task. In future work, other machine learning algorithms may be tried, such as a CNN, which is more flexible in understanding patterns and substructures in a dataset.

We struggled to give an astrophysical interpretation of the features related to the first task. The YSCs are well described while both GCs and NSCs not. Maybe with a different machine learning algorithm and purpose-built dataset, we can achieve better and more explicative results.
The features found in analyzing the SHAP values related to the `label_ngen` task, i.e. assessing the hierarchical mergers efficiency, almost correspond with the ones derived by "classical" statistics methods. One possible evolution to study the efficiency of hierarchical mergers in more detail is to study systems of BHs in their stellar cluster hosts individually, to gain more insights in features importance in different type of stellar clusters. 
