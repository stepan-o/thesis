[GitHub Flavored Markdown Spec](https://github.github.com/gfm/)
# Description of methodology
# Exploratory Data Analysis (EDA)
From [Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm)
## Data analysis approaches
Three popular data analysis approaches are:
* Classical
* Exploratory (EDA)
* Bayesian
### Paradigms for Analysis Techniques	
These three approaches are similar in that they all start with a general science/engineering problem and all yield science/engineering conclusions. The difference is the sequence and focus of the intermediate steps.

* For classical analysis, the sequence is  
Problem => Data => Model => Analysis => Conclusions

* For EDA, the sequence is  
Problem => Data => Analysis => Model => Conclusions

* For Bayesian, the sequence is  
Problem => Data => Model => Prior Distribution => Analysis => Conclusions

### Method of dealing with underlying model for the data distinguishes the 3 approaches	
* Thus for classical analysis, the data collection is followed by the 
    * imposition of a model (normality, linearity, etc.) 
    * and the analysis, 
    * estimation, 
    * and testing that follows are 
    * **focused on the parameters** of that model. 
* For EDA, the 
    * data collection is **not followed by a model imposition**; 
    * rather it is followed immediately by analysis 
    * with a goal of inferring what model would be appropriate. 
* Finally, for a Bayesian analysis, 
    * the analyst **attempts to incorporate scientific/engineering knowledge/expertise** into the analysis 
    * by imposing a data-independent distribution on the parameters of the selected model; 
    * the analysis thus consists of 
        * formally combining both the prior distribution on the parameters 
        * and the collected data to jointly make inferences and/or test assumptions about the model parameters.

In the real world, data analysts freely mix elements of all of the above three approaches (and other approaches). The above distinctions were made to emphasize the major differences among the three approaches.
## What is EDA?
### Approach	

Exploratory Data Analysis (EDA) is an **approach/philosophy for data analysis** that employs a variety of techniques (mostly graphical) to
1. maximize insight into a data set;
2. uncover underlying structure;
3. extract important variables;
4. detect outliers and anomalies;
5. test underlying assumptions;
6. develop parsimonious models; and
7. determine optimal factor settings.
### Focus	
The EDA approach is precisely that -- _an approach_ -- **not a set of techniques, but an attitude/philosophy** about how a data analysis should be carried out.
### Philosophy	
EDA is not identical to statistical graphics although the two terms are used almost interchangeably. Statistical graphics is a collection of techniques -- all graphically based and all focusing on one data characterization aspect. EDA encompasses a larger venue; EDA is an **approach to data analysis that postpones the usual assumptions about what kind of model the data follow** with the more direct approach of **allowing the data itself to reveal its underlying structure** and model. EDA is not a mere collection of techniques; EDA is a philosophy as to how we dissect a data set; what we look for; how we look; and how we interpret. It is true that EDA heavily uses the collection of techniques that we call "statistical graphics", but it is not identical to statistical graphics per se.
### History	
The seminal work in EDA is [Exploratory Data Analysis, Tukey, (1977)](https://apps.dtic.mil/dtic/tr/fulltext/u2/a266775.pdf). Over the years it has benefited from other noteworthy publications such as [Data Analysis and Regression, Mosteller and Tukey (1977)](https://search.library.utoronto.ca/details?3492606&uuid=d6f7b66b-7037-4c3f-b042-2ff3a76bb5a5), Interactive Data Analysis, Hoaglin (1977), The ABC's of EDA, Velleman and Hoaglin (1981) and has gained a large following as "the" way to analyze a data set.
### Techniques	
Most EDA techniques are graphical in nature with a few quantitative techniques. The reason for the heavy reliance on graphics is that by its very nature the main role of EDA is to open-mindedly explore, and graphics gives the analysts unparalleled power to do so, enticing the data to reveal its structural secrets, and being always ready to gain some new, often unsuspected, insight into the data. In combination with the natural pattern-recognition capabilities that we all possess, graphics provides, of course, unparalleled power to carry this out.
The particular graphical techniques employed in EDA are often quite simple, consisting of various techniques of:

1. Plotting the raw data (such as data traces, histograms, bihistograms, probability plots, lag plots, block plots, and Youden plots.
2. Plotting simple statistics such as mean plots, standard deviation plots, box plots, and main effects plots of the raw data.
3. Positioning such plots so as to maximize our natural pattern-recognition abilities, such as using multiple plots per page.
### Further discussion of the distinction between the classical and EDA approaches
Focusing on EDA versus classical, these two approaches differ as follows:
1. Models
    * _Classical_	  
The classical approach imposes models (both deterministic and probabilistic) on the data. Deterministic models include, for example, regression models and analysis of variance (ANOVA) models. The most common probabilistic model assumes that the errors about the deterministic model are normally distributed -- this assumption affects the validity of the ANOVA F tests.
    * _Exploratory_	 
The Exploratory Data Analysis approach does not impose deterministic or probabilistic models on the data. On the contrary, the EDA approach allows the data to suggest admissible models that best fit the data.
2. Focus
    * _Classical_	 
    The two approaches differ substantially in focus. For classical analysis, the focus is on the model -- estimating parameters of the model and generating predicted values from the model.
    * _Exploratory_	 
    For exploratory data analysis, the focus is on the data -- its structure, outliers, and models suggested by the data.
3. Techniques
    * _Classical_	 
    Classical techniques are generally quantitative in nature. They include ANOVA, t tests, chi-squared tests, and F tests.
    * _Exploratory_	  
    EDA techniques are generally graphical. They include scatter plots, character plots, box plots, histograms, bihistograms, probability plots, residual plots, and mean plots.
4. Rigor
    * _Classical_	 
    Classical techniques serve as the probabilistic foundation of science and engineering; the most important characteristic of classical techniques is that they are rigorous, formal, and "objective".
    * _Exploratory_	 
    EDA techniques do not share in that rigor or formality. EDA techniques make up for that lack of rigor by being very suggestive, indicative, and insightful about what the appropriate model should be. EDA techniques are subjective and depend on interpretation which may differ from analyst to analyst, although experienced analysts commonly arrive at identical conclusions.
5. Data Treatment
    * _Classical_	 
    Classical estimation techniques have the characteristic of taking all of the data and mapping the data into a few numbers ("estimates"). This is both a virtue and a vice. The virtue is that these few numbers focus on important characteristics (location, variation, etc.) of the population. The vice is that concentrating on these few characteristics can filter out other characteristics (skewness, tail length, autocorrelation, etc.) of the same population. In this sense there is a loss of information due to this "filtering" process.
    * _Exploratory_	 
    The EDA approach, on the other hand, often makes use of (and shows) all of the available data. In this sense there is no corresponding loss of information.
6. Assumptions
    * _Classical_ 	
    The "good news" of the classical approach is that tests based on classical techniques are usually very sensitive -- that is, if a true shift in location, say, has occurred, such tests frequently have the power to detect such a shift and to conclude that such a shift is "statistically significant". The "bad news" is that classical tests **depend on underlying assumptions (e.g., normality)**, and hence the validity of the test conclusions becomes dependent on the validity of the underlying assumptions. Worse yet, the exact underlying assumptions may be unknown to the analyst, or if known, untested. Thus the **validity of the scientific conclusions becomes intrinsically linked to the validity of the underlying assumptions**. In practice, if such assumptions are unknown or untested, the validity of the scientific conclusions becomes suspect.
    * _Exploratory_	 
    Many EDA techniques make little or no assumptions -- they present and show the data -- all of the data -- as is, with fewer encumbering assumptions.
### How Does Exploratory Data Analysis Differ from Summary Analysis?
* _Summary_	

A summary analysis is **simply a numeric reduction of a historical data set**. It is quite passive. Its focus is in the past. Quite commonly, its purpose is to simply arrive at a few key statistics (for example, mean and standard deviation) which may then either replace the data set or be added to the data set in the form of a summary table.

* _Exploratory_
	
In contrast, EDA has as its broadest goal the desire to gain insight into the engineering/scientific process behind the data. Whereas **summary statistics are passive and historical, EDA is active and futuristic**. In an attempt to "understand" the process and improve it in the future, EDA uses the data as a "window" to peer into the heart of the process that generated the data. There is an archival role in the research and manufacturing world for summary statistics, but there is an enormously larger role for the EDA approach.
## What are the EDA Goals?
### Primary and Secondary Goals	
The primary goal of EDA is to maximize the analyst's insight into a data set and into the underlying structure of a data set, while providing all of the specific items that an analyst would want to extract from a data set, such as:
1. a good-fitting, parsimonious model
2. a list of outliers
3. a sense of robustness of conclusions
4. estimates for parameters
5. uncertainties for those estimates
6. a ranked list of important factors
7. conclusions as to whether individual factors are statistically significant
8. optimal settings
### Insight into the Data	
Insight implies detecting and uncovering underlying structure in the data. Such underlying structure may not be encapsulated in the list of items above; such items serve as the specific targets of an analysis, but the real insight and "feel" for a data set comes as the analyst judiciously probes and explores the various subtleties of the data. 

The "feel" for the data comes almost exclusively from the application of various graphical techniques, the collection of which serves as the window into the essence of the data. Graphics are irreplaceable--there are no quantitative analogues that will give the same insight as well-chosen graphics.

To get a "feel" for the data, it is **not enough for the analyst to know what is in the data**; the analyst also **must know what is not in the data**, and the only way to do that is to draw on our own human pattern-recognition and comparative abilities in the context of a series of judicious graphical techniques applied to the data.
### The Role of Graphics
#### Quantitative/Graphical	
Statistics and data analysis procedures can broadly be split into two parts:
*  _Quantitative_	   
Quantitative techniques are the set of statistical procedures that yield numeric or tabular output. Examples of quantitative techniques include:  
    * hypothesis testing;
    * analysis of variance;
    * point estimates and confidence intervals;
    * least squares regression.  
    * These and similar techniques are all valuable and are mainstream in terms of classical analysis.

* _Graphical_	 
On the other hand, there is a large collection of statistical tools that we generally refer to as graphical techniques. These include:
    * scatter plots;
    * histograms;
    * probability plots;
    * residual plots;
    * box plots;
    * block plots.  
#### EDA Approach Relies Heavily on Graphical Techniques	
The EDA approach relies heavily on these and similar graphical techniques. Graphical procedures are not just tools that we could use in an EDA context, they are tools that we must use. Such graphical tools are the shortest path to gaining insight into a data set in terms of
* testing assumptions
* model selection
* model validation
* estimator selection
* relationship identification
* factor effect determination
* outlier detection

**If one is not using statistical graphics, then one is forfeiting insight into one or more aspects of the underlying structure of the data.**

[Anscombe (1973)](http://www.sjsu.edu/faculty/gerstman/StatPrimer/anscombe1973.pdf) points out to the importance of graphs in statistical analysis, as graphs can serve various purposes, such as:
* to help us perceive and appreciate some broad features of the data;
* to let us look behind those broad features and see what else is there;
* to help us test the validity of the assumptions we take about the behaviour of the data when performing statistical calculations.