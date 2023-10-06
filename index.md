---
layout: default
---

# Introduction/Background
National and regional differences have had significant policy implications for global sustainability; such geopolitical and cultural differences have led to divergent outcomes on a range of social, economic, and environmental measures. Understanding the current trajectory, and disparate outcomes, by utilizing statistical learning is critical in identifying key trends that can inform future global sustainability efforts.
<br />
<br />
Prior literature has demonstrated such analysis on a regional scale. Mathrani et al. analyze national data for Asian nations from the 2022 United Nations Sustainable Development Goal (SDG) Report. They find, that across four major dimensions of sustainability, sub-regions in Asia perform differently. Most notably, West and Central Asian nations performed well in economic sustainability, while East Asian nations performed well in social and institutional sustainability.
<br />
<br />
A previous study used 2019 UN SDG data and clustered all nations using K-Means. The study identified five clusters which were then analyzed. The resulting clusters were found to have similar characteristics in SGD progress, as well as social, economic, and political characteristics. 
These studies focus on the UN SDGs, which aim to harmonize economic, social, and environmental measures (Di Vaio et al., 2020a). While this approach utilizes data with numerous features for analysis, it is limited in its time span. Our approach aims to widen the scope of analysis temporally.

# Problem definition
Given the World Sustainability Dataset we want to cluster countries by relevant features that correspond to the economic, environmental, and social prosperity of the country. (Note: these categories of prosperity are not comprehensive and are subject to change) Determining which columns from this dataset correspond to these categories will be a preprocessing step we must do before. Once these clusters are created we want to map the country codes from each cluster to their geographic location and notice correlation between location and any of categories of prosperity. Once a relationship is determined we can use these findings to understand which parts of the world perform best and worst in certain areas. In doing so we can take proper steps to accurately address the shortcomings of countries that are not prospering. Additionally, because some of the data is chronological we can potentially predict a country becoming an outlier in their cluster, and if this is in a negative direction, our findings can help inform which sectors of prosperity must be worked on. 


# Methods
After identifying proper features/dimensions needed, we will perform GMM clustering to see which countries are closely related in their sustainability approach. Additionally, we will use linear regression to see if there is any correlation between countries and any sustainability characteristic. Furthermore, random forests and support vector machines will be used to make predictions. The goal is to identify countries that have similar sustainability practices and why. 

# Potential Results and Discussion
Results of clustering will be evaluated using internal measures because the dataset does not contain labels for it. The goodness of the clustering will be evaluated using methods such as Silhouette Coefficient, normalized cut, Beta-CV, and Davies-Bouldin Index. Information Theory techniques can be applied to determine the effect of each feature on the final classification. With the predictions obtained from a random forest or support vector machine, the linear regression generated will be evaluated using any of three APIs offered by Scikit. The accuracy of the predictions will give insight into how correlated certain features are and which are likely to be predictors of future prosperity in a given category. This can inform different choices in features to use for clustering to create more accurate models.

# References 
Çağlar, M., Gürler, C. Sustainable Development Goals: A cluster analysis of worldwide countries. Environ Dev Sustain 24, 8593–8624 (2022). https://doi.org/10.1007/s10668-021-01801-6
<br />
<br />
Di Vaio, A., Palladino, R., Hassan, R., & Alvino, F. (2020a). Human resources disclosure in the EU Directive 2014/95/EU perspective: A systematic literature review. Journal of Cleaner Production, 257, 120509. https://doi.org/10.1016/j.jclepro.2020.120509
<br />
<br />
Mathrani, Anuradha, Jian Wang, Ding Li, and Xuanzhen Zhang. 2023. "Clustering Analysis on Sustainable Development Goal Indicators for Forty-Five Asian Countries" Sci 5, no. 2: 14. https://doi.org/10.3390/sci5020014

# Contribution Table

| Name        | Contribution|
| ----------- | ----------- |
| Ansh Vijay  | Setting up Github Pages <br /> Contribution Table <br /> UI of website <br /> Problem definition|
| John Zhang  | Potential Results and Discussion <br /> Recording video |
| Nicholas Polimeni | Found dataset <br /> Introduction/Background <br /> Gannt Chart|
| Lalith Siripurapu | Recording Video <br /> Methods |

# Gantt Chart
[Link to Gantt Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/npolimeni3_gatech_edu/EcyA5LHVj-VInbvsiN7a9zIBYMocgsMFTuQmjAW3kMClgQ?e=rHj83t)

# Data Set
[World Sustainability Dataset](https://www.kaggle.com/datasets/truecue/worldsustainabilitydataset?select=WorldSustainabilityDataset.csv)

# Video Link 
[Link to video]()

