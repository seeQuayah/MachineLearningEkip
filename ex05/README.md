## Part 5 : application of unsupervised learning

### Dataset

We choose to use the Open Food Facts database, we will focus on the macronutrient

Here is the [link](https://world.openfoodfacts.org/data)

You can use **setup.sh** to download the dataset in csv (about 7.4Gb)


### Analysis

Since we have a very huge dataset, we take a sample of 10000 products.

We make a first selection of quantitatives variables and start to make some historigram in order to visualise what's is going on with this dataset.

![historigram](https://cdn.discordapp.com/attachments/579658084595662848/1085230268178235523/Capture_decran_2023-03-14_a_16.57.33.png)

This first analysis show us many thing:
- 
We want to keep on macronutriments data and the nutrition-score-fr_100g is already an analized data.
We also updated our code to get more result and swapped energy_100g to energy-kcal_100g to get more readable results.

![historigram](https://cdn.discordapp.com/attachments/579658084595662848/1085235552711888917/Capture_decran_2023-03-14_a_17.18.33.png)

Based on the analysis of the OpenFoodFacts dataset using the code provided in the analysis.py script, here are some possible conclusions that can be drawn:

Histograms of the quantitative variables show that the distribution of values for each variable is generally right-skewed, with a few extreme values on the right-hand side of the distributions. The means and standard deviations of the variables are also provided in the output of the describe() function, and they show that there is a large variation in the values of each variable.

The histograms of sodium_100g  show that the distribution of values is also right-skewed, with a few extreme values on the right-hand side of the distributions.

We also see that salt_100g is a 1 to 1 copy of sodium_100g so we decided to cut it out of our future analysis.

Using the generated Boxplot we can identify some outliers.

![Boxplot](https://cdn.discordapp.com/attachments/579658084595662848/1085523080002142259/image.png)

The boxplot of the quantitative variables shows that there are a few extreme outliers in the dataset, particularly for the energy-kcal_100g and sodium_100g variables. These outliers could be the result of measurement errors, data entry errors, or other factors, and they should be further investigated to determine whether they are valid data points or should be removed from the dataset.

![Heatmap](https://cdn.discordapp.com/attachments/579658084595662848/1085240027463815188/Figure_1.png)

The correlation matrix of the quantitative variables shows that there are moderate to strong positive correlations between several pairs of variables, such as fat_100g and saturated-fat_100g, carbohydrates_100g and sugars_100g. These correlations indicate that some variables may be redundant or highly related to each other, and this should be taken into account when conducting further analyses.
