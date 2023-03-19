
# Part 5 : application of unsupervised learning

## Dataset

We choose to use the Open Food Facts database.

Here is the link to find the dataset [on the openfoodfacts.org website](https://world.openfoodfacts.org/data).

You can use **setup\.sh** to download the dataset in csv (about 7.4Gb).

Our goal here is to perform some analysis and find if there is products that have same nutritional profiles using k-means clustering.

We will first do some basic analysis in ord

## Analysis

#### Introduction

Since we have a very huge dataset, we take a sample *500 000 products*.
You can find the code related to this part in **analysis.py** file

#### Story 1: First Historigrams

We make a first selection of quantitatives variables that are macronutriment and start to make some historigram in order to visualise what's is going on with this dataset.

And there is the result:

![historigram](https://cdn.discordapp.com/attachments/579658084595662848/1087075254813331536/Figure_1.png)

This first analysis show us many thing:
- We see that there is a link between the salt and the sodium as the have the same historigram. We decide to remove the salt for further study of the dataset.
- We also see that the distribution of values for each variable is generally right-skewed
- We potentialy detect here some outliers, we will need to find them and get rid of these.

#### Story 2: Outliers hunter

As far as we know, the best way to figure out outliers is to visualise the sample with a **boxplot**.

Here is the result: 

![Boxplot](https://cdn.discordapp.com/attachments/579658084595662848/1085523080002142259/image.png)

We see that there are a few extreme outliers in the sample, particularly for the energy-kcal_100g and sodium_100g variables.

In order to get rid of these we will use IQR 

After some try here is the result:

![outliers-p1](https://cdn.discordapp.com/attachments/579658084595662848/1086649004726681661/image.png)


![outlier-p2](https://cdn.discordapp.com/attachments/579658084595662848/1086649101510266951/image.png)

We can now make a new boxplot int order to see if the result is better.

![boxplot-2](https://cdn.discordapp.com/attachments/579658084595662848/1086649226135605288/image.png)

As we can see, we have know better data, we can make better historigram and perform further analysis.
  
  
![](https://cdn.discordapp.com/attachments/579658084595662848/1087075765759246478/Figure_1.png)



#### Descriptive Statistics 

Next we decided to analyse some statistics about our sample and thanks to the *panda's describe methods*

```
       energy-kcal_100g     fat_100g  saturated-fat_100g  carbohydrates_100g  sugars_100g  proteins_100g  sodium_100g
count       183922.000000  183922.000000       183922.000000       183922.000000  183922.000000  183922.000000  183922.000000
mean         242.468460      10.391480            3.310310           30.850989      10.201241       7.085206       0.366972
std          157.668172      10.704978            4.011757           25.111221      12.332375       5.755209       0.308583
min          0.000000         0.000000            0.000000            0.000000       0.000000       0.000000       0.000000
25%          104.000000       1.500000            0.000000            8.160000       1.180000       2.820000       0.090920
50%          229.000000       7.040000            1.754386           24.418605       4.166667       6.000000       0.320000
75%          367.000000      16.666667            5.330000           52.380952      16.129032      10.256410       0.541667
max          812.500000      48.400000           16.670000          100.000000      48.333333      23.474178       1.364360
```

With these statistics we can see that:

Similar patterns can be observed for the other features as well, with some right-skewed distributions and a few outliers with very high values.

Based on these insights, making clusters may be a good idea because it can help identify groups of food items that are similar in terms of their nutrient content. For example, one cluster may contain food items that are high in energy and fat content, while another cluster may contain food items that are low in energy and fat but high in protein. Clustering can help in identifying such patterns and can be used to develop personalized nutrition recommendations or to identify healthy food alternatives for individuals with specific dietary needs.


#### Corelation Matrix


![](https://cdn.discordapp.com/attachments/579658084595662848/1085858187212763217/Figure_1.png)

To see if clustering will be a good idea we created a correlation matrix.

We can see that there are multiple correlation between multiple value like th energy with fat and saturated fat.

We can also see another correlation between carbohydrates and sugar.

Those might be 2 possible clusters but more might still be hidden to us.

#### WCSS

At this step, we need to **scale** the data in order to use correctly the k-means algorithm.


![hello](https://cdn.discordapp.com/attachments/579658084595662848/1086649424840753263/image.png)

To determines the optimal number of cluster to use, we created a WCSS plot.

To create this plot, we generated multiple clusters using our dataset (1 to 10) and 

Using the "elbow" method, we can see that the optimal number of cluster to use for our analysis is 5.

#### Correlation Matrix with clusters


![hello](https://cdn.discordapp.com/attachments/579658084595662848/1086649515181867108/image.png)

We created a new correlation matrix including our 5 clusters to see to possible link to some macronutritions values.

For cluster 0, we cannot really see any link with any nutritional values.

For cluster 1, we can see that it contains a huge number of product with a lot of fat, saturated fat and energy and also a lots of sodiums.

We can assumes cluster 1 is for junkfood.

After getting a sample of the products, it confirms our suspicions.
```
Cluster 1 sample:
327083                                Prawn Cocktail Shells
260630                             Mozzarella cheese sticks
328431                        White cheddar captains wafers
24076                                           Jeera Khari
433950                                Croissants Pur Beurre
317311                  Fully Cooked Natural Casing Wieners
66816     Sunshine Cheez-It Crackers Sharp White Cheddar...
99529                                       Sportsman'x mix
133402                                        Corn in a Cip
112417                 Cheese spread made with real cheddar
137712                    Ready Fresh Go Beef Sausage Bites
335048                                               Polish
233519                                               Stacon
245872                           Freybe, herb liver sausage
202680                                          Onion Rings
```

For cluster 2, we can see that it contains a huge number of product contains sugars, carbohydrates and a bit of fat.

We can assumes cluster 1 is for sweets.

After getting a sample of the products, it confirms our suspicions.

```
Cluster 2 sample:
274535                              Tropical trail mix
295838                               Vanilla ice cream
143237                            Ice cream sandwiches
409761                         Cantuccini with almonds
456807                                   Nature Valley
66619                                      The big dig
148441          Culinaria, ice cream, sea salt caramel
277400                                   Cinnamon Roll
59842     Biscoff Cookie & Cookie Butter Ice Cream Bar
457775                          Suisse longue chocolat
369724                    Caramels made with real milk
456722             Chocolate filled shortbread cookies
465957                                Tarte aux poires
47541                                        Ice cream
302658                  Durhams, Ellis Pretzels Yogurt
```

For cluster 3, we can see that it contains a huge number of product that contains carbohydrates and a lot deficit of fat products.

We can't really get an idea of the type of products yet.

```
Cluster 3 sample:
447929                        Baguette Duchesse aux graines
180171                                       Cream crackers
43509           Plsbry heat n go mini pancakes maple burstn
124946                                      Tafte flatbread
326064          Pan-O-Gold Baking Co., Enriched Wheat Bread
37941                                     Fiber bran cereal
304651       Cheddar cheese filling poppers, cheddar cheese
488443                              Braums buttermilk bread
30894                                 Sourdough burger buns
73759                                           Smelt fours
129753    Toasted corn cereal with honey & natural almon...
44601                                          French rolls
326478                                   Panko breading mix
255933                  Pâtes volanti au pesto à l'ail roti
321757                         Honey wheat braided pretzels
```

After looking at the products contained in the cluster 3, we can see that this cluster is filled with cereal derived products (bread, crackers...).

For cluster 4, we can see that it contains a huge number of product with a lot of proteins and are very low on sugar.

We can assume that this is the cluster for meat and meat based products.

```
Cluster 4 sample:
465015            Chipolatas à l'espelette produit du gouet
138200                                    Smoked Baby Clams
467082                                    Top sirloin steak
41348                    Luncheon loaf with chicken & pork.
451299                          Yucatan style chicken thigh
490037                                       Sockeye Salmon
4865                      Italian roasted vegetable ravioli
332760                                Petite stuffed bagels
324181              Excelsior jack mackerel in tomato sauce
125495                                    Surf, Squid Rings
425469                               Saumon Atlantique Fumé
191009                                              Ravioli
346969    Dry-Cured Prosciutto, Mozzarella Cheese & Brea...
446057                      Parfait de Chapon Farci (Cèpes)
335284                               Beef & pork cannelloni
```

After getting a sample of the products, it confirms our suspicions and we can also add the vegetarian alternative of meat to the cluster.

