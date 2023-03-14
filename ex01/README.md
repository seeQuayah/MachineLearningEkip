## Part 1 : data distribution and the law of large numbers

#### Question 1

We take as an example the height and weight of a population.
We consider the real random variables X (height in cm) and Y (weight in kg) for each individual of the population.

We assume that X follows a normal distribution with mean 170 cm and standard deviation 10 cm, and that Y follows a normal distribution with mean 70 kg and standard deviation 5 kg.

Z = (X, Y) represents the height and weight of an individual in the population.

Z is equal to the means of X and Y, so E[Z] = (E[X], E[Y]) = (170, 70).


#### Question 2

We sample n = 1000 points of the law of Z and represent them in a 2-dimensional figure:

![Distribution for Height and Weight for sample n = 1000](https://cdn.discordapp.com/attachments/496721203176800287/1081215098711986258/ex01_02.png)


#### Question 3

We now compute the empirical mean of the first n samples as a function of the number of samples n, and check that it converges to the expected value.

!['Distance from true mean for sample n = 1000](https://cdn.discordapp.com/attachments/496721203176800287/1081217194723119104/ex_01_03.png)

We find that the Euclidean distance decreases rapidly at first and then slowly converges to zero, confirming that the empirical mean of the samples does converge to the expected value of Z.
