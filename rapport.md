
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

Here is the code:

```py
import  numpy  as  np
import  matplotlib.pyplot  as  plt

mean_x  ,  sigma_x  ,  mean_y  ,  sigma_y,  n  =  170  ,  10  ,  70  ,  5,  1000

x  =  np.random.normal(mean_x,  sigma_x,  n)
y  =  np.random.normal(mean_y,  sigma_y,  n)
z  =  np.column_stack((x,  y))

plt.scatter(x,  y)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Distribution of Height and Weight')
plt.show()
```

#### Question 3

We now compute the empirical mean of the first n samples as a function of the number of samples n, and check that it converges to the expected value.

!['Distance from true mean for sample n = 1000](https://cdn.discordapp.com/attachments/496721203176800287/1081217194723119104/ex_01_03.png)

We find that the Euclidean distance decreases rapidly at first and then slowly converges to zero, confirming that the empirical mean of the samples does converge to the expected value of Z.

Here is the code:
```py
import  numpy  as  np
import  matplotlib.pyplot  as  plt

mean_x,  sigma_x,  mean_y,  sigma_y,  n  =  170,  10,  70,  5,  1000

z_samples  =  np.random.normal(loc=[mean_x,  mean_y],  scale=[sigma_x,  sigma_y],  size=(n,  2))

mean_distances  =  []
for  k  in  range(1,  n+1):
	means  =  np.mean(z_samples[:k],  axis=0)
	distance  =  np.linalg.norm(means  -  [mean_x,  mean_y])
	mean_distances.append(distance)

plt.plot(range(1,  n+1),  mean_distances)
plt.xlabel('Number of samples')
plt.ylabel('Distance from true mean')
plt.show()
```

## Part 2: Meteorological data : dimensionality reduction and visualization


We choose to use PCA like in the classes.

Here is the plot for the 2D:
![enter image description here](https://cdn.discordapp.com/attachments/1080232339751325736/1081276288553193573/ex_02_pca_center_2D.png)

Here is the plot for the 3D:
![enter image description here](https://cdn.discordapp.com/attachments/1080232339751325736/1081276288892936282/ex_02_pca_center_3D.png)

It appear that the third dimension allow better predictions.

Here is the code:
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load('data.npy')
labels = np.load('labels.npy')

pca_2d = PCA(n_components=2)
reduced_data_2d = pca_2d.fit_transform(data)

x_data = reduced_data_2d[:, 0] - np.mean(reduced_data_2d[:, 0])
y_data = reduced_data_2d[:, 1] - np.mean(reduced_data_2d[:, 1])
plt.scatter(x_data, y_data, c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D')
plt.show()


pca_3d = PCA(n_components=3)
reduced_data_3d = pca_3d.fit_transform(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_data = reduced_data_3d[:, 0] - np.mean(reduced_data_3d[:, 0])
y_data = reduced_data_3d[:, 1] - np.mean(reduced_data_3d[:, 1])
z_data = reduced_data_3d[:, 2] - np.mean(reduced_data_3d[:, 2])
ax.scatter(x_data, y_data, z_data, c=labels)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D')
plt.show()
```

## Part 4 : exploitation/exploration compromise

```py
from agent import Agent

BASE_POSITION = 0

def check_if_reward(map: list) -> int:
    max_reward = max(map)
    if max_reward != 0:
        for i in range(len(map)):
            if map[i] == max_reward:
                return i
    return -1

def l_o_r(actual : int, obj: int) -> str:
    if actual == obj:
        return "none"
    elif obj > actual:
        return "right"
    else:
        return "left"

def ekip_policy(agent: Agent) -> str:
    actions = ["left", "right", "none"]
    action = ""

    if agent.position == 0:
        BASE_POSITION = 7
    else:
        BASE_POSITION = 0

    pos = check_if_reward(agent.known_rewards)

    if pos != -1:
        action = l_o_r(agent.position, pos)
    else:
        action = l_o_r(agent.position, BASE_POSITION)
    
    assert action in actions
    return action
```

Average score 27