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