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