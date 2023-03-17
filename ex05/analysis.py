import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, scale

class Analysis:
    def create_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep='\t', low_memory=False, nrows=self.nb_rows) 
        return df

    def init_z_score(self):
        res = {}

        res['energy-kcal_100g'] = 3
        res['fat_100g'] = 3
        res['saturated-fat_100g'] = 3
        res['carbohydrates_100g'] = 3
        res['sugars_100g'] = 3
        res['proteins_100g'] = 3
        res['sodium_100g'] = 3

        return res

    def __init__(self):
        self.path = 'dataset/dataset.csv'
        self.nb_rows = 100000
        self.z_score = self.init_z_score()
        self.quants_vars = ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'sodium_100g']
        self.df = self.create_dataframe()
        print(f"Number of rows: {len(self.df)}")
        self.clean_dataframe()
        print(f"Number of rows after cleaning: {len(self.df)}")
        # scale the dataframes 
    
    def clean_dataframe(self):
        self.df = self.df.dropna(subset=self.quants_vars)
        self.df = self.df.drop_duplicates(subset=self.quants_vars)
        self.remove_outliers()

    
    def detect_outliers(self) -> pd.DataFrame:
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        ax_idx = [0, 0]
        # Detect outliers using z-score and display plot for each variable with outliers detected
        # display 4 plots per row
        
        count = 0
        for col in self.quants_vars:
            if count == 4:
                plt.show()
                ax_idx = [0, 0]
                fig, ax = plt.subplots(2, 2, figsize=(30, 20))
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            
            IQR = Q3 - Q1
            

            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            ax[ax_idx[0], ax_idx[1]].scatter(self.df.index, self.df[col], color='blue')
            ax[ax_idx[0], ax_idx[1]].scatter(outliers.index, outliers[col], color='red')
            ax[ax_idx[0], ax_idx[1]].set_title(col)
            ax[ax_idx[0], ax_idx[1]].set_xlabel('Index')
            ax[ax_idx[0], ax_idx[1]].set_ylabel(col)
            


            if ax_idx[1] == 1:
                ax_idx[0] += 1
                ax_idx[1] = 0
            else:
                ax_idx[1] += 1
            count += 1
        plt.show()

    def remove_outliers(self) -> pd.DataFrame:
        for col in self.quants_vars:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            
            IQR = Q3 - Q1
            self.df = self.df[(self.df[col] >= Q1 - 1.5 * IQR) & (self.df[col] <= Q3 + 1.5 * IQR)]


    def print_statistics(self) -> None:
        print(self.df[self.quants_vars].describe())

    def get_historigram(self) -> None:
        self.df[self.quants_vars].hist(bins=50, figsize=(20,15))
        plt.title("Histograms")
        plt.show()

    def get_boxplot(self) -> None:
        sns.boxplot(data=self.df[self.quants_vars], orient="v", palette="Set2")
        plt.title("Boxplot")
        plt.show()

    def get_correlation_matrix(self) -> None:
        corr_matrix = self.df[self.quants_vars].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation matrix")
        plt.show()

    def wcss_analysis(self) -> None:
        wcss = []

        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=10)
            kmeans.fit(self.df[self.quants_vars])
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, 11), wcss)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS') 
        plt.show()

    def corr_matrix_kmean4(self) -> None:
        # Show the correlation matrix for the 4 clusters
        n_clusters = 5
        model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
        x = self.df[self.quants_vars].copy()

        model.fit(x)
        x['cluster'] = model.labels_
        # print sample of each cluster with only product name

        for i in range(n_clusters):
            print("--------------------------------------------------")
            print(f"Cluster {i} sample: ")
            print(self.df[x['cluster'] == i]['product_name'].sample(5))
            print("--------------------------------------------------")
            
        cluster_0 = x[x['cluster'] == 0]
        cluster_1 = x[x['cluster'] == 1]
        cluster_2 = x[x['cluster'] == 2]
        cluster_3 = x[x['cluster'] == 3]
        cluster_4 = x[x['cluster'] == 4]

        

        var = self.quants_vars.copy()
        var.append('cluster_0')
        var.append('cluster_1')
        var.append('cluster_2')
        var.append('cluster_3')
        var.append('cluster_4')

        df = pd.DataFrame(columns=var)
        df.loc[0] = [cluster_0[col].mean() for col in self.quants_vars] + [1, 0, 0, 0, 0]
        df.loc[1] = [cluster_1[col].mean() for col in self.quants_vars] + [0, 1, 0, 0, 0]
        df.loc[2] = [cluster_2[col].mean() for col in self.quants_vars] + [0, 0, 1, 0, 0]
        df.loc[3] = [cluster_3[col].mean() for col in self.quants_vars] + [0, 0, 0, 1, 0]
        df.loc[4] = [cluster_4[col].mean() for col in self.quants_vars] + [0, 0, 0, 0, 1]

        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.title("Correlation matrix")
        plt.show()

    def scale_dataframe(self) -> None:
        self.df[self.quants_vars] = scale(self.df[self.quants_vars])

    def run(self) -> None:
        """
        Use this function to run your analysis
        Uncomment the functions you want to use
        If you want to do analysis with an unclined dataframe, comment the line self.clean_dataframe()
        For the outliers detection, you can change the z-score in the init_z_score function
        You can also change the number of rows to read in the init function
        """
        self.print_statistics()
        # self.detect_outliers()
        # self.get_historigram()
        #self.get_boxplot()
        # self.get_correlation_matrix()
        
        self.scale_dataframe()
        self.wcss_analysis()
        self.corr_matrix_kmean4()


if __name__ == '__main__':
    analysis = Analysis()
    analysis.run()
    



