# Corporacion-Favorita-Grocery-Pre-Sales-Forecasting
This is a group project wrote by Chaoyang zheng, Qiuhua Liu and Yanhan Peng for PHD level course: Machine Learning for Large-Scale Data Analysis and Decision Making in HEC MONTREAL.
![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/0.png
 "Optional title")

## 1. Project question 
In a competitive business environment, companies always want to increase sales and improve customer service by better preparation of their products. Usually companies forecast the future demand based on historical sales data, and then coordinate each department to meet predicted demand. However, companies can’t always be in such cases, especially when they are going to release a new product into the market. It is impossible to forecast demand with the traditional ways.  

In our example, Favorita Grocery runs multiple grocery stores. We plan to forecast the demand for a specific item in a specific store before sales season in month level, which helps the specific store to determine the amount of item in advance, therefore, to reduce the rate of out of stock and increase the revenue and customer satisfaction. We choose forecasting in month level because it will be much more practical than in daily level. 
 
## 2. Literature Review: 
### 2.1 A comparative study on the forecast of fresh food sales. 
With the changing of the structure of society and households, the way fresh food is sold in the market is rapidly changing. Freshness and rapid speed of turnover are important consideration. Fresh food is characterized by two factors: its short shelf-life and its importance as a revenue producer for convenience stores. Reliable prediction of sales is of immense benefit to a business because it can improve the quality of the business strategy and decrease costs due to waste, thereby increasing profit(Wan-I Lee et al. 2012). 
 
This paper focus on making this type of prediction with different forecasting sales models, such as Logistic Regression, a good choice for binary data, the Moving Averagemethod, a good way for simple prediction, and the Back-Propagation Neural Networkmethod, a good selection for long term data(Wan-I Lee et al. 2012). 
 
Taiwan’s Hi-Life convenience store chain is selected as the research subject in this paper. The ordering cycle is defined as daily and the sales data included number of sales and amount of fresh food discarded. Furthermore, the information of weather data, holidays, marketing actions, promotions, fashion, and economic environment can be used as explanatory variables. Besides authors used the correct forecast percentage and define error in the correct percentage as the incorrect percentage to calculate the error(Wan-I Lee et al. 2012). 
 
The result in this paper indicates that the correct percentage obtained by LR to be better than that obtained by the BPNN and MA models. After reading this paper, we select four different models in our project: Nearest Neighborhood, Linear Regression, Neural Network. 
 
### 2.2 The use of analogies in forecasting the annual sales of new electronics products. 
Forecasting the sales of products that have yet to be launched is an important problem for companies. In particular, forecasts of the future values of sales time series (e.g. sales in each of the first n years of a product’s life) will guide decisions relating to future production capacity, marketing budgets, human resource planning and research and development. These forecasts can also be used to estimate the discounted future returns on the investment that will be needed to develop and market the new product. (Goodwin et al. 2013) 
 
In this paper, Bass model was used to describe the sales and adoption patterns of products in the years following their launch and one of the most popular of these models. However, using this model to forecast sales time series for new products is problematical because there is no historic time series data with which to estimate the model’s parameters. Thus, the authors used analogies to estimate parameters for the bass model, which means fitting the model to the sales time series of analogous products that have been launched in an earlier time period and to assume that the parameter values identified for the analogy are applicable to the new product(Goodwin et al. 2013). 
 
The result indicates that all of the methods tended to lead to forecasts with high absolute percentage errors, which is consistent with other studies of new product sales forecasting. The use of the means of published parameter values for analogies led to higher errors than the parameters estimated from the data (Goodwin et al. 2013). This paper gave us the idea to build a model that more accurately forecasts the demand of items before sales according to the known sales of the same category items and help the stores stock the amountof items accurately. 
 
## 2.3 Analogue-based demand forecasting of short life-cycle products: a regression approach and a comprehensive assessment 
For short life-cycle products, they increased companies’ competitions, but their demands were also difficult to forecast due to high variability in demand and scarcity in historical sales data.Authors aimed to solve the forecast problemby using the time series of similar products. 
 
The authors designed data set X with N time series of analogous products, and the length of each time series is greater than or equal to t time periods. They also assume each time series has AR relationship. Authors divided the whole forecast process into two parts. In the first part, authors determined the hyper parameter, such as the number of clusters and the number of lags in AR regression, then did clustering and obtained the MLR coefficients by minimizing the least squares problem weighted by the fuzzy weights for each cluster. In the second part, authors calculated the regression estimate demand in time period t of the current time series for each cluster with the obtained coefficients. Then they assigned the current time series to the cluster with the smallest distance measure. Based on the choice, they calculated the final forecast for time period t.  
 
With various experiments, this algorithm achievedmore accurateforecasts with short processing times compared with state-of-the-art methods. Those results also revealed that the combination of clustering and regression is a simple and eﬀective forecasting toolsfor supporting replenishment decisions for SLCP. 

## 3. Data description feature engineering  
The data comes in the shape of multiple files, while in generally, they can be divided into 2 parts: The main sales data andsupplementary features data.  
Themain sales data essentially contains the sales by date, store, and item.In our project, to reduce training complexity and increase the prediction accuracy, we aggregate the daily item unit sales to monthly item unit sales, byitemid and store id.

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/1.png
 "Optional title")

Plot: Boxplot for the average monthly sales of items among stores (Please noted that each point is the average monthly sales of one item in a specific store) 

The supplementary features data included several features related to the daily item sales among stores. In generally, those features come from 4 dataset: Store, Items, Holidays & Events and Transactions. All features are merged with our main sales data (monthly), through either dates, stores number oritems number. The table below shows the usage of features selected for this project. 

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/2.png
 "Optional title")
Table: Usage of features selected for this project
 
## 4. Proposed model description 
### 4.1.General introduction for proposed model 
We design an algorithm based on the similarity as well as regression or neural network to forecast demand of the product before sales season. 

We first apply clustering to select one or multiple products based on similarity between new released products and ever-sold products. Second, if we just choose one similar product, then its sales data would be directly used as the demand for the target product. But if we choose multiple similar products, we randomly split them into 2 parts- training set and validation set. In validation set, we randomly select one product, whereas the training set has the rest where we expect to estimate the parameters by considering those similar products together. 

There are two noticeable point in our algorithm. First, we use the unchangeable features to cluster and select similar products, which is easy to apply and needs less effort to do market research. The second point is that we consider more features, such as promotion, holidays and so on, to fit the model and forecast. 

Further, we will clearly introduce the main models or algorithms we used. 

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/3.png
 "Optional title")
 
### 4.2 Benchmark model: nearest neighborhood 
Nearest neighbor search (NNS), as a form of proximity search, is the optimization problem of finding the point in a given set that is closest (or most similar) to a given point. Closeness is typically expressed in terms of a dissimilarity function: the less similar the objects, the larger the function values. NNS was used as benchmark in our project to make comparison with other models. 
 
In order to use NNS predicting the sale of our target item, first, we need to combine our data to acquire the new data, which only includes information of all features of each item sold in the CorporaciónFavorita Grocery. After dropping some attributes, such as year month and unit sales, and aggregating the same items, we got the new dataset, named items_for_clustering.csv. And this new 
dataset includes the values of store_nbr, store_city, store_state, store_type, store_cluster, item_family, item_class, item_perishable. And all these values represent the features of items. 
 
By looking at the items_for_clustering.csv, we can see that all the features are non-continuous. Thus we chose hamming distance to process these categorical features and calculate the distance between each item and our target item. Finally selecting the item which has the smallest hamming distance with our target item and assigning the sale of the item to our target item as a prediction value.  
 
We import hamming distance from scipy.spatial. The Hamming distance between 1-D arrays u and v, is simply the proportion of disagreeing components in u and v. If u and v are boolean vectors, the Hamming distance is , where 𝑐 is the number of occurrences of u[k]=i and v[k]=j for for k<n. hamming distance also has a hyperparameter w, the weights for each value in u and v, and default is None. In our feature dataset, the hamming distance is the number of different features of items dividing the total number of features. In our code, we used a loop to compare all items with our target item, and finally got the item with 0 hamming distance. 
 
### 4.3 Step1 Clustering  
This project starts from implementing clustering method to assign “items + stores” into multiple clusters.  The models selected is K-means clustering with one hot encoding and K-modes. Since we are doing a real-life clustering, some variables may have stronger impacts to build cluster, therefore selected features for clustering should be weighted. Based on the official data description, all variables are equally weighted, except for features: perishable and locations of stores (both states and cities), which are assigned with higher weights. 
 
####### Clustering Model 1:  K-means clustering with one hot encoding 
The standard k-means algorithm isn't directly applicable to categorical data, for various reasons. The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space isn't really meaningful. Thus, our first model for cluttering is: K-means clustering with one hot encoding, which convert categorical attributes to binary values, and then doing k-means as if these were numeric values. 
 
We use the elbow plot to choose the optimal K in the proposed K-Means model.  

 ![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/4.png
 "Optional title")
 
The above graph show that the “elbow” method does not work quite well in our data, and a rather smooth curve was observed, while the optimal value of K is unclear. However we may still notice that there is “big jump” in the “sun of squared distance” (though not very significant) when the value of k equals to 10, hence we select 10 as our optimal number of cluster. 
 
###### Clustering Model 2:  K-modes 
The K-modes model with a section on "k-prototypes" make it an ideal model to be applied to data with a mix of categorical and numeric features. It uses a distance measure which mixes the Hamming distance for categorical features and the Euclidean distance for numeric features. 

  ![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/5.png
 "Optional title")
 
Similar to the result of K-means model, the “elbow method” also works not so well in K-modes, item_nbr:1083152 (specific) cluster as our target item, namely our test data. However, we noticed that for each value of K, the K-modes model return slightly small values compared with K-means (with on hot encoding). In other words we can conclude that in generally the K-modes model performs better than the K-means model (with one-hot encoding) in our data. 
 
### 4.4 Step2 Forecasting 
Based on the results of clutering, we plan to use two easily used models- linear regression and neuron network. 

###### Forecasting Model1: Linear regression 
Linear regression is the most commonly used algorithm. With it, we want to model the linear relationship between response variable (unit_sales) and multiple explanatory variables, such as holiday, promotion, item_perishable and so on. According to the criteria of least squard error, We got estimated values of parameters for the fitted model. Furthermore, we applied the model to do forecast with the new values of explanatory variables of the target product. 
 
###### Forecasting Model2: Neural network 
Unlike linear regression, by neural network we want to model the non-linear relationship between response variable (unit_sales) and multiple explanatory variables. According to knowledge, we need to tune various hyperparameters first, such as the number of hidden layers, the number of neurons in each hidden layer and the activation functions. We use for loop to test different combination of hyperparameters and choose values or setting based on the best value in validation set. 
 
### 5. Model comparison 
### 5.1 Clustering model comparison: 
As it’s difficult to found the optimal value of K in K-modes, we select K = 10 for K-modes to make its results comparable to K-means. The sum of squared distances of those two models can be seen from the table below. As discussed earlier, theK-modes model generally performs better than the K-means model, therefore we select K-model (with k = 10) as our clustering method, and used its clustering results for our following forecasting approach:

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/6.png
 "Optional title")
 
 ### 5.2 Forecasting model comparison: 
 ![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/Corporaci-n-Favorita-Grocery-Pre-Sales-Forecasting/master/graph/7.png
 "Optional title")
 
The result of NNS: After clustering all items in the CorporaciónFavorita Grocery, we choose the item of 
item_nbr:1083152 sold in 40 store as our target item, namely our test data. Then we ran the code of loop in hamming distance and found the item of item_nbr:164088 sold in 40 store has the smallest hamming distance 0.111 with the target item: 1083152, which means this item is most similar to our target item. Thus, we can predict the sale of our target item through assigning the value of the similar item to it. Finally, we calculate the mean square error in the test dataset and the value is 24288. 
 
The result of Liner regression and Neural network: Based on the same dataset, either in trainset, validation set and test set, we run multiple linear regression and neural network on them. We can see both of them offered better results than nearest neighbouhood- the benchmark- in any dataset. So we think clustering first is a better choice. But the comparison between linear regression and neural network shows us an unexpected result. The simpler model linear regression gives a better result in all of dataset, which somehow represent a ‘good’ result for business because it is easier to apply and need less advanced analytical skill.   
 
## 6. Conclusion: 
This project applied an approach that combines a clustering method and forecasting method to implement a presale forecasting using analogies. The result shows that the linear regression model has the better performance between these three models. However, all results indicate that all of the methods tended to lead to forecasts with high mean square errors. On the one hand, this may related to the fact that our clustering models don’t have good performances and the selected forecasting models are unable to explain all selected features. On the other hand, this may also give us some hints that the use of the values of analogies led to higher errors than the values estimated from the target data. Because our goal is to forecast the sale of new item, which means this is a presale forecasting, we don’t have historic time series data of our target item. Thus, we couldn’t establish a good model fitting on a nonexistent data. From our project, we found the sales time series for these analogous items can then be used to identify a forecasting model that it is hoped will accurately represent the future sales pattern of the product which is due to be launched. 
 
Furthermore, in our algorithm, we consider a general case for pre-sales. But in reality, we can go deeper. Based on the knowledge of product preparation process, we can decide different forecast time horizon according to the predicted item, which will be more reasonable and meaningful. 
 
# Reference :

Wan-I Lee, Cheng-Wu Chen, Kung-Hsing Chen, Tsung-Hao Chen and Chia-Chi Liu. “A comparative study on the forecast of fresh food sales using logistic regression, moving averge and BNPP methods”. Journal of Marine Science and Technology, Vol. 20, No. 2, pp. 142-152 (2012) 
 
Goodwin, P., K. Dyussekeneva, and S. Meeran. 2013. “The use of analogies in forecasting the annual sales of new electronics products.” IMA Journal of Management Mathematics 24 (4): 407-422. 
 
Huang, Z. Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values. Data Mining and Knowledge Discovery 2, 283–304 (1998) doi:10.1023/A:1009769707641 
 
Mario Jose Basallo-Triana and Jesus Andres Rodrıguez-Sarasty and Hernan Darıo Benitez-Restrepo:Analogue-based demand forecasting of short life-cycle products: a regression approach and a comprehensive assessment,International Journal of Production Research · October 2016 
