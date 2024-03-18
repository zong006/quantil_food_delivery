The task here is to use quantile regression to predict and upper and lower estimates of delivery times for a food delivery service. 

#### Quantile Regression for Food Delivery

A typical regression line uses the OLS (ordinary least squares) method, fitting to the mean of the distribution ie, it is a mean regression line. The OLS loss function is given by 

$$
L(y, \hat{y}) = \sum_{i}(y_i - \hat{y}_i)^2
$$

where $L$ is the loss, $\hat{y}_i$ is the predicted value and $y_i$ is the true value.

In quantile regression however, the aim is to have a regression line such that for a given quantile $\tau$, the number of true values lying below the regression line is $100 \times \tau$%. This is achieved by fitting the regression line on the loss function

$$
L(\tau, y, \hat{y}) = \sum_{i} \max\{(\tau - 1)(y_i - \hat{y}_i), \tau(y_i - \hat{y}_i)\}.
$$

As an example, the plot below shows three regression lines at quantiles $\tau = 0.1, 0.5, 0.9$. Roughly speaking, a regression line at 0.9 quantile aims to overestimate 90% of the time, and vice-versa for regression lines at other quantiles. It can be seen from the plot that the $\tau = 0.9$ line approximately divides the data points into 10% above the line, and 90% below it.
![Quantile Regression Example](quantil_eg.jpeg)


In the context of a food delivery service, we want to give an upper and lower bound of the estimated delivery time for the customer, giving us a prediction interval for the delivery time. Here, the quantiles of $\tau = 0.9$ and $\tau = 0.1$ are used, such that giving the upper and lower estimate for the delivery time, respectively.

Note that quantile regression is more robust to skewed distributions and outliers as compared to OLS (mean) regression, since it focuses on quantiles rather than the mean.


#### DataSet 
The dataset used here for is from the following Kaggle URL :

https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset

Variables are as follows:

| Variable                  | Description                                      |
|---------------------------|--------------------------------------------------|
| ID                        | Delivery order ID                                |
| Delivery_person_ID        | ID of the delivery person                        |
| Delivery_person_Age       | Age of the delivery person                       |
| Delivery_person_Ratings   | Ratings of the delivery person                   |
| Restaurant_latitude       | Latitude of the restaurant                       |
| Restaurant_longitude      | Longitude of the restaurant                      |
| Delivery_location_latitude| Latitude of the delivery location                |
| Delivery_location_longitude| Longitude of the delivery location              |
| Order_Date                | Date of the order                                |
| Time_Orderd               | Time when the order was placed                   |
| Time_Order_picked         | Time when the order was picked up                |
| Weatherconditions         | Weather conditions (e.g., sunny, stormy)          |
| Road_traffic_density      | Density of road traffic                          |
| Vehicle_condition         | Condition of the vehicle                         |
| Type_of_order             | Type of order (e.g., snack, drinks)              |
| Type_of_vehicle           | Type of vehicle used for delivery(e.g., motorcycle, scooter)                |
| multiple_deliveries       | Number of deliveries in that trip                 |
| Festival                  | Indicator for festival days                      |
| City                      | City where the delivery took place               |
| Time_taken(min)           | Time taken for delivery (in minutes)             |



#### Choice of models and evaluations

A simple neural network in PyTorch is used, where there is one hidden layer and one output layer. Three different quantiles are used,  $\tau = 0.9$ and $\tau = 0.1$ for upper and lower estimates of the delivery time, as mentioned, as well as the median $\tau = 0.5$ for reference. A k-fold cross validation is also used for a better assessment of the model performance.

To evaluate the model, we can take it that the model should generalize as well on the test data as it does on the training data. In other words, the regression lines at quantiles $\tau = 0.9$, $\tau = 0.5$ and $\tau = 0.1$ should ideally overestimate 90%, 50%, and 10% of the time, respectively. The overestimation percentages of the models for the three quantiles are as follows:

|Quantile| Train  | Test   |
|--------|--------|--------|
|10%     | 8.47   | 11.47  |
|50%     | 49.87  | 49.99  |
|90%     | 92.99  | 90.01  |
    

The models for the three quantiles perform generally well on both the train and test dataset. 



