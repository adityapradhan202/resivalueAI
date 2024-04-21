# resivalueAI
ResivalueAI is a discord bot powered by machine learning and AI which gives you the best house price predictions based on the dataset on which it is being trained! The machine learning model is trained on pseudo data, a data which is valid enough to perform linear and polynomial regression with L1,L2 and elastic regularization!
### Getting started
#### Step by step procedure for creating the machine learning model
1. Select label, and the features.
2. Check if the data is really valid for performing Linear Regression or not.
3. If it is valid then check if polynomial regression will be better or simple linear regression will be better.
4. Choose the degree for polynomial regression.
5. Use graphs to see what degree can cause overfitting.
6. Create an instance of PolynomialFeatures and turn the features into polynimal features.
7. Perform train-test split on X = polynomial_features, y = y_train.
8. Scale X_train and X_test using StandardScaler.
9. Use ElasticnetCV to combine both lasso and ridge regression.
10. Train the model, make predictions, and evaluate performance metrics.
11. Find the best value of l1_ratio and alpha.
12. Perform Cross-Validation to enhance performance.
13. Create and dump final machine learning model and final instance of PolynomialFeatures.

#### For discord bot...
- First go to discord's developer portal to create your discord application. Invite the bot using your URL.
- Then just create a simple discord bot which uses commands to perform some task.  
You can check the discord API documentation here - [Discord API docs](https://discordpy.readthedocs.io/en/stable/api.html)

#### Discord bot image...
![discordbot_ss](https://github.com/adityapradhan202/resivalueAI/blob/main/ss_resivalueAI.png)

#### License:
This project is lincensed under MIT LICENSE. You can check that here - [MIT LICSENSE](https://github.com/adityapradhan202/resivalueAI/blob/main/LICENSE)
