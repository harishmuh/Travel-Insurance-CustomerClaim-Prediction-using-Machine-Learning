![image](https://images.unsplash.com/opengraph/1x1.png?blend=https%3A%2F%2Fimages.unsplash.com%2Fphoto-1530521954074-e64f6810b32d%3Fblend%3D000000%26blend-alpha%3D10%26blend-mode%3Dnormal%26crop%3Dfaces%252Cedges%26h%3D630%26mark%3Dhttps%253A%252F%252Fimages.unsplash.com%252Fopengraph%252Fsearch-input.png%253Fh%253D84%2526txt%253Dtravel%252Binsurance%2526txt-align%253Dmiddle%25252Cleft%2526txt-clip%253Dellipsis%2526txt-color%253D000000%2526txt-pad%253D80%2526txt-size%253D40%2526txt-width%253D660%2526w%253D750%2526auto%253Dformat%2526fit%253Dcrop%2526q%253D60%26mark-align%3Dmiddle%252Ccenter%26mark-w%3D750%26w%3D1200%26auto%3Dformat%26fit%3Dcrop%26q%3D60%26ixid%3DM3wxMjA3fDB8MXxzZWFyY2h8Nnx8dHJhdmVsJTIwaW5zdXJhbmNlfGVufDB8fHx8MTcxNTA4MjAxNXww%26ixlib%3Drb-4.0.3&blend-w=1&h=630&mark=https%3A%2F%2Fimages.unsplash.com%2Fopengraph%2Flogo.png&mark-align=top%2Cleft&mark-pad=50&mark-w=64&w=1200&auto=format&fit=crop&q=60)

# **Travel Insurance Claim-Prediction using Machine Learning**
Travel Insurance claim prediction using Adaptive Booster

## **Problem statement**

Maintaining a sufficient balance between customer premiums with claims and expenses is a crucial issue in the travel insurance industry. The enterprise must be able to calculate the number of premiums that need to be collected and the number of claims that they will be able to handle. The travel insurance enterprise needs to be able to calculate the risk accurately. The ability to predict customers who are claiming and not claiming, and calculate the probability of it will help them to manage risks. By being able to distinguish customers who are likely to claim and not claim, insurers can make better planning in the allocation of their funds, optimize their operations, potentially lower operational costs, and overall improve profitability.

**Goal**

Based on the previous problem statement, an international travel insurance enterprise wants to initiate predictive modeling to improve its risk assessment system. This model should be able to predict which one of their customers is more likely to claim the insurance based on the given historical data. In addition to that, we want to know which risk factors or variables that are crucial in increasing customer probability to claim the insurance.

**Analytical approach**

We want to analyze data to learn about patterns that can differentiate customers based on who will claim the insurance and who will not. We will build classification models to help the travel insurance enterprise predict which policyholders that is more likely to claim travel insurance and which policyholders won't. After that, We will compare the classification model using nine different machine learning models consisting of Logistic Regression, AdaBoost, Gradient Booster, LightGBM, CatBoost, XGBoost, Random Forest, Decision Tree, and KNN. Next, we will conduct hyperparameter tuning on the three best models and compare the performance metrics using ROC-AUC. 


**Dataset source**

The travel insurance dataset (private) can be accessed through this [link](https://drive.google.com/file/d/1emDTGFvku7UuuVT3W-EmEvg3i61QrIU-/view)




## **Results**

**Best Model**
* Our best model based on the result of the ROC AUC score, classification report, and confusion matrix is the tuned AdaBoost model that demonstrates a high reduction in both false positives and false negatives compared to the logistic regression and gradient boosting model. This model has an ROC AUC score of 0.715 which indicates 71.5% in distinguishing claim and non-claim. while this is not perfect, it provides a reasonable level of discrimination ability, particularly in the context of a highly imbalanced dataset.
![Model Perfomance](https://github.com/harishmuh/Travel-Insurance-CustomerClaim-Prediction-using-Machine-Learning/blob/main/adaboost_performance.PNG)
	
* The best parameter of the tuned AdaBoost model consists of n_estimators (180), learning_rate (0.01), and max_depth (2)

**Feature Importance**
* Based on feature importance, features that are crucial for the AdaBoost model travel insurance claim consist of 'Agency_3', 'Net Sales', 'Commission (in value)', 'Product Name_3', and 'Destination_3'. This suggests that specific agencies, financial metrics, and insurance product types, play an important role in the claim prediction model.
![Feature Performance](https://github.com/harishmuh/Travel-Insurance-CustomerClaim-Prediction-using-Machine-Learning/blob/main/feature%20importance.PNG)

**LIME for Model Explanation**
* Based on local interpretation using LIME (Local Interpretable Model-agnostic Explanations) for the AdaBoost model,
  * 'Net Sales' & 'Commision in (Value)': A higher value of these features contributes positively to the likelihood of a claim.
  * 'Duration' or 'Duration Group': Longer coverage durations might increase the likelihood of claims due to the extended period of risk exposure.
  * 'Destination': A specific destination like 'Singapore' is associated with more claims.

**LIME for the positive or claiming customer prediction**
![positive](https://github.com/harishmuh/Travel-Insurance-CustomerClaim-Prediction-using-Machine-Learning/blob/main/lime%20interpretation%20positive.PNG)

**LIME for the negative or non-claiming customer prediction**
![negative](https://github.com/harishmuh/Travel-Insurance-CustomerClaim-Prediction-using-Machine-Learning/blob/main/lime%20interpretation%20negative.PNG)

## **Recommendations for Machine Learning model**

* Further investigate and research features that might improve model performance and lower overfitting.

* Perform a more thorough hyperparameter tuning using techniques such as Grid Search to fine-tune the model parameters and achieve better performance.

* Adding features such as a policy price that potentially improve the model and can be explained better.

* Beyond SMOTE, explore other resampling techniques such as ADASYN or SMOTE-ENN to handle the extreme class imbalance more effectively.

* Implement cost-sensitive algorithms that assign different penalties to misclassification errors of the minority class (claims) and the majority class (non-claims).

* Trying to explore other machine learning metrics as standard to find ways for improving precision and further reducing False Positive and True Negative.

## **Assets**
* [Presentation (PDF)](https://github.com/harishmuh/Travel-Insurance-CustomerClaim-Prediction-using-Machine-Learning/blob/main/Travel%20Insurance%20-%20Claim%20Prediction.pdf)
* [ML & Data Analyst Notebook](https://github.com/harishmuh/Travel-Insurance-CustomerClaim-Prediction-using-Machine-Learning/blob/main/travel_insurance%20%20-%20Harish%20Muhammad.ipynb)


