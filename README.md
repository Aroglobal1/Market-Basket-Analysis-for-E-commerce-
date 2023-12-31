# Market-Basket-Analysis-for-E-commerce
### Project Overview
The objective of this project is to use Market Basket analysis, a critical component of data science in retail and e-commerce. The aim is to analyze a Grocery store dataset, extracting valuable insights from transaction data in a bid to understand customer purchasing behavior and optimize business strategies.

### Data Source
The data used for this Analysis is the "Market Basket Analysis - Grocery_dataset.csv" file which contains detailed information on items that were bought from a Grocery Store during a given period of time.

### Tools

- Excel
- Data Analysis Tool: Python (Libraries used are Pandas, Numpy)
- Data Visualization Tool: Matplotlib, Seaborn
- Scikit-Learn
- Jupyter Notebook

### Steps 

These are the various steps involved in the analysis of the data.


#### Exploratory Data Analysis (EDA)

This starts by importing all the necessary modules that are to be used in the course of the analysis.
``` python
# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


Then, read the csv file as a pandas data frame. 
```python
# load the csv file to be used for data analysis
data = pd.read_csv(r"C:\Users\USER\Documents\COURSES\Flit Apprenticeship\Projects\Market Basket Analysis - Groceries_dataset.csv")
data.head()
```
![1](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/92f9bd5b-b87c-4896-a4b5-d641392bf58e)

The dataset consists of detailed information on purchases made at the Grocery store which includes Transaction Date, Member Number and Item Description.

This gives the summary statistics of the data to be analysed.
```python
for col in data.columns:
    print(data[col].describe())
```
![2](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/2e9fc245-44d0-49d8-8507-0c5353b72729)

The chart below shows the first 20 most frequently bought items
```python
# Most Frequent Items Bar plot
color = plt.cm.rainbow(np.linspace(0, 1, 40))
data['itemDescription'].value_counts().head(20).plot.bar(color = color, figsize = (10, 4))
plt.title('Frequency of Most Popular Items', fontsize = 20)
plt.xticks(rotation = 90)
plt.grid()
plt.show()
```
![Frequency of Most Popular Items](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/9fa7796f-a1f1-4286-ac94-c372688f7a23)
From the visualization above, the top 5 most purchased items in the Grocery store include Whole milk, other vegetables, rolls/buns, soda and yogurt.

##### Data Preparation
The data needs to be converted into a format that optimally suits the Apriori algorithm before conducting Market Basket analysis. The Apriori algorithm is used to calculate association rules between items, uncovering the most frequent ones.

To start with, a new column is created which contains the Member number and the transaction date of items purchased.

```python
# Form a new column 
data['Transaction'] = data['Member_number'].astype(str) + '_' + data['Date'].astype(str)
data.head()
![3](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/260ce536-43bd-468d-be51-5982550f5b7d)


Then, pivot the table to convert items to columns while the transactions(new column) to rows so as to observe the items that are bought in each transaction.
```python
# Pivot table to have Items in columns while transactions in rows
data2 = pd.crosstab(data['Transaction'], data['itemDescription'])
data2.head()
```
![image](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/f73f9c8f-20c7-4891-95a9-63748984584b)

We have like 167 columns and 5 rows. It's observed that just few items were bought, this is the reason why there are a lot of zeroes as seen in the table above.
  

The final step here involves encoding all the values in the data frame to 0 and 1.
```python
# Encoding to 0 and 1
def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
Basket_input = data2.applymap(encode)
```

#### Market Basket Analysis
The first step here is to install MLXtend python package and then, import the Apriori Algorithm from the MLXtend in order to obtain the combination of items that are frequently bought together:
```python
# install mlxtend
!pip install mlxtend
```

```python
# import Apriori Algorithm from MLXtend and apply the algorithm to find frequent itemsets
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# convert the DataFrame to boolean type
Basket_input = Basket_input.astype('bool')

# find the frequent itemsets
frequent_itemsets = apriori(Basket_input, min_support = 0.01, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 0.7)
rules.head()
```
![image](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/e3bb908b-4e22-4a2a-bb1a-4a2d6e42045b)

As shown above, the 'antecedents' and 'consequents' columns reveal items that are frequently purchased together. In the first row, it shows that an individual that bought "other vegetables" is also likely to buy rolls/buns.

In order to get the most frequently bought-together combined items, there is need to sort the dataset by support, confidence and lift.
```python
# sorting the frequent itemsets bought together
rules.sort_values(["support", "confidence", "lift"], axis = 0, ascending = False).head(10)
```
![image](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/6413cd79-7abf-4936-9bff-5600c4b54136)

In the table above, the four items combination that are frequently bought together the most are:
- other vegetables and whole milk
- roll/buns and whole milk
- soda and whole milk
- yogurt and whole milk

Whole milk was the most frequently bought item but along with other items for different purposes. 


#### Visualization

These are the visualizations of the dataset to help discover the items that are frequently bought together in order to make a better decision in promoting the market items in the Grocery Store.
```python
# transform antecedent, consequent, and support columns into matrix
support_table = rules.pivot(index = 'consequents', columns = 'antecedents', values = 'support')

# Creating a heatmap visualization to display the consequents and antecedents
plt.figure(figsize = (10, 6))
sns.heatmap(support_table, annot = True, cbar = True)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.yticks(rotation = 0)
plt.title('Heatmap of Frequently Bought Items')

plt.show()
```
![image](https://github.com/Aroglobal1/Market-Basket-Analysis-for-E-commerce-/assets/148555924/cb1d875f-b4aa-455d-8749-1d084abf258b)


The Heatmap displayed above illustrates the correlation between frequently bought items in a Market Basket analysis.

#### Interpretation and Insights

The heatmap visualization shows the correlation between items that were frequently bought together in a Market Basket analysis of a Grocery store dataset. On the x-axis, we have the antecedents of the rules, while the y-axis represents the consequents.

The coloring of the heatmap is determined by the support of the rules. The darker the color of a cell, the higher the correlation between the two items, indicating that they are frequently purchased together.

Analyzing the heatmap, it is evident that other vegetables are commonly bought alongside rolls/buns, and whole milk is frequently purchased in conjunction with yogurt, soda, and rolls/buns. The varying shades on the heatmap provide a visual representation of these correlations, offering valuable insights into the prevalent shopping patterns within the dataset.


#### Recommendations

The Grocery store could enhance its recommendations by suggesting Rolls/buns to customers who typically purchase other vegetables. Additionally, it might be beneficial to recommend yogurt, soda, and rolls/buns to customers who buy whole milk.

Furthermore, the store could optimize its target market campaigns by focusing on items such as yogurt, soda, and rolls/buns that are likely to be frequently purchased alongside other vegetables and whole milk. This strategic approach aims to capitalize on the observed buying patterns and preferences of the customer base
