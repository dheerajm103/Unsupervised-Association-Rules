import pandas as pd                                                   # importing libraries
from mlxtend.frequent_patterns import apriori , association_rules
from matplotlib import pyplot as plt
df = pd.read_csv("my_movies.csv")                                     # importing data set
df

# Data cleansing
df1 = df.drop(["V1","V2","V3","V4","V5"], axis = 1)                   # dropping nominal columns 
df1
df1.info()                                                            # checking for null values and data types
df1.duplicated().sum()                                                # checking for duplicate records

# creating frequencies of  movies
z = list(set(df1.columns))
y = []
for i in range (0,10):
    n = (df1.iloc[:,i]== 1).sum()
    y.append(n)

fitems = pd.DataFrame(columns = ["X","Y"])
fitems["X"] = z
fitems["Y"] = y
fitems.sort_values("Y",ascending = False , inplace = True)

# plotting frequencies of movies
plt.bar(x = fitems.X, height = fitems.Y, color ='rgmyk')
plt.xticks( fitems.X, rotation=90)
plt.xlabel('item-sets')
plt.ylabel('count')
plt.show()

# rules for most frequent movies
frequent_itemsets = apriori(df1, min_support = 0.005 , max_len = 0 , use_colnames = True)
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

# plotting most frequent movies
plt.bar(x = list(range(0, 10)), height = frequent_itemsets.support[0:10], color ='rgmyk')
plt.xticks(list(range(0, 10)), frequent_itemsets.itemsets[0:10], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

# using association rules dependency of movies
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

# removing redendency in itemsets
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
k = rules_no_redudancy.sort_values('lift', ascending = False).head(10)

