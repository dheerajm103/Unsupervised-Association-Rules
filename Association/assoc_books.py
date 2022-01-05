import pandas as pd                                                  # importing libraries
from mlxtend.frequent_patterns import apriori , association_rules
from matplotlib import pyplot as plt
df = pd.read_csv("book.csv")                                         # importing data set
df

# Data cleansing

df.info()                                                            # checking null values and data types
df.duplicated().sum()                                                # checking for duplicate records

# creating frequency for each book 
z = list(set(df.columns))
y = []
for i in range (0,11):
    n = (df.iloc[:,i]== 1).sum()
    y.append(n)

fitems = pd.DataFrame(columns = ["X","Y"])
fitems["X"] = z
fitems["Y"] = y
fitems.sort_values("Y",ascending = False , inplace = True)

# plotting frequency of each books
plt.bar(x = fitems.X, height = fitems.Y, color ='rgmyk')
plt.xticks( fitems.X, rotation=90)
plt.xlabel('item-sets')
plt.ylabel('count')
plt.show()

# rules for  most books set 
frequent_itemsets = apriori(df, min_support = 0.005 , max_len = 0 , use_colnames = True)
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

# plotting most frequent book sets
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

# using association rules dependency
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

# removing redundancy in bookets
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

