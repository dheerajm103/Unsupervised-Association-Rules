import pandas as pd                                               # impoting libraries
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
retail_items = []

with open("C:/Users/dheer/transactions_retail1.csv") as f:       # importing text data                                # importing text data set
    retail_items = f.read()

# splitting the data into separate transactions using separator as "\n"
retail_items = retail_items.split("\n")

retail_list = []
for i in retail_items:
    retail_list.append(i.split(","))

all_retail_list = [i for item in retail_list for i in item]

from collections import Counter

item_frequencies = Counter(all_retail_list)
del(item_frequencies['NA'])
item_frequencies

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies[:2621]]))
items = list(reversed([i[0] for i in item_frequencies[:2621]]))


# plotting for item frequencies
plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11], rotation = 90)
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

retail_series = pd.DataFrame(pd.Series(retail_list))
retail_series = retail_series.iloc[:9835, :] # removing the last empty transaction
retail_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = retail_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

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


