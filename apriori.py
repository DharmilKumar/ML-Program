# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 04:10:01 2023

@author: 91814
"""
from apyori import apriori
import pandas as pd

# Create a sample dataset for market basket analysis
data = [['A', 'B'],
        ['A', 'C'],
        ['B', 'C', 'D'],
        ['A', 'C', 'D'],
        ['A', 'C']]

# Apply the Apriori algorithm
association_rules = list(apriori(data, min_support=0.2, min_confidence=0.6, min_lift=1.0))

# Print the association rules
for rule in association_rules:
    items = [item for item in rule.items]
    association = [item for item in rule.ordered_statistics[0].items_base]
    support = rule.support
    confidence = rule.ordered_statistics[0].confidence
    lift = rule.ordered_statistics[0].lift
    print("Rule:", items, "->", association, "Support:", support, "Confidence:", confidence, "Lift:", lift)
