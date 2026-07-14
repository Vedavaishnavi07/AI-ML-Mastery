import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("WORLD HAPPINESS REPORT ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/world_happiness_report_2005_2025.csv")
print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:\n")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")

print(df.describe())

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

df = df[(df["year"] >= 2015) & (df["year"] <= 2023)]

print("\nDataset Shape After Filtering (2015-2023):")
print(df.shape)

duplicates = df.duplicated().sum()

print("\nDuplicate Records:")
print(duplicates)

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

print("\nMissing Values After Filtering:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlation = df[
[
"happiness_score",
"explained_log_gdp_per_capita",
"explained_social_support",
"explained_healthy_life_expectancy",
"explained_freedom",
"explained_generosity",
"explained_corruption"
]
].corr()

print(correlation["happiness_score"].sort_values(ascending=False))

import numpy as np

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

regression_df = df[
["explained_log_gdp_per_capita","happiness_score"]
].dropna()

x = regression_df["explained_log_gdp_per_capita"]
y = regression_df["happiness_score"]

slope, intercept = np.polyfit(x, y, 1)

print("\nRegression Equation:")

print(f"Happiness Score = {slope:.2f} * GDP + {intercept:.2f}")

print("\n" + "=" * 60)
print("TOP 10 HAPPIEST COUNTRIES (2023)")
print("=" * 60)

latest = df[df["year"] == 2023]

top10 = latest.sort_values(
by="happiness_score",
ascending=False
)[["country","happiness_score"]]

print(top10.head(10))

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Top 10 Happiest Countries (2023)
plt.figure(figsize=(10,5))
top10 = latest.sort_values(by="happiness_score", ascending=False).head(10)

plt.bar(top10["country"], top10["happiness_score"])

plt.title("Top 10 Happiest Countries (2023)")
plt.xlabel("Country")
plt.ylabel("Happiness Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("images/top10_happiest_countries.png")
plt.close()

# 2. Average Happiness by Year
plt.figure(figsize=(10,5))

yearly = df.groupby("year")["happiness_score"].mean()

plt.plot(yearly.index, yearly.values, marker="o")

plt.title("Average Happiness Score (2015-2023)")
plt.xlabel("Year")
plt.ylabel("Average Happiness Score")

plt.tight_layout()
plt.savefig("images/happiness_trend.png")
plt.close()

# 3. GDP vs Happiness
plt.figure(figsize=(8,5))

regression_df = df[
["explained_log_gdp_per_capita","happiness_score"]
].dropna()

plt.scatter(
regression_df["explained_log_gdp_per_capita"],
regression_df["happiness_score"]
)

plt.title("GDP vs Happiness")
plt.xlabel("GDP")
plt.ylabel("Happiness Score")

plt.tight_layout()
plt.savefig("images/gdp_vs_happiness.png")
plt.close()

# 4. Social Support vs Happiness
plt.figure(figsize=(8,5))

social = df[
["explained_social_support","happiness_score"]
].dropna()

plt.scatter(
social["explained_social_support"],
social["happiness_score"]
)

plt.title("Social Support vs Happiness")
plt.xlabel("Social Support")
plt.ylabel("Happiness Score")

plt.tight_layout()
plt.savefig("images/social_support.png")
plt.close()

# 5. Freedom vs Happiness
plt.figure(figsize=(8,5))

freedom = df[
["explained_freedom","happiness_score"]
].dropna()

plt.scatter(
freedom["explained_freedom"],
freedom["happiness_score"]
)

plt.title("Freedom vs Happiness")
plt.xlabel("Freedom")
plt.ylabel("Happiness Score")

plt.tight_layout()
plt.savefig("images/freedom.png")
plt.close()

# 6. Corruption vs Happiness
plt.figure(figsize=(8,5))

corruption = df[
["explained_corruption","happiness_score"]
].dropna()

plt.scatter(
corruption["explained_corruption"],
corruption["happiness_score"]
)

plt.title("Corruption vs Happiness")
plt.xlabel("Corruption")
plt.ylabel("Happiness Score")

plt.tight_layout()
plt.savefig("images/corruption.png")
plt.close()

print("\n6 Visualizations Saved Successfully!")