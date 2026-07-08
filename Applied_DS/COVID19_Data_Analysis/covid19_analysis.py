import pandas as pd
import matplotlib.pyplot as plt


# LOAD DATASET
print("=" * 60)
print("COVID-19 DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/owid-covid-data.csv")

print("\nDataset Loaded Successfully!")

# DATA EXPLORATION
print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:\n")
print(df.columns)

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())

# DATA CLEANING
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Keep only useful columns

columns = [
    "location",
    "continent",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths",
    "new_deaths",
    "population",
    "gdp_per_capita",
    "life_expectancy",
    "human_development_index"
]

df = df[columns]

print("\nSelected Columns:\n")
print(df.columns)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Remove duplicate rows
duplicates = df.duplicated().sum()

print("\nDuplicate Records:")
print(duplicates)

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Remove aggregate rows like World, Asia etc.
df = df[df["continent"].notna()]

print("\nDataset Shape After Filtering:")
print(df.shape)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

print("\nColumn Names:\n")
print(df.columns)

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())

# COVID ANALYSIS
print("\n" + "=" * 60)
print("COVID DATA ANALYSIS")
print("=" * 60)

# Latest record for every country
latest = (
    df
    .sort_values("date")
    .groupby("location", as_index=False)
    .last()
)

# TOP 10 COUNTRIES BY TOTAL CASES
top_cases = (
    latest[
        ["location", "total_cases"]
    ]
    .sort_values(
        by="total_cases",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Countries by Total Cases:\n")
print(top_cases)


# TOP 10 COUNTRIES BY TOTAL DEATHS
top_deaths = (
    latest[
        ["location", "total_deaths"]
    ]
    .sort_values(
        by="total_deaths",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Countries by Total Deaths:\n")
print(top_deaths)

# CASES BY CONTINENT
continent_cases = (
    latest
    .groupby("continent")["total_cases"]
    .sum()
    .sort_values(ascending=False)
)

print("\nCases by Continent:\n")
print(continent_cases)

# GLOBAL SUMMARY
global_cases = latest["total_cases"].sum()

global_deaths = latest["total_deaths"].sum()

print("\nGlobal Summary:\n")

print(f"Total Cases : {global_cases:,.0f}")

print(f"Total Deaths: {global_deaths:,.0f}")


# GDP ANALYSIS
top_gdp = (
    latest[
        ["location", "gdp_per_capita"]
    ]
    .dropna()
    .sort_values(
        by="gdp_per_capita",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 GDP Per Capita Countries:\n")
print(top_gdp)

# LIFE EXPECTANCY
top_life = (
    latest[
        ["location", "life_expectancy"]
    ]
    .dropna()
    .sort_values(
        by="life_expectancy",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Life Expectancy Countries:\n")
print(top_life)

# HUMAN DEVELOPMENT INDEX
top_hdi = (
    latest[
        ["location", "human_development_index"]
    ]
    .dropna()
    .sort_values(
        by="human_development_index",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Human Development Index Countries:\n")
print(top_hdi)


# CREATING VISUALIZATIONS
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Top 10 Countries by Total Cases
plt.figure(figsize=(10,6))

plt.bar(
    top_cases["location"],
    top_cases["total_cases"]
)

plt.title("Top 10 Countries by Total COVID-19 Cases")
plt.xlabel("Country")
plt.ylabel("Total Cases")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("images/top10_cases.png")
plt.close()

# 2. Top 10 Countries by Total Deaths
plt.figure(figsize=(10,6))

plt.bar(
    top_deaths["location"],
    top_deaths["total_deaths"]
)

plt.title("Top 10 Countries by Total COVID-19 Deaths")
plt.xlabel("Country")
plt.ylabel("Total Deaths")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("images/top10_deaths.png")
plt.close()


# 3. Cases by Continent
plt.figure(figsize=(8,6))

continent_cases.plot(
    kind="bar"
)

plt.title("COVID-19 Cases by Continent")
plt.xlabel("Continent")
plt.ylabel("Total Cases")

plt.tight_layout()
plt.savefig("images/continent_cases.png")
plt.close()

# 4. GDP vs Total Cases
plt.figure(figsize=(8,6))

plt.scatter(
    latest["gdp_per_capita"],
    latest["total_cases"],
    alpha=0.6
)

plt.title("GDP Per Capita vs Total COVID Cases")
plt.xlabel("GDP Per Capita")
plt.ylabel("Total Cases")

plt.tight_layout()
plt.savefig("images/gdp_vs_cases.png")
plt.close()

# 5. Life Expectancy vs Total Deaths
plt.figure(figsize=(8,6))

plt.scatter(
    latest["life_expectancy"],
    latest["total_deaths"],
    alpha=0.6
)

plt.title("Life Expectancy vs Total Deaths")
plt.xlabel("Life Expectancy")
plt.ylabel("Total Deaths")

plt.tight_layout()
plt.savefig("images/life_expectancy_vs_deaths.png")
plt.close()


# 6. Human Development Index Distribution
plt.figure(figsize=(8,6))

plt.hist(
    latest["human_development_index"].dropna(),
    bins=25
)

plt.title("Human Development Index Distribution")
plt.xlabel("Human Development Index")
plt.ylabel("Number of Countries")

plt.tight_layout()
plt.savefig("images/hdi_distribution.png")
plt.close()

print("6 Visualizations Saved Successfully!")


# FINAL SUMMARY
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"Countries Analysed : {latest.shape[0]}")
print(f"Global Cases       : {global_cases:,.0f}")
print(f"Global Deaths      : {global_deaths:,.0f}")

print("\nTop Country by Cases :")
print(top_cases.iloc[0])

print("\nTop Country by Deaths :")
print(top_deaths.iloc[0])

print("\nTop GDP Country :")
print(top_gdp.iloc[0])

print("\nHighest Life Expectancy :")
print(top_life.iloc[0])

print("\nHighest HDI :")
print(top_hdi.iloc[0])


print("COVID-19 Analysis Completed Successfully!")


