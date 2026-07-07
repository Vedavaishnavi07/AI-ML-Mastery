import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind

print("=" * 60)
print("MARKETING CAMPAIGN ANALYSIS")
print("=" * 60)

df = pd.read_csv("data/bank.csv")

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:\n")
print(df.columns)

print("\nDataset Information:\n")
df.info()

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())

# DATA CLEANING
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nDuplicate Records:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

print("\nDeposit Value Counts:")
print(df["deposit"].value_counts())

print("\nDeposit Percentage:")
print(round(df["deposit"].value_counts(normalize=True) * 100, 2))

#Exploratory Data Analysis(EDA)
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nJob Distribution:")
print(df["job"].value_counts())

print("\nMarital Status Distribution:")
print(df["marital"].value_counts())

print("\nEducation Distribution:")
print(df["education"].value_counts())

print("\nContact Method Distribution:")
print(df["contact"].value_counts())

print("\nAverage Balance:")
print(df["balance"].mean())

print("\nAverage Campaign Contacts:")
print(df["campaign"].mean())

# CHI-SQUARE TEST

print("\n" + "=" * 60)
print("CHI-SQUARE TEST")
print("=" * 60)

contingency_table = pd.crosstab(df["contact"], df["deposit"])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Statistic:")
print(round(chi2,2))

print("\nP-value:")
print(round(p,5))

if p < 0.05:
    print("\nResult: Contact method significantly affects customer subscription.")
else:
    print("\nResult: Contact method does not significantly affect customer subscription.")


# T-TEST

print("\n" + "=" * 60)
print("INDEPENDENT T-TEST")
print("=" * 60)

deposit_yes = df[df["deposit"]=="yes"]["balance"]

deposit_no = df[df["deposit"]=="no"]["balance"]

t_stat, p_value = ttest_ind(deposit_yes, deposit_no)

print("\nT-Statistic:")
print(round(t_stat,2))

print("\nP-value:")
print(round(p_value,5))

if p_value < 0.05:
    print("\nResult: Average balance differs significantly between customers who subscribed and those who did not.")
else:
    print("\nResult: No significant difference in average balance.")


# VISUALIZATIONS

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Deposit Distribution
plt.figure(figsize=(6,4))
df["deposit"].value_counts().plot(kind="bar")
plt.title("Deposit Subscription Distribution")
plt.xlabel("Deposit")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("images/deposit_distribution.png")
plt.close()

# 2. Job vs Deposit
job_deposit = pd.crosstab(df["job"], df["deposit"])
job_deposit.plot(kind="bar", figsize=(10,5))
plt.title("Job vs Deposit Subscription")
plt.xlabel("Job")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("images/job_vs_deposit.png")
plt.close()

# 3. Education vs Deposit
education_deposit = pd.crosstab(df["education"], df["deposit"])
education_deposit.plot(kind="bar", figsize=(8,5))
plt.title("Education vs Deposit Subscription")
plt.xlabel("Education")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("images/education_vs_deposit.png")
plt.close()

# 4. Contact Method
plt.figure(figsize=(6,4))
df["contact"].value_counts().plot(kind="bar")
plt.title("Contact Method Distribution")
plt.xlabel("Contact Method")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("images/contact_method.png")
plt.close()

# 5. Balance Distribution
plt.figure(figsize=(8,5))
plt.hist(df["balance"], bins=30)
plt.title("Account Balance Distribution")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/balance_distribution.png")
plt.close()

# 6. Age Distribution by Deposit
plt.figure(figsize=(8,5))

plt.hist(df[df["deposit"]=="yes"]["age"],
         bins=20,
         alpha=0.7,
         label="Yes")

plt.hist(df[df["deposit"]=="no"]["age"],
         bins=20,
         alpha=0.7,
         label="No")

plt.title("Age Distribution by Deposit")
plt.xlabel("Age")
plt.ylabel("Customers")
plt.legend()

plt.tight_layout()
plt.savefig("images/age_distribution.png")
plt.close()

print("\n6 Visualizations Saved Successfully!")  


