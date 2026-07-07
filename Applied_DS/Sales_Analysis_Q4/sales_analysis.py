# ============================================
# Sales Analysis - Regional Q4
# Author: Veda Vaishnavi
# ============================================

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 50)
print("SALES ANALYSIS - REGIONAL Q4")
print("=" * 50)

# Load the dataset
df = pd.read_csv("data/SampleSuperstore.csv")

# -------------------------------
# Display first 5 rows
# -------------------------------
print("\nFirst 5 Rows of the Dataset:\n")
print(df.head())

# -------------------------------
# Dataset Shape
# -------------------------------
print("\nDataset Shape:")
print(df.shape)

# -------------------------------
# Column Names
# -------------------------------
print("\nColumn Names:")
print(df.columns)

# -------------------------------
# Dataset Information
# -------------------------------
print("\nDataset Information:")
df.info()

# -------------------------------
# Statistical Summary
# -------------------------------
print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# Basic Business Statistics
# -------------------------------

print("\n" + "=" * 50)
print("BUSINESS OVERVIEW")
print("=" * 50)

print(f"\nTotal Sales: ${df['Sales'].sum():,.2f}")

print(f"Total Profit: ${df['Profit'].sum():,.2f}")

print(f"Average Sales: ${df['Sales'].mean():.2f}")

print(f"Average Profit: ${df['Profit'].mean():.2f}")

print(f"\nTotal Orders: {len(df)}")

print(f"Number of Regions: {df['Region'].nunique()}")

print(f"Number of States: {df['State'].nunique()}")

print(f"Number of Categories: {df['Category'].nunique()}")

print("\nRegions:")

print(df['Region'].unique())

print("\nCategories:")

print(df['Category'].unique())

# ============================================
# GROUPBY ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("GROUPBY ANALYSIS")
print("=" * 60)

sales_by_region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
profit_by_region = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)

sales_by_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
profit_by_category = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)

top_states = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10)

print("\nSales by Region\n")
print(sales_by_region)

print("\nProfit by Region\n")
print(profit_by_region)

print("\nSales by Category\n")
print(sales_by_category)

print("\nProfit by Category\n")
print(profit_by_category)

print("\nTop 10 States by Sales\n")
print(top_states)

plt.figure(figsize=(8,5))
sales_by_region.plot(kind="bar")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("images/sales_by_region.png")
plt.show()

plt.figure(figsize=(8,5))
profit_by_region.plot(kind="bar")
plt.title("Total Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("images/profit_by_region.png")
plt.show()

plt.figure(figsize=(8,5))
sales_by_category.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Sales by Category")
plt.tight_layout()
plt.savefig("images/sales_by_category.png")
plt.show()

plt.figure(figsize=(8,5))
profit_by_category.plot(kind="bar")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("images/profit_by_category.png")
plt.show()

plt.figure(figsize=(10,6))
top_states.plot(kind="barh")
plt.title("Top 10 States by Sales")
plt.xlabel("Sales")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("images/top10_states_sales.png")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["Sales"], df["Profit"])
plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("images/sales_vs_profit.png")
plt.show()

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

print("1. West region generated the highest sales.")
print("2. West region also generated the highest profit.")
print("3. Technology is the highest revenue generating category.")
print("4. Furniture has high sales but comparatively low profit.")
print("5. California contributes the highest sales among all states.")

print("\nProject Completed Successfully!")