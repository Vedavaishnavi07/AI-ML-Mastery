import csv


def load_orders(filename):
    orders = []

    with open(filename, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            orders.append(row)

    return orders


def calculate_total_revenue(orders):
    total = 0

    for order in orders:
        total += int(order["Quantity"]) * int(order["Price"])

    return total


def product_revenue(orders):
    products = {}

    for order in orders:
        product = order["Product"]
        revenue = int(order["Quantity"]) * int(order["Price"])

        if product in products:
            products[product] += revenue
        else:
            products[product] = revenue

    return products


def customer_revenue(orders):
    customers = {}

    for order in orders:
        customer = order["Customer"]
        revenue = int(order["Quantity"]) * int(order["Price"])

        if customer in customers:
            customers[customer] += revenue
        else:
            customers[customer] = revenue

    return customers


def category_revenue(orders):
    categories = {}

    for order in orders:
        category = order["Category"]
        revenue = int(order["Quantity"]) * int(order["Price"])

        if category in categories:
            categories[category] += revenue
        else:
            categories[category] = revenue

    return categories


def generate_report(orders):
    total_orders = len(orders)
    total_revenue = calculate_total_revenue(orders)

    products = product_revenue(orders)

    sorted_products = sorted(
        products.items(),
        key=lambda x: x[1],
        reverse=True
    )

    customers = customer_revenue(orders)

    sorted_customers = sorted(
        customers.items(),
        key=lambda x: x[1],
        reverse=True
    )

    categories = category_revenue(orders)

    with open("report.txt", "w") as file:

        file.write("CUSTOMER ORDER ANALYSIS REPORT\n")
        file.write("=" * 35 + "\n\n")

        file.write(f"Total Orders: {total_orders}\n")
        file.write(f"Total Revenue: {total_revenue}\n\n")

        file.write("Top Products:\n")

        for index, (product, revenue) in enumerate(sorted_products[:5], 1):
            file.write(f"{index}. {product} - {revenue}\n")

        file.write("\nTop Customers:\n")

        for index, (customer, revenue) in enumerate(sorted_customers[:5], 1):
            file.write(f"{index}. {customer} - {revenue}\n")

        file.write("\nCategory Revenue:\n")

        for category, revenue in categories.items():
            file.write(f"{category} - {revenue}\n")


orders = load_orders("orders.csv")

print("Total Orders:", len(orders))

print("Total Revenue:", calculate_total_revenue(orders))


print("\nTop Products by Revenue:")

products = product_revenue(orders)

sorted_products = sorted(
    products.items(),
    key=lambda x: x[1],
    reverse=True
)

for product, revenue in sorted_products[:5]:
    print(product, "-", revenue)


print("\nTop Customers by Revenue:")

customers = customer_revenue(orders)

sorted_customers = sorted(
    customers.items(),
    key=lambda x: x[1],
    reverse=True
)

for customer, revenue in sorted_customers[:5]:
    print(customer, "-", revenue)


print("\nCategory Revenue:")

categories = category_revenue(orders)

for category, revenue in categories.items():
    print(category, "-", revenue)


generate_report(orders)

print("\nReport generated successfully!")