# Online Retail Orders Analysis

## **Project Overview**
This project analyzes the "orders" database of **Reliant Retail Limited**, an online retail store chain, to provide insights and support data-driven decision-making. The analysis is conducted using SQL queries to address specific business problems, leveraging techniques like filtering, grouping, and calculations.

The SQL queries and their corresponding outputs are designed to extract meaningful insights, which include customer segmentation, inventory management, and shipping analysis.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Dataset Description](#dataset-description)
4. [Key Objectives and Queries](#key-objectives-and-queries)
5. [Sample Outputs](#sample-outputs)
6. [How to Use](#how-to-use)
7. [Acknowledgments](#acknowledgments)

---

## **Introduction**
The "Online Retail Orders Analysis" project focuses on utilizing SQL queries to solve real-world problems related to retail operations. The primary goal is to help **Reliant Retail Limited** optimize operations, improve inventory management, and enhance customer service.

---

## **Technologies Used**
- SQL
- Relational Database Management System (RDBMS)
- Querying techniques: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `HAVING`, `COUNT`, `SUM`

---

## **Dataset Description**
The project is based on a relational database schema named `orders`, containing the following tables:
1. **Customers**
2. **Products**
3. **Orders**
4. **Shippers**
5. **Order Details**

The schema provides essential information about customer demographics, product inventory, sales data, and shipping logistics.

---

## **Key Objectives and Queries**

### **1. Customer Segmentation**
- Categorize customers based on their creation date into three groups:
  - **Category A**: Customers created before 2005.
  - **Category B**: Customers created between 2005 and 2010.
  - **Category C**: Customers created after 2011.

### **2. Unsold Products and Discounts**
- Identify unsold products and propose discounts based on their price:
  - 20% discount for products > ₹20,000.
  - 15% discount for products > ₹10,000.
  - 10% discount for products ≤ ₹10,000.

### **3. High Inventory Products**
- Display high inventory products (inventory value > ₹100,000) with their class details.

### **4. Order Cancellations**
- Identify customers who have canceled all their orders.

### **5. Shipping Analysis**
- Provide details on the number of customers and consignments handled by **DHL** in each city.

### **6. Payment-Based Analysis**
- Analyze orders paid in cash by customers whose last names start with 'G'.

### **7. Carton Fit Analysis**
- Find the largest order (by volume) that fits into a specific carton.

### **8. Inventory Status**
- Assess inventory health based on category-specific thresholds for inventory-to-sales ratios.

### **9. Product Combination Sales**
- Identify products sold along with a specific product but not shipped to certain cities.

### **10. Address-Based Shipping**
- Display order details for even-order IDs shipped to locations where the pin code does not start with "5".

---

## **Sample Outputs**
The outputs for each query are presented in tabular formats, providing actionable insights into:
- Customer demographics and behavior.
- Product inventory health and discounting strategies.
- Shipping efficiency and order-level details.

Refer to the accompanying documentation for detailed figures.

---

## **How to Use**
1. Load the provided `orders` database into a SQL-compatible RDBMS.
2. Execute the `.SQL` file containing the queries in sequence.
3. Review the generated outputs to analyze the results.

---

## **Acknowledgments**
This project was completed as part of the Postgraduate Program in Data Science and Business Analytics (PGP-DSBA). Special thanks to **Reliant Retail Limited** for providing the dataset.

---
