# Project Title: House Price Prediction

## 📄 Project Description
This project is submitted as a part of the PGDBA Course done through **Great Learnings** in Affiliation with **McCombs School of Business Texas**.

### Problem statement:  
A house value is simply more than location and square footage. Like the features that make up a 
person, an educated party would want to know all aspects that give a house its value. For 
example, you want to sell a house and you don’t know the price which you may expect — it can’t 
be too low or too high. To find house price you usually try to find similar properties in your 
neighborhood and based on gathered data you will try to assess your house price.

### Objective:
Take advantage of all of the feature variables available below, use it to analyse and predict house 
prices.  
1. cid: a notation for a house 
2. dayhours: Date house was sold 
3. price: Price is prediction target 
4. room_bed: Number of Bedrooms/House 
5. room_bath: Number of bathrooms/bedrooms 
6. living_measure: square footage of the home 
7. lot_measure: quare footage of the lot 
8. ceil: Total floors (levels) in house 
9. coast: House which has a view to a waterfront 
10. sight: Has been viewed 
11. condition: How good the condition is (Overall) 
12. quality: grade given to the housing unit, based on grading system 
13. ceil_measure: square footage of house apart from basement 
14. basement_measure: square footage of the basement 
15. yr_built: Built Year 
16. yr_renovated: Year when house was renovated 
17. zipcode: zip 
18. lat: Latitude coordinate 
19. long: Longitude coordinate 
20. living_measure15: Living room area in 2015(implies-- some renovations) This might or 
might not have affected the lotsize area 
21. lot_measure15: lotSize area in 2015(implies-- some renovations) 
22. furnished: Based on the quality of room  
23. total_area: Measure of both living and lot

---

## 🛠️ Tools and Techniques Used
- **Programming Language:** [ Python]
- **Libraries/Frameworks:** [e.g., Pandas, NumPy, Scikit-Learn, Matplotlib, LightGBM]
- **Techniques:**
  - Data Cleaning and Preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature Engineering and Selection
  - Machine Learning Models [e.g., Linear Regression, LightGBM, etc.]
  - Evaluation Metrics [e.g., RMSE, R²]

---

## ✨ Key Results and Insights
- Achieved an R² score of 0.85 on the test dataset.
- Identified key factors affecting house prices, such as renovation age and waterfront view.
- Improved prediction accuracy using ensemble techniques like LightGBM.

## 🚀 Instructions for Running the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/project-repo-name.git
   cd project-repo-name
2. **Install the requirements.txt**
   ```bash
   pip install -r .\requirements.txt
   pip install openpyxl
3. The input data file is in ".\data\input\innercity.xlsx"
4. Run the scrip file "example.py" in python environment, or better still run the jupyter notebook "example.ipynb", this file has detailed explanation of the approach taken for the project and the callouts.
