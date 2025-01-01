# Inferential Statistics Project

## **Project Overview**
This project focuses on utilizing inferential statistics to answer business and scientific questions using sample data. It applies statistical techniques such as probability calculations, hypothesis testing, and ANOVA to explore relationships, identify significant differences, and provide actionable insights. The project includes four problems with varying datasets and objectives.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Dataset Description](#dataset-description)
4. [Key Objectives](#key-objectives)
5. [Methodology](#methodology)
6. [Findings and Insights](#findings-and-insights)
7. [Recommendations](#recommendations)
8. [Acknowledgments](#acknowledgments)

---

## **Introduction**
As part of the PGP-DSBA program, this project demonstrates the application of inferential statistics to analyze sample data and infer population parameters. The key tools include probability density functions (PDFs), Z-scores, T-tests, and ANOVA, relying on Python's statistical packages for analysis.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - NumPy, Pandas (Data Manipulation)
  - SciPy (Statistical Analysis)
  - Matplotlib, Seaborn (Visualization)
  - Statsmodels (Advanced Statistical Tests)

---

## **Dataset Description**
The project involves the following datasets:

1. **Player Injury Data**:
   - A contingency table with positions (e.g., Striker, Winger) and injury status.

2. **Gunny Bag Data**:
   - Breaking strength of gunny bags (kg/cm²), normally distributed.

3. **Stone Hardness Data**:
   - Brinell Hardness Index (BHI) of polished and unpolished stones.

4. **Dental Implant Data**:
   - Hardness data influenced by dentist, method, alloy, and temperature.

---

## **Key Objectives**
1. Calculate probabilities for injury occurrences and position-related trends.
2. Analyze proportions of breaking strength values in the gunny bag dataset.
3. Test hypotheses about the suitability of materials for specific uses.
4. Determine the impact of categorical factors (e.g., dentist, method) on outcomes using ANOVA.

---

## **Methodology**
1. **Problem 1: Player Injury Data**:
   - Calculated probabilities using contingency tables.
   - Analyzed positional trends in injuries.

2. **Problem 2: Gunny Bag Data**:
   - Constructed a normal distribution from population parameters.
   - Computed proportions for breaking strength ranges.

3. **Problem 3: Stone Hardness Data**:
   - Used one-sample and two-sample T-tests to compare means.
   - Verified normality and variance equality before testing.

4. **Problem 4: Dental Implant Data**:
   - Performed ANOVA to examine the effects of categorical factors.
   - Used Tukey HSD for pairwise comparisons.

---

## **Findings and Insights**
### **Problem 1: Player Injury Data**
- 62% probability of injury for a randomly chosen player.
- Players in forward or winger positions are injured 52% of the time.

### **Problem 2: Gunny Bag Data**
- 11.12% of bags have breaking strength below 3.17 kg/cm².
- 82.47% of bags have breaking strength above 3.6 kg/cm².

### **Problem 3: Stone Hardness Data**
- Polished stones have significantly higher BHI than unpolished stones (p < 0.05).
- Unpolished stones are unsuitable for printing due to low hardness.

### **Problem 4: Dental Implant Data**
- Method 3 significantly impacts implant hardness.
- Interaction effects between dentist and method are significant for alloy types.

---

## **Recommendations**
1. **Player Injury Data**:
   - Focus injury prevention efforts on forwards and wingers.

2. **Gunny Bag Data**:
   - Target production adjustments to improve the breaking strength consistency.

3. **Stone Hardness Data**:
   - Prefer polished stones for printing due to superior hardness.

4. **Dental Implant Data**:
   - Standardize methods to reduce variability in implant hardness.

---

## **Acknowledgments**
This project was completed as part of the PGP-DSBA program. Special thanks to instructors and mentors for their guidance.

---
