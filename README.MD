# Institutional Credit Default Risk & Underwriting Simulator

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Tableau Public](https://img.shields.io/badge/Visualization-Tableau%20Public-E97627.svg)](https://public.tableau.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end predictive modeling pipeline and interactive Business Intelligence platform designed to evaluate institutional credit risk, project expected portfolio loss ($EL = PD \times EAD \times LGD$), and simulate real-time underwriting approval cutoffs for credit portfolio optimization.

---

## 🔗 Live Interactive Dashboard

Explore the live underwriting simulator and adjust risk thresholds dynamically:  
👉 **[View Interactive Tableau Public Dashboard](https://public.tableau.com/)** *(Paste your live dashboard link here)*

---

## 📌 Executive Summary

Underwriting decisions require balancing approval volume against credit default exposure. Traditional static cutoffs fail to quantify immediate capital trade-offs across shifting macro environments. 

This framework solves this problem by coupling a **Random Forest credit default scoring model** with an **interactive Tableau underwriting engine**:
* **Total Portfolio Scope:** $375M across 15,000 corporate/institutional loan facilities.
* **Model Discrimination:** **0.86 ROC-AUC**, effectively isolating non-linear risk drivers across class-imbalanced defaults.
* **Baseline Downside Exposure:** Identifies $67M in total Expected Loss ($EL$) across the analyzed portfolio.
* **Underwriting Cutoff Simulation:** Enables credit officers to stress-test approval thresholds from 5% to 50% default probability, quantifying real-time capital deployment vs. bad debt exposure.

---

## 📊 Dashboard Architecture & Key Visualizations

The published executive dashboard integrates three core modules:

1. **Executive Scorecard (KPI Banner):**  
   Tracks top-level portfolio health, including total facilities analyzed (15,000), total capital deployed ($375M), projected Expected Loss ($67M), and baseline portfolio default rate (34.3%).
2. **Dynamic Underwriting Simulator:**  
   Driven by a custom Tableau parameter (`Risk Cutoff Threshold`). Moving the slider updates the `Underwriting Status` calculation in real time, recalculating total approved vs. declined loan volumes ($) instantly.
3. **Capital Exposure by Risk Tier:**  
   Categorizes debt commitments into institutional credit tiers (`Low`, `Moderate`, `High`, `Severe`), providing visibility into portfolio concentration and collateral buffers.

---

## ⚙️ Technical Architecture & Pipeline

```text
├── Data Generation & Preprocessing (15,000 records, high-dimensional credit attributes)
├── Predictive Modeling (Stratified split, Random Forest Classifier, 0.86 ROC-AUC)
├── Expected Loss Valuation Engine (EL = PD * EAD * LGD)
├── Production Feed Export (credit_risk_tableau_feed.csv)
└── Executive BI Deployment (Tableau Public Dashboard with interactive parameter logic)