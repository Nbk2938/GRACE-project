# GRACE-Temperature Analysis: Key Findings and Interpretation

**Date:** March 28, 2026  
**Analysis Period:** 2002-2025 (249 months)

---

## Research Question
**"How strongly do Greenland ice-mass anomalies co-vary with regional warming?"**

---

## 1. Summary of Results

### PCA Variance Explained (First Mode)
| Dataset | PC1 Variance | Interpretation |
|---------|-------------|----------------|
| **GRACE-LWE Anomaly** | **88.2%** | Single dominant spatial pattern (secular mass loss trend) |
| Monthly Temperature | 59.6% | Strong but not overwhelming first mode |
| **GRACE-LWE Rate** | **41.8%** | More complex spatial-temporal variability |
| Monthly PDD | 39.7% | Most distributed variance across modes |

-> PCA correlations will be week on their own it's better to make a direct correlation between the spatial fields

**Key Insight:** GRACE cumulative anomaly is dominated by one spatial pattern (overall mass loss), while rate-of-change shows more month-to-month complexity.

---

### Correlation Results: Most Meaningful Findings

#### ⭐ **GRACE Rate vs PDD (STRONGEST SIGNAL)**
- **Lag 0 correlation:** r = -0.167 (r² = 4.1%)
- **Statistical significance:** 60.5% of pixels significant (p < 0.05)
- **Best lag:** 0 months (immediate response)
- **Spatial pattern:** Strongest correlations in northern regions (r up to -0.62 at lat 80-81°N)
- **Physical meaning:** More positive degree-days → faster mass loss rate

#### **GRACE Rate vs Temperature**
- **Lag 0 correlation:** r = 0.035 (very weak)
- **Best lag:** 1 month (r = -0.063, 61.6% significant pixels)
- **Physical meaning:** Temperature shows delayed effect, weaker than PDD

#### **GRACE Anomaly vs PDD/Temperature** (Less meaningful)
- **Very weak correlations:** r ≈ -0.02 to -0.04
- **Low significance:** 7-14% of pixels significant
- **Why weak:** The cumulative mass anomaly integrates long-term processes and is dominated by the secular trend (as shown by 88% PCA mode 1), making it less responsive to monthly temperature variations

---

## 2. Why De-seasonalization Was Essential

### The Problem Without De-seasonalization
Your data contains strong seasonal cycles:
- **Temperature/PDD:** Summer warmth vs winter cold (amplitude ~30-40°C)
- **GRACE:** Annual accumulation/ablation cycles

**Without de-seasonalization, you would find:**
- Artificially inflated correlations (r ≈ 0.9+): we don't want to test for co-seasonality
- These would be meaningless - they just tell us "summer differs from winter"
- Everyone already knows melting happens in summer!

### What De-seasonalization Does
**Removes the expected seasonal pattern** and isolates:
- **Interannual variability:** "Was summer 2012 warmer than usual?"
- **Anomalous behavior:** "Did unusually warm months cause unusually high melting?"

### Trade-offs of De-seasonalization

✅ **Benefits:**
- Reveals true physical relationships beyond seasonal cycles
- Allows comparison of same-month anomalies across years
- Answers: "Do warmer-than-normal summers cause more-than-normal melting?"

⚠️ **Consequences:**
- **Weaker correlations:** r = 0.17 instead of r = 0.9
- **But these are real, physically meaningful relationships!**
- **Removes/reduces trend information** (see Section 3)
- Lost information about seasonal timing (which we already know)

---

"How strongly do Greenland ice-mass anomalies co-vary with regional warming?"

Answer:
Rate of mass change shows moderate negative correlation with PDD (r ≈ -0.17, r² ≈ 4%)

60-71% of pixels show significant relationships
Strongest in northern regions (r up to -0.62)
Immediate response (lag 0 optimal)
Cumulative mass anomaly shows very weak correlation with temperature/PDD

Dominated by long-term trend (88% variance in PC1)
Monthly climate variations explain <1% of variance
Temperature vs PDD: PDD is a better predictor because it:

Captures only melting-relevant temperatures (>0°C)
Shows stronger correlations with mass loss
Has higher fraction of significant pixels

Physical Interpretation:
4% mean greeenland wide variance explained means (up to 40% near costal regions): "Warmer-than-normal months explain ~4% of month-to-month variations in melting rate"
The other 96% comes from: precipitation, ice dynamics, albedo feedbacks, measurement noise
This is reasonable for a complex system with multiple drivers



## 3. Critical Limitation: Why We Cannot Conclude Trend Relationships

### Initial Hypothesis (Logical, Likely True)
**"Long-term ice-mass loss is driven by long-term warming trend"**
- Temperatures increasing over decades (warmer winters and summers)
- Ice sheet losing mass over decades
- These should be related!

### Why Our Analysis Doesn't Test This

#### Problem 1: De-seasonalization Removes Trend Information
When you de-seasonalize, you're asking:
- ❌ **NOT:** "Do warmer decades have more mass loss?"
- ✅ **BUT:** "Do warmer-than-expected months have more-than-expected mass loss?"

**The secular trends (long-term warming + long-term mass loss) get largely absorbed into the climatology you subtract.**

#### Problem 2: Cumulative vs Rate Variable Mismatch
We compared:
- **GRACE LWE anomaly:** Cumulative mass (like a bank balance)
- **Temperature anomaly:** Monthly condition (like monthly income)

**Analogy:** Your monthly income variations don't correlate strongly with your total bank balance, because balance = sum of ALL past months. A single unusual month barely moves the total.

**BUT:** Monthly income DOES correlate with monthly change in balance!  
→ This is why **GRACE rate vs PDD works better** (r = -0.17, 60% significant)

#### Problem 3: What the PCA Actually Shows
- **88% variance in PC1** = the dominant secular trend
- This is shared by all pixels (everyone losing mass over time)
- Monthly temperature anomalies explain **variations around this trend**, not the trend itself

---

## 4. What Our Analysis Actually Tests vs What We Wanted

| What Our Analysis Tests | Answer | What We Wanted to Test | Tested? |
|------------------------|--------|------------------------|---------|
| "Do unusually warm Junes correlate with unusually low ice mass?" | Weakly (r ≈ -0.02) | "Does multi-decadal warming correlate with multi-decadal mass loss?" | ❌ No |
| "Do unusually warm months lead to faster mass loss rates?" | Yes, moderately (r ≈ -0.17, 60% significant) | "Does the spatial pattern of warming match mass loss?" | ❌ No |

---

## 5. Valid Conclusions from Our Analysis

### ✅ What We CAN Conclude:

1. **Month-to-month temperature anomalies explain ~4% of month-to-month mass loss RATE variations**
   - This is reasonable for a complex system with multiple drivers
   - The other 96% comes from: precipitation, ice dynamics, albedo feedbacks, measurement noise

2. **PDD is more physically relevant than raw temperature** for predicting melt
   - Stronger correlations (r = -0.17 vs r = 0.04)
   - Higher fraction of significant pixels (60% vs 21%)
   - Makes sense: only above-freezing temperatures cause melting

3. **Cumulative mass is dominated by long-term processes**, not monthly weather variations
   - 88% variance in single PC mode = coherent long-term trend
   - Monthly anomalies explain <1% of cumulative mass variance

4. **Northern Greenland (80-81°N) shows strongest month-to-month climate-melt coupling**
   - Top correlations: r = -0.62
   - These regions respond most sensitively to monthly temperature anomalies

5. **Response is immediate (lag 0 optimal)**
   - PDD in month t correlates with mass loss rate in month t
   - Suggests surface melt dominates over delayed processes

### ❌ What We CANNOT Conclude:

1. **Whether long-term warming drives long-term mass loss** (not tested with this method)
2. **The magnitude of climate's role in total ice loss** (only tested monthly anomalies)
3. **Trend relationships between temperature and ice mass** (removed by de-seasonalization)

---

## 6. How to Test the Original Trend Hypothesis

If you want to test "Does long-term warming cause long-term mass loss?", you would need:

### **Option 1: Trend Analysis**
```python
# Fit linear trends to both time series
grace_trend = fit_linear_trend(grace_data, time)  # mm/year at each pixel
temp_trend = fit_linear_trend(temp_data, time)    # °C/year at each pixel
# Correlate the spatial patterns of trends
correlate(grace_trend_map, temp_trend_map)
```

### **Option 2: Annual or Multi-Year Means**
```python
# Use annual means instead of monthly anomalies
grace_annual = grace_data.resample('1Y').mean()
temp_annual = temp_data.resample('1Y').mean()
# Correlation captures long-term changes, not monthly noise
```

### **Option 3: Low-Pass Filtering**
```python
# Keep only variations longer than 2-3 years
grace_lowpass = apply_lowpass_filter(grace_data, cutoff='3Y')
temp_lowpass = apply_lowpass_filter(temp_data, cutoff='3Y')
# Focuses on multi-year climate variations
```

---

## 7. The Scientific Story

### What This Analysis Reveals

> "While Greenland ice loss likely results from long-term warming trends, **month-to-month temperature anomalies explain only ~4% of monthly mass loss rate variations**. This suggests that ice sheet response integrates climate forcing over longer timescales and that other factors (precipitation, ice dynamics, albedo feedbacks) play important roles in monthly-to-seasonal variability. 
>
> The strong dominance of PC1 (88% variance) indicates that **the ice sheet responds coherently to large-scale, long-term forcing** rather than showing strong sensitivity to individual monthly weather anomalies. The immediate correlation (lag 0) between PDD and mass loss rate (r = -0.17, 60% of pixels significant) confirms that surface melting responds quickly to temperature, but this represents only a fraction of the total mass balance signal."

### Why This Is Interesting

**This is actually more scientifically interesting than finding r = 0.9!**

It reveals:
- **Timescale mismatch** between climate forcing and ice sheet response
- **Multi-factorial control** on ice mass (not just temperature)
- **Integration of forcing** over longer periods than individual months
- **Spatial heterogeneity** (northern regions more responsive)

---

## 8. Recommended Results to Report

### Primary Results (Most Meaningful):
1. **GRACE rate vs PDD lag-0 analysis** - Shows immediate melting response
2. **GRACE rate vs PDD lag analysis** - Confirms lag-0 is optimal
3. **PCA variance** - Shows dominance of long-term trend
4. **Top pixels analysis** - Shows northern Greenland sensitivity

### Secondary Results:
- Temperature correlations (weaker than PDD, as expected)
- Lag-1 temperature response (1-month delay)

### Less Critical (Can Omit):
- GRACE anomaly correlations (too weak, not physically meaningful)
- Lags beyond 2 months (no significant improvement)

---

## 9. Key Methodological Insights

### What De-seasonalization Does:
- **Removes:** Expected seasonal cycle, some trend information
- **Preserves:** Interannual variability, anomalous behavior
- **Result:** Lower correlations, but physically meaningful

### Why Correlations Are "Low":
- r² = 4% means temperature explains 4% of mass loss variance
- This is **appropriate** for a complex system with multiple drivers
- Would be concerning if r² = 90% (would mean we're missing drivers)

### Variable Choice Matters:
- **Rate variables** (GRACE rate, PDD) correlate better than cumulative
- **Physically-based variables** (PDD) perform better than raw variables (temperature)
- **Matching timescales** is crucial (monthly with monthly, trends with trends)

---

## References to Key Output Files

- **Best result:** `outputs/correlation_results/grace_rate_anomaly_vs_pdd_anomaly_lag0/summary.csv`
- **Lag analysis:** `outputs/correlation_results/grace_rate_anomaly_vs_pdd_anomaly_lag_analysis_maxlag7/`
- **PCA diagnostics:** `outputs/pca_results/grace-lwe_rate_of_change_de-seasonalized_pca_variance_explained.csv`
- **Top correlations:** `outputs/correlation_results/grace_rate_anomaly_vs_pdd_anomaly_lag0/top_pixels.csv`
