# ECG Arrhythmia Detection Project

This project implements an end-to-end pipeline for ECG signal processing, R-peak detection, heart rate analysis, and arrhythmia classification using machine learning. The code is organized for clarity and reproducibility, and produces both visual and tabular outputs for analysis.

---

## Directory Structure

```
.
├── main.py                       # Main source code for the pipeline
├── Biomedical_ECGArrhythmiaDetection.pdf  # Structured project report
├── signal/                       # Input ECG signals (MIT-BIH format)
│   ├── 100.atr
│   ├── 100.dat
│   └── ...
├── output/                       # Output graphs and results
│   ├── confusion_matrix_decision_tree.png
│   ├── ecg_with_annotations_record_100.png
│   ├── heart_rate_record_100.png
│   └── ...
```

---

## Project Steps

### 1. Data Loading and Visualization

- **Input:** MIT-BIH ECG records in `signal/`.
- **Process:** Loads ECG signals and annotations using WFDB.
- **Visualization:** 
  - Plots the original ECG signal for each record.
  - Plots ECG signals with annotation markers.
- **Output:** Saved in `output/original_signal_record_*.png` and `output/ecg_with_annotations_record_*.png`.

### 2. Signal Preprocessing and Noise Removal

- **Baseline Wander Removal:** High-pass filtering (FFT-based) to remove low-frequency drift.
- **Powerline Interference Removal:** Notch filter at 50 Hz to suppress powerline noise.
- **Bandpass Filtering:** Butterworth filter (0.5–50 Hz) for general noise reduction.
- **Visualization:** 
  - Step-by-step plots of each preprocessing stage.
  - Frequency domain (power spectrum) plots before and after filtering.
- **Output:** Saved in `output/record_*_preprocessing_steps.png`, `output/record_*_comparison.png`, and `output/record_*_spectra.png`.

### 3. R-Peak Detection and Heart Rate Analysis

- **R-Peak Detection:** Uses `scipy.signal.find_peaks` on the filtered ECG signal.
- **RR Interval Calculation:** Computes intervals between detected R-peaks.
- **Heart Rate Analysis:** 
  - Calculates average heart rate.
  - Detects bradycardia, tachycardia, or normal rhythm.
- **Visualization:** 
  - ECG with detected R-peaks.
  - Histogram of RR intervals.
  - Heart rate trend over time.
- **Output:** Saved in `output/r_peaks_record_*.png`, `output/rr_intervals_record_*.png`, and `output/heart_rate_record_*.png`.

### 4. Arrhythmia Detection and Classification

- **Feature Extraction:** 
  - Segments ECG into windows.
  - Extracts features: mean RR, std RR, SDNN, RMSSD, pNN50.
  - Labels each segment as normal or abnormal based on annotations.
- **Data Balancing:** Uses SMOTE to address class imbalance.
- **Feature Selection:** Selects important features using Random Forest.
- **Model Training:** 
  - Trains Decision Tree, Random Forest, Logistic Regression, and Voting Classifier.
  - Evaluates using accuracy, precision, recall, F1, confusion matrix, and ROC curve.
- **Output:** 
  - Confusion matrices and ROC curves for each classifier in `output/`.

### 5. Report

- **Comprehensive Report:** 
  - The file `Biomedical_ECGArrhythmiaDetection.pdf` contains a detailed description of the methodology, results, and analysis.

---

## How to Run

1. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   *(Make sure to include all required packages in `requirements.txt`)*

2. **Prepare Data:**
   - Place MIT-BIH ECG records (`.dat`, `.hea`, `.atr`) in the `signal/` directory.

3. **Run the Pipeline:**
   ```sh
   python main.py
   ```

4. **View Results:**
   - All output figures and results will be saved in the `output/` directory.
   - The report is available as `Biomedical_ECGArrhythmiaDetection.pdf`.

---

## Notes

- The code is modular and each function is documented for clarity.
- You can adjust preprocessing parameters and model settings in `main.py`.
- For more details, refer to the report.

---

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- WFDB Python Library: https://github.com/MIT-LCP/wfdb-python

---