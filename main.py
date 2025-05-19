import os
import numpy as np
import pandas as pd
import wfdb 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as sig 

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from collections import Counter
from pathlib import Path
from scipy.fft import fft, ifft, fftfreq


# ===========================================================
# Phase 1: Data Loading and Preprocessing					=
# ===========================================================


def get_all_records(db_path):
    record_files = [f.stem for f in Path(db_path).glob("*.hea")] 
    return sorted(record_files) 


def ensure_dir_exists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_record(record_num, db_path):
    record_path = f"{db_path}/{record_num}"
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    return record, annotation


def extract_signal_segment(record, start_time, duration):
    fs = record.fs
    start_sample = int(start_time * fs)
    end_sample = start_sample + int(duration * fs)
    
    channel_idx = 0
    if 'MLII' in record.sig_name:
        channel_idx = record.sig_name.index('MLII')
    
    signal = record.p_signal[start_sample:end_sample, channel_idx]
    time = np.arange(start_sample, end_sample) / fs
    channel_name = record.sig_name[channel_idx]
    
    return signal, time, fs, channel_name


def plot_original_signal(record, record_num, start_time, duration, output_dir):
    ensure_dir_exists(output_dir)
    signal = record.p_signal[:, 0]  # Assuming single-lead ECG
    fs = record.fs  # Sampling frequency
    time = np.arange(len(signal)) / fs
    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    segment_time = time[start_idx:end_idx]
    segment_signal = signal[start_idx:end_idx]

    # Plot the signal
    plt.figure(figsize=(15, 6))
    plt.plot(segment_time, segment_signal, label="ECG Signal", color="blue")
    plt.title(f"Original ECG Signal - Record {record_num}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the figure
    output_path = f"{output_dir}/original_signal_record_{record_num}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved: {output_path}")


def plot_ecg_with_annotations(record, annotations, record_num, start_time, duration, output_dir):
    ensure_dir_exists(output_dir)

    signal = record.p_signal[:, 0] 
    fs = record.fs

    time = np.arange(len(signal)) / fs

    start_idx = int(start_time * fs)
    end_idx = int((start_time + duration) * fs)
    segment_time = time[start_idx:end_idx]
    segment_signal = signal[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(segment_time, segment_signal, label="ECG Signal", color="blue")

    for i, sample in enumerate(annotations.sample):
        if start_idx <= sample < end_idx:
            annotation_time = sample / fs
            annotation_symbol = annotations.symbol[i]
            ax.axvline(annotation_time, color="red", linestyle="--", label="Annotation" if i == 0 else "")
            ax.text(annotation_time, segment_signal[sample - start_idx], annotation_symbol, color="red", fontsize=8, ha="center", va="bottom")

    ax.set_title(f"ECG Signal with Annotations - Record {record_num}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()

    output_path = f"{output_dir}/ecg_with_annotations_record_{record_num}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved: {output_path}")


# ==========================================================
# Phase 2: Signal Preprocessing and Noise Removal          =
# ==========================================================


def remove_baseline_wander_fft(signal, fs):
    # Compute the FFT of the signal
    n = len(signal)
    fft_signal = fft(signal)
    
    # Compute frequency bins
    freq = fftfreq(n, 1/fs)
    
    # Create a high-pass filter mask (remove frequencies below 0.5 Hz)
    mask = np.abs(freq) > 0.5
    
    # Apply the mask to the FFT
    fft_signal_filtered = fft_signal * mask
    
    # Compute the inverse FFT
    filtered_signal = np.real(ifft(fft_signal_filtered))
    
    return filtered_signal


def remove_powerline_interference(signal, fs, powerline_freq=50.0, quality_factor=30.0):
    # Design a notch filter
    b, a = iirnotch(powerline_freq, quality_factor, fs)
    
    # Apply the filter with forward-backward filtering to avoid phase distortion
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def apply_bandpass_filter(signal, fs, lowcut=0.5, highcut=50.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design a Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter with forward-backward filtering to avoid phase distortion
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def preprocess_ecg_signal(signal, fs, config=None):
    # Default configuration if not provided
    if config is None:
        config = {
            'bandpass_lowcut': 0.5,     # Hz - high-pass cutoff for baseline wander
            'bandpass_highcut': 50.0,   # Hz - low-pass cutoff for noise reduction
            'notch_freq': 50.0,         # Hz - power line interference frequency
            'notch_quality': 30.0       # Quality factor for notch filter
        }
    
    # Step 1: Remove baseline wander using FFT-based high-pass filtering
    baseline_removed = remove_baseline_wander_fft(signal, fs)
    
    # Step 2: Remove powerline interference using a notch filter
    powerline_removed = remove_powerline_interference(
        baseline_removed, 
        fs, 
        config['notch_freq'], 
        config['notch_quality']
    )
    
    # Step 3: Apply bandpass filtering for general noise reduction
    final_filtered = apply_bandpass_filter(
        powerline_removed, 
        fs, 
        config['bandpass_lowcut'], 
        config['bandpass_highcut']
    )
    
    # Store intermediate signals for analysis
    intermediate_signals = {
        'original': signal,
        'baseline_removed': baseline_removed,
        'powerline_removed': powerline_removed,
        'final_filtered': final_filtered
    }
    
    return final_filtered, intermediate_signals


def save_figure(fig, filename, output_dir, dpi=100):
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")
    

def visualize_preprocessing_steps(signals, time, fs, record_num, channel_name, output_dir='./output'):
    # Create a figure with subplots for each preprocessing step
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # 1. Plot original signal
    axs[0].plot(time, signals['original'], label='Original Signal')
    axs[0].set_title(f'Original ECG Signal (Record {record_num}, Channel {channel_name})')
    axs[0].set_ylabel('Amplitude [mV]')
    axs[0].grid(True)
    axs[0].legend()
    
    # 2. Plot signal after baseline wander removal
    axs[1].plot(time, signals['baseline_removed'], label='Baseline Removed', color='green')
    axs[1].set_title('After Baseline Wander Removal (High-Pass Filter)')
    axs[1].set_ylabel('Amplitude [mV]')
    axs[1].grid(True)
    axs[1].legend()
    
    # 3. Plot signal after powerline interference removal
    axs[2].plot(time, signals['powerline_removed'], label='Powerline Interference Removed', color='orange')
    axs[2].set_title('After Powerline Interference Removal (Notch Filter)')
    axs[2].set_ylabel('Amplitude [mV]')
    axs[2].grid(True)
    axs[2].legend()
    
    # 4. Plot final filtered signal
    axs[3].plot(time, signals['final_filtered'], label='Final Filtered Signal', color='red')
    axs[3].set_title('After Bandpass Filtering for Noise Reduction')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Amplitude [mV]')
    axs[3].grid(True)
    axs[3].legend()
    
    plt.tight_layout()
    save_figure(fig, f'record_{record_num}_preprocessing_steps.png', output_dir)
    plt.show()
    
    # Create direct comparison plot (original vs final)
    fig2, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time, signals['original'], label='Original Signal', alpha=0.7)
    ax.plot(time, signals['final_filtered'], label='Filtered Signal', color='red', alpha=0.7)
    ax.set_title(f'Original vs. Filtered ECG Signal (Record {record_num}, Channel {channel_name})')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [mV]')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    save_figure(fig2, f'record_{record_num}_comparison.png', output_dir)
    plt.show()
    
    # Create frequency domain visualization
    fig3, axs = plt.subplots(2, 1, figsize=(15, 10))
    
    # Calculate and plot power spectrum of original signal
    f_orig, pxx_orig = sig.welch(signals['original'], fs, nperseg=min(1024, len(signals['original'])))
    axs[0].semilogy(f_orig, pxx_orig, label='Original Signal')
    axs[0].set_title('Power Spectrum - Original Signal')
    axs[0].set_ylabel('PSD [V²/Hz]')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xlim([0, min(100, fs/2)])  # Focus on relevant frequency range
    
    # Calculate and plot power spectrum of filtered signal
    f_filt, pxx_filt = sig.welch(signals['final_filtered'], fs, nperseg=min(1024, len(signals['final_filtered'])))
    axs[1].semilogy(f_filt, pxx_filt, label='Filtered Signal', color='red')
    axs[1].set_title('Power Spectrum - Filtered Signal')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('PSD [V²/Hz]')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_xlim([0, min(100, fs/2)])  # Focus on relevant frequency range
    
    # Add shaded areas highlighting the removed frequency bands
    # Baseline wander region (0-0.5 Hz)
    axs[0].axvspan(0, 0.5, alpha=0.2, color='green', label='Baseline Wander Region')
    # Powerline interference (50 Hz)
    axs[0].axvspan(48, 52, alpha=0.2, color='orange', label='Powerline Interference')
    
    plt.tight_layout()
    save_figure(fig3, f'record_{record_num}_spectra.png', output_dir)
    plt.show()


def analyze_preprocessing_effects(original_signal, processed_signal, fs):
    # Compute power spectra
    f_orig, pxx_orig = sig.welch(original_signal, fs, nperseg=min(1024, len(original_signal)))
    f_proc, pxx_proc = sig.welch(processed_signal, fs, nperseg=min(1024, len(processed_signal)))

    # 1. Estimate SNR improvement
    # For simplicity, we'll use the ratio of total signal power to the power in noise bands
    
    # Find indices for the ECG frequency band (typically 0.5-40 Hz)
    ecg_band_idx = np.logical_and(f_orig >= 0.5, f_orig <= 40)
    
    # Find indices for noise bands (below 0.5 Hz and above 40 Hz)
    noise_band_idx = np.logical_or(f_orig < 0.5, f_orig > 40)
    
    # Compute power in these bands
    orig_ecg_power = np.sum(pxx_orig[ecg_band_idx])
    orig_noise_power = np.sum(pxx_orig[noise_band_idx])
    
    proc_ecg_power = np.sum(pxx_proc[ecg_band_idx])
    proc_noise_power = np.sum(pxx_proc[noise_band_idx])
    
    # Compute SNR in dB
    if orig_noise_power > 0:
        orig_snr = 10 * np.log10(orig_ecg_power / orig_noise_power)
    else:
        orig_snr = float('inf')
        
    if proc_noise_power > 0:
        proc_snr = 10 * np.log10(proc_ecg_power / proc_noise_power)
    else:
        proc_snr = float('inf')
    
    snr_improvement = proc_snr - orig_snr
    
    # 2. Quantify baseline wander reduction
    # Power in very low frequencies (below 0.5 Hz)
    baseline_idx = f_orig < 0.5
    orig_baseline_power = np.sum(pxx_orig[baseline_idx])
    proc_baseline_power = np.sum(pxx_proc[baseline_idx])
    
    baseline_reduction_pct = 100 * (1 - proc_baseline_power / orig_baseline_power) if orig_baseline_power > 0 else 0
    
    # 3. Quantify powerline interference reduction
    # Power around 50 Hz (or 60 Hz)
    powerline_idx = np.logical_and(f_orig >= 49, f_orig <= 51)  # Adjust for 60 Hz if needed
    orig_powerline_power = np.sum(pxx_orig[powerline_idx])
    proc_powerline_power = np.sum(pxx_proc[powerline_idx])
    
    powerline_reduction_pct = 100 * (1 - proc_powerline_power / orig_powerline_power) if orig_powerline_power > 0 else 0
    
    # Calculate overall signal variance change
    orig_var = np.var(original_signal)
    proc_var = np.var(processed_signal)
    variance_reduction_pct = 100 * (1 - proc_var / orig_var) if orig_var > 0 else 0
    
    # Return all metrics
    metrics = {
        'snr_original_db': orig_snr,
        'snr_processed_db': proc_snr,
        'snr_improvement_db': snr_improvement,
        'baseline_reduction_pct': baseline_reduction_pct,
        'powerline_reduction_pct': powerline_reduction_pct,
        'variance_reduction_pct': variance_reduction_pct
    }
    
    return metrics


def process_and_analyze_record(record_num, db_path, config, output_dir='./output', start_time=0, duration=10):
    # Load the record
    record, _ = load_record(record_num, db_path)
    if record is None:
        print(f"Could not load record {record_num}")
        return None
    
    # Extract signal segment
    signal_data, time, fs, channel_name = extract_signal_segment(record, start_time, duration)
    if signal_data is None:
        print("Failed to extract signal data")
        return None
    
	# Apply preprocessing
    filtered_signal, intermediate_signals = preprocess_ecg_signal(signal_data, fs, config['preprocessing'])

	# Visualize preprocessing steps
    visualize_preprocessing_steps(intermediate_signals, time, fs, record_num, channel_name, output_dir)

	# Analyze preprocessing effects
    metrics = analyze_preprocessing_effects(signal_data, filtered_signal, fs)
    

# ===========================================================
# Phase 3: R-Peak Detection and Heart Rate Analysis			=
# ===========================================================


def detect_r_peaks(ecg_signal, fs, threshold_factor=0.6):
    threshold = threshold_factor * np.max(ecg_signal)
    min_distance = int(fs * 0.3) 
    r_peaks, _ = find_peaks(ecg_signal, height=threshold, distance=min_distance)
    return r_peaks


def calculate_rr_intervals(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs
    return rr_intervals


def analyze_heart_rate(rr_intervals):
    avg_hr = 60 / np.mean(rr_intervals)
    observations = []
    if avg_hr < 60:
        observations.append("Bradycardia detected (low heart rate).")
    elif avg_hr > 100:
        observations.append("Tachycardia detected (high heart rate).")
    else:
        observations.append("Normal heart rate.")
    return avg_hr, observations


def plot_r_peaks(ecg_signal, r_peaks, time, fs, record_num, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(time, ecg_signal, label="ECG Signal", color="blue")
    plt.scatter(r_peaks / fs, ecg_signal[r_peaks], color="red", label="R-peaks", zorder=5)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mV]")
    plt.title(f"ECG Signal with Detected R-peaks - Record {record_num}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    output_path = f"{output_dir}/r_peaks_record_{record_num}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved: {output_path}")


def plot_rr_intervals(rr_intervals, record_num, output_dir):
    plt.figure(figsize=(8, 6))
    plt.hist(rr_intervals, bins=20, color="green", alpha=0.7)
    plt.xlabel("RR Interval (s)")
    plt.ylabel("Frequency")
    plt.title(f"RR Interval Distribution - Record {record_num}")
    plt.grid()
    plt.tight_layout()
    output_path = f"{output_dir}/rr_intervals_record_{record_num}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved: {output_path}")


def plot_heart_rate(rr_intervals, record_num, output_dir):
    heart_rate = 60 / rr_intervals
    beat_numbers = np.arange(1, len(heart_rate) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(beat_numbers, heart_rate, marker="o", linestyle="-", color="blue", label="Heart Rate")
    plt.axhline(y=np.mean(heart_rate), color="red", linestyle="--", label=f"Mean: {np.mean(heart_rate):.1f} BPM")
    plt.xlabel("Beat Number")
    plt.ylabel("Heart Rate (BPM)")
    plt.title(f"Heart Rate Over Time - Record {record_num}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    output_path = f"{output_dir}/heart_rate_record_{record_num}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Figure saved: {output_path}")


# ===========================================================
# Phase 4: Arrhythmia Detection and Classification          =
# ===========================================================

def save_confusion_matrix(cm, classifier_name, output_dir):
    """Save the confusion matrix as an image."""
    ensure_dir_exists(output_dir)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {classifier_name}')
    output_path = os.path.join(output_dir, f'confusion_matrix_{classifier_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(fpr, tpr, roc_auc, classifier_name, output_dir):
    ensure_dir_exists(output_dir)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.legend(loc="lower right")
    output_path = os.path.join(output_dir, f'roc_curve_{classifier_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path)
    plt.close()


def is_heart_rate_normal(average_heart_rate):
    if 60 <= average_heart_rate <= 100:
        print(f"Average heart rate ({average_heart_rate:.2f} bpm) is Normal.")
        return "Normal"
    else:
        print(f"Average heart rate ({average_heart_rate:.2f} bpm) is Abnormal, potentially indicating arrhythmia.")
        return "Abnormal (Arrhythmia)"


def preprocess_signal(signal):
    """Placeholder for signal preprocessing (if needed)."""
    return signal


def train_and_evaluate_classifier(classifier, classifier_name, X_train, X_test, y_train, y_test, output_dir):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    analyze_classification_results(y_test, y_pred, y_pred_proba, classifier_name, output_dir)
    
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")


def analyze_classification_results(y_test, y_pred, y_pred_proba, classifier_name, output_dir):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{classifier_name} Classifier Performance:")
    print("=" * 80)
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print("\nDetailed Analysis:")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  True Negatives (Correctly Normal): {tn}")
        print(f"  False Positives (Normal misclassified as Abnormal): {fp}")
        print(f"  False Negatives (Abnormal misclassified as Normal): {fn}")
        print(f"  True Positives (Correctly Abnormal): {tp}")
    else:
        print("  Confusion matrix is not 2x2, detailed analysis is not available.")
    print("=" * 80)
    save_confusion_matrix(cm, classifier_name, output_dir)

    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        save_roc_curve(fpr, tpr, roc_auc, classifier_name, output_dir)
    else:
        print(f"  ROC curve cannot be generated for {classifier_name} as the test set contains only one class.")


def extract_features_and_labels(record_name, segment_length_sec=10, min_peaks=5):
    features_list = []
    labels_list = []
    
    record = wfdb.rdrecord(f'signal/{record_name}')
    annotation = wfdb.rdann(f'signal/{record_name}', 'atr')
    signal_raw = preprocess_signal(record.p_signal[:, 0], record.fs)
    fs_local = record.fs
    peaks_local, _ = find_peaks(signal_raw, distance=int(0.4 * fs_local))  
    ann_samples = annotation.sample
    ann_symbols_all = annotation.symbol
    peak_annotations = []

    for peak_idx in peaks_local:
        closest_ann_idx = np.argmin(np.abs(ann_samples - peak_idx))
        if np.abs(ann_samples[closest_ann_idx] - peak_idx) < (0.14 * fs_local):
            peak_annotations.append(ann_symbols_all[closest_ann_idx])
        else:
            peak_annotations.append('N')

    segment_length_samples_local = int(segment_length_sec * fs_local)

    for i in range(0, len(signal_raw) - segment_length_samples_local + 1, segment_length_samples_local):
        segment_start = i
        segment_end = i + segment_length_samples_local
        segment_peaks_indices_mask = (peaks_local >= segment_start) & (peaks_local < segment_end)
        segment_peak_values = peaks_local[segment_peaks_indices_mask]

        if len(segment_peak_values) < min_peaks:
            continue

        # Calculate RR intervals
        segment_rr_intervals = np.diff(segment_peak_values) / fs_local
        if len(segment_rr_intervals) == 0:
            continue

        # Calculate features
        mean_rr = np.mean(segment_rr_intervals)
        std_rr = np.std(segment_rr_intervals)
        sdnn = np.std(segment_rr_intervals)  # Standard deviation of RR intervals
        rmssd = np.sqrt(np.mean(np.diff(segment_rr_intervals) ** 2))  # Root mean square of successive differences
        pnn50 = len(np.where(np.abs(np.diff(segment_rr_intervals)) > 0.05)[0]) / len(segment_rr_intervals) * 100  # Percentage of successive RR intervals > 50ms

        # Determine label
        current_segment_peak_orig_indices = [idx for idx, p_val in enumerate(peaks_local) if (p_val >= segment_start and p_val < segment_end)]
        segment_beat_symbols = [peak_annotations[idx] for idx in current_segment_peak_orig_indices]

        if not segment_beat_symbols:
            label = 0
        else:
            label = 0 if all(sym == 'N' for sym in segment_beat_symbols) else 1

        # Append features and label
        features_list.append([mean_rr, std_rr, sdnn, rmssd, pnn50])
        labels_list.append(label)

    return features_list, labels_list


def select_features(X_train, X_test, y_train):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector


def preprocess_signal(signal, fs):
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 50.0 / nyq
    b, a = butter(2, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def main():
    db_path = "./signal"
    output_dir = "./output"
    config = {
        'preprocessing': {
            'bandpass_lowcut': 0.5,
            'bandpass_highcut': 50.0,
            'notch_freq': 50.0,
            'notch_quality': 30.0
        }
    }

    print("\n" + "=" * 80)
    print(f"{'Phase 1: Data Loading and Preprocessing':^80}")
    print("=" * 80)

    # Phase 1: Data Loading and Preprocessing
    record_list = ['100', '101', '102']
    for record_num in record_list:
        try:
            record, annotation = load_record(record_num, db_path)
            plot_original_signal(record, record_num, start_time=0, duration=10, output_dir=output_dir)
            plot_ecg_with_annotations(record, annotation, record_num, start_time=0, duration=10, output_dir=output_dir)
        except Exception as e:
            print(f"Error processing record {record_num}: {e}")

    print("\n" + "=" * 80)
    print(f"{'Phase 2: Signal Preprocessing and Noise Removal':^80}")
    print("=" * 80)

    # Phase 2: Signal Preprocessing and Noise Removal
    for record_num in record_list:
        try:
            record, _ = load_record(record_num, db_path)
            signal, time, fs, channel_name = extract_signal_segment(record, start_time=0, duration=10)
            filtered_signal, intermediate_signals = preprocess_ecg_signal(signal, fs, config['preprocessing'])
            visualize_preprocessing_steps(intermediate_signals, time, fs, record_num, channel_name, output_dir)
        except Exception as e:
            print(f"Error preprocessing record {record_num}: {e}")

    print("\n" + "=" * 80)
    print(f"{'Phase 3: R-Peak Detection and Heart Rate Analysis':^80}")
    print("=" * 80)

    # Phase 3: R-Peak Detection and Heart Rate Analysis
    for record_num in record_list:
        try:
            record, _ = load_record(record_num, db_path)
            signal, time, fs, _ = extract_signal_segment(record, start_time=0, duration=10)
            filtered_signal, _ = preprocess_ecg_signal(signal, fs, config['preprocessing'])
            r_peaks = detect_r_peaks(filtered_signal, fs)
            rr_intervals = calculate_rr_intervals(r_peaks, fs)
            avg_hr, observations = analyze_heart_rate(rr_intervals)
            print(f"Record {record_num}: Average Heart Rate = {avg_hr:.2f} BPM")
            for obs in observations:
                print(f"  - {obs}")
            plot_r_peaks(filtered_signal, r_peaks, time, fs, record_num, output_dir)
            plot_rr_intervals(rr_intervals, record_num, output_dir)
            plot_heart_rate(rr_intervals, record_num, output_dir)
        except Exception as e:
            print(f"Error analyzing record {record_num}: {e}")

    print("\n" + "=" * 80)
    print(f"{'Phase 4: Arrhythmia Detection and Classification':^80}")
    print("=" * 80)

    # Phase 4: Arrhythmia Detection and Classification
    all_features = []
    all_labels = []

    for record_num in record_list:
        try:
            feats, labs = extract_features_and_labels(record_num)
            if feats:
                all_features.extend(feats)
                all_labels.extend(labs)
            else:
                print(f"No features/labels extracted for record {record_num}.")
        except Exception as e:
            print(f"Error processing record {record_num}: {e}")

    if all_features:
        features_df = pd.DataFrame(all_features, columns=['mean_rr', 'std_rr', 'sdnn', 'rmssd', 'pnn50'])
        features_df['label'] = all_labels

        X = features_df[['mean_rr', 'std_rr', 'sdnn', 'rmssd', 'pnn50']]
        y = features_df['label']

        # Balance dataset using SMOTE
        print("\nBalancing dataset with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        # Feature selection
        print("Performing feature selection...")
        X_train_selected, X_test_selected, selector = select_features(X_train, X_test, y_train)

        # Train and evaluate classifiers
        classifiers = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
        }

        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            train_and_evaluate_classifier(clf, name, X_train_selected, X_test_selected, y_train, y_test, output_dir)

        # Voting Classifier
        print("\nTraining Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('dt', classifiers["Decision Tree"]),
                ('rf', classifiers["Random Forest"]),
                ('lr', classifiers["Logistic Regression"])
            ],
            voting='soft'
        )
        train_and_evaluate_classifier(voting_clf, "Voting Classifier", X_train_selected, X_test_selected, y_train, y_test, output_dir)
    else:
        print("No features were extracted for the ML model. Cannot proceed with training.")


if __name__ == "__main__":
    main()
