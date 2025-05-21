import numpy as np
import pandas as pd


K_FACTOR_STATS = 3.0  
EPSILON = 1e-9        

WEIGHTS = {
    'stats': 0.30,
    'psd_abs': 0.20,
    'psd_ratio': 0.20,
    'classifier': 0.30
}

if abs(sum(WEIGHTS.values()) - 1.0) > 1e-6:
    raise ValueError("Weights must sum to 1.0")

TERRAIN_CLASSES = ['Flat', 'Ridges', 'Rivers'] 




dataset_data = {
    'Flat': {
        'stats': {'MH_avg': 0.4947, 'MH_std': 0.0511, 'R_avg': 0.1965, 'R_std': 0.0179, 'S_avg': 0.0097, 'S_std': 0.0011},
        'psd_abs': {'P_low': 7.0165e+13, 'P_mid': 1.3583e+09, 'P_high': 8.0611e+07},
        'psd_ratio': {'R_low': 9.9998e-01, 'R_mid': 1.9358e-05, 'R_high': 1.1489e-06},
        'classifier_acc': 1.0 
    },
    'Ridges': {
        'stats': {'MH_avg': 0.5265, 'MH_std': 0.0427, 'R_avg': 0.1825, 'R_std': 0.0151, 'S_avg': 0.0144, 'S_std': 0.0017},
        'psd_abs': {'P_low': 7.8663e+13, 'P_mid': 1.9488e+09, 'P_high': 9.7346e+07},
        'psd_ratio': {'R_low': 9.9997e-01, 'R_mid': 2.4774e-05, 'R_high': 1.2375e-06},
        'classifier_acc': 1.0
    },
    'Rivers': {
        'stats': {'MH_avg': 0.4280, 'MH_std': 0.0461, 'R_avg': 0.1855, 'R_std': 0.0199, 'S_avg': 0.0132, 'S_std': 0.0016},
        'psd_abs': {'P_low': 5.2602e+13, 'P_mid': 1.7527e+09, 'P_high': 8.9729e+07},
        'psd_ratio': {'R_low': 9.9996e-01, 'R_mid': 3.3318e-05, 'R_high': 1.7058e-06},
        'classifier_acc': 1.0
    }
}


generated_model_data = {
    'Flat': {
        'stats': {'MH_avg': 0.4297, 'R_avg': 0.1672, 'S_avg': 0.0106},
        'psd_abs': {'P_low': 5.3642e+13, 'P_mid': 1.4039e+09, 'P_high': 7.4088e+07},
        'psd_ratio': {'R_low': 9.9997e-01, 'R_mid': 2.6172e-05, 'R_high': 1.3811e-06},
        'classifier_acc': 49/50  
    },
    'Ridges': {
        'stats': {'MH_avg': 0.4414, 'R_avg': 0.1665, 'S_avg': 0.0146},
        'psd_abs': {'P_low': 5.6516e+13, 'P_mid': 1.9774e+09, 'P_high': 8.3965e+07},
        'psd_ratio': {'R_low': 9.9996e-01, 'R_mid': 3.4986e-05, 'R_high': 1.4856e-06},
        'classifier_acc': 25/50  
    },
    'Rivers': {
        'stats': {'MH_avg': 0.3470, 'R_avg': 0.1551, 'S_avg': 0.0127},
        'psd_abs': {'P_low': 3.5828e+13, 'P_mid': 1.5209e+09, 'P_high': 6.7199e+07},
        'psd_ratio': {'R_low': 9.9996e-01, 'R_mid': 4.2449e-05, 'R_high': 1.8755e-06},
        'classifier_acc': 43/50  
    }
}





random_noise_data = {
    'Flat': { 
        'stats': {'MH_avg': 0.4982, 'R_avg': 0.2887, 'S_avg': 0.2628},
        'psd_abs': {'P_low': 6.9313e+13, 'P_mid': 1.3822e+10, 'P_high': 2.7333e+10},
        'psd_ratio': {'R_low': 9.9941e-01, 'R_mid': 1.9930e-04, 'R_high': 3.9410e-04},
        'classifier_acc': 0/50  
    },
    'Ridges': { 
        'stats': {'MH_avg': 0.4980, 'R_avg': 0.2886, 'S_avg': 0.2630},
        'psd_abs': {'P_low': 6.9275e+13, 'P_mid': 1.3808e+10, 'P_high': 2.7341e+10},
        'psd_ratio': {'R_low': 9.9941e-01, 'R_mid': 1.9920e-04, 'R_high': 3.9444e-04},
        'classifier_acc': 50/50  
    },
    'Rivers': { 
        'stats': {'MH_avg': 0.4980, 'R_avg': 0.2886, 'S_avg': 0.2629},
        'psd_abs': {'P_low': 6.9278e+13, 'P_mid': 1.3828e+10, 'P_high': 2.7327e+10},
        'psd_ratio': {'R_low': 9.9941e-01, 'R_mid': 1.9949e-04, 'R_high': 3.9422e-04},
        'classifier_acc': 0/50  
    }
}
USE_RANDOM_NOISE_BASELINE = True 



def calculate_stats_similarity(gen_stats, ds_stats, k_factor):
    """Calculates similarity for basic statistical features."""
    
    ds_mh_std = max(ds_stats['MH_std'], EPSILON * abs(ds_stats['MH_avg']) if ds_stats['MH_avg'] != 0 else EPSILON)
    ds_r_std = max(ds_stats['R_std'], EPSILON * abs(ds_stats['R_avg']) if ds_stats['R_avg'] != 0 else EPSILON)
    ds_s_std = max(ds_stats['S_std'], EPSILON * abs(ds_stats['S_avg']) if ds_stats['S_avg'] != 0 else EPSILON)

    sim_mh = max(0, 1 - abs(gen_stats['MH_avg'] - ds_stats['MH_avg']) / (k_factor * ds_mh_std))
    sim_r  = max(0, 1 - abs(gen_stats['R_avg'] - ds_stats['R_avg']) / (k_factor * ds_r_std))
    sim_s  = max(0, 1 - abs(gen_stats['S_avg'] - ds_stats['S_avg']) / (k_factor * ds_s_std))
    return (sim_mh + sim_r + sim_s) / 3.0

def calculate_psd_abs_similarity(gen_psd_abs, ds_psd_abs):
    """Calculates similarity for PSD absolute power."""
    sim_p_low  = max(0, 1 - abs(gen_psd_abs['P_low'] - ds_psd_abs['P_low']) / (ds_psd_abs['P_low'] + EPSILON))
    sim_p_mid  = max(0, 1 - abs(gen_psd_abs['P_mid'] - ds_psd_abs['P_mid']) / (ds_psd_abs['P_mid'] + EPSILON))
    sim_p_high = max(0, 1 - abs(gen_psd_abs['P_high'] - ds_psd_abs['P_high']) / (ds_psd_abs['P_high'] + EPSILON))
    return (sim_p_low + sim_p_mid + sim_p_high) / 3.0

def calculate_psd_ratio_similarity(gen_psd_ratio, ds_psd_ratio):
    """Calculates similarity for PSD power ratios."""
    sim_r_low  = max(0, 1 - abs(gen_psd_ratio['R_low'] - ds_psd_ratio['R_low']) / (ds_psd_ratio['R_low'] + EPSILON))
    sim_r_mid  = max(0, 1 - abs(gen_psd_ratio['R_mid'] - ds_psd_ratio['R_mid']) / (ds_psd_ratio['R_mid'] + EPSILON))
    sim_r_high = max(0, 1 - abs(gen_psd_ratio['R_high'] - ds_psd_ratio['R_high']) / (ds_psd_ratio['R_high'] + EPSILON))
    return (sim_r_low + sim_r_mid + sim_r_high) / 3.0



def calculate_all_scores(model_data_to_score, dataset_gt_data, model_name="Model"):
    """Calculates and prints all CSS and TSS scores for a given model's data."""
    print(f"\n--- Calculating Scores for: {model_name} ---")
    per_class_css = {cls: {} for cls in TERRAIN_CLASSES}
    per_class_tss = {}
    
    for terrain_class in TERRAIN_CLASSES:
        if terrain_class not in model_data_to_score:
            print(f"Warning: Data for class '{terrain_class}' not found in {model_name} data. Skipping.")
            continue
        if terrain_class not in dataset_gt_data:
            print(f"Warning: Data for class '{terrain_class}' not found in dataset_gt_data. Skipping.")
            continue
            
        gen_class_data = model_data_to_score[terrain_class]
        ds_class_data = dataset_gt_data[terrain_class]

        css_stats = calculate_stats_similarity(gen_class_data['stats'], ds_class_data['stats'], K_FACTOR_STATS)
        css_psd_abs = calculate_psd_abs_similarity(gen_class_data['psd_abs'], ds_class_data['psd_abs'])
        css_psd_ratio = calculate_psd_ratio_similarity(gen_class_data['psd_ratio'], ds_class_data['psd_ratio'])
        css_classifier = gen_class_data['classifier_acc'] 

        per_class_css[terrain_class]['CSS_Stats'] = css_stats
        per_class_css[terrain_class]['CSS_PSD_Abs'] = css_psd_abs
        per_class_css[terrain_class]['CSS_PSD_Ratio'] = css_psd_ratio
        per_class_css[terrain_class]['CSS_Classifier'] = css_classifier

        class_tss = (
            WEIGHTS['stats'] * css_stats +
            WEIGHTS['psd_abs'] * css_psd_abs +
            WEIGHTS['psd_ratio'] * css_psd_ratio +
            WEIGHTS['classifier'] * css_classifier
        )
        per_class_tss[terrain_class] = class_tss

    
    df_css = pd.DataFrame.from_dict(per_class_css, orient='index')
    
    if per_class_tss:
        df_css['TSS_Class (0-1)'] = pd.Series(per_class_tss)
    
    print("\nComponent Similarity Scores (CSS) and Per-Class TSS (0-1 scale):")
    if not df_css.empty:
        print(df_css.to_string(float_format="%.4f"))
    else:
        print("No scores to display.")


    if not per_class_tss: 
        print(f"\nNo per-class TSS scores calculated for {model_name}. Cannot compute overall TSS.")
        return None

    overall_tss_0_1 = sum(per_class_tss.values()) / len(per_class_tss) 
    overall_tss_0_100 = overall_tss_0_1 * 100

    print(f"\nOverall Terrain Similarity Score (TSS) for {model_name}: {overall_tss_0_100:.2f} / 100")
    return overall_tss_0_100

if __name__ == "__main__":
    print("--- Terrain Similarity Score Calculation ---")
    print(f"Using K_FACTOR_STATS: {K_FACTOR_STATS}")
    print(f"Using Weights: {WEIGHTS}")

    
    generated_overall_tss = calculate_all_scores(generated_model_data, dataset_data, model_name="Generated Diffusion Model")

    
    if USE_RANDOM_NOISE_BASELINE:
        print("\nIMPORTANT: Random noise data has been populated with your provided values.")
        random_overall_tss = calculate_all_scores(random_noise_data, dataset_data, model_name="Random Noise Baseline")
    else:
        print("\nSkipping Random Noise Baseline calculation (USE_RANDOM_NOISE_BASELINE is False).")

