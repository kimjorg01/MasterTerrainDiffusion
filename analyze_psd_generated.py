import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_radial_profile(psd_2d):
    """Calculates the radially averaged 1D PSD profile."""
    if psd_2d is None:
        logging.error("PSD 2D is None, cannot calculate radial profile.")
        return None, None
    h, w = psd_2d.shape
    center_y, center_x = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_int = r.astype(int)

    try:
        tbin = np.bincount(r_int.ravel(), psd_2d.ravel())
        nr = np.bincount(r_int.ravel())
    except ValueError as e:
        logging.error(f"Error in bincount, possibly due to indices (max r_int: {r_int.max()}, tbin length target: {psd_2d.size}): {e}")
        return None, None

    radial_profile = np.zeros_like(tbin, dtype=float)
    valid_bins = nr > 0
    
    # Ensure tbin and nr have the same length before division
    min_len = min(len(tbin), len(nr))
    tbin = tbin[:min_len]
    nr = nr[:min_len]
    radial_profile = radial_profile[:min_len]
    valid_bins = valid_bins[:min_len]

    radial_profile[valid_bins] = tbin[valid_bins] / nr[valid_bins]
    radii = np.arange(len(radial_profile))
    return radii, radial_profile

def analyze_psd_profile(radii, profile, image_shape):
    """Calculates summary statistics from the 1D radial profile."""
    if profile is None or radii is None:
        return {}

    h, w = image_shape
    # Max radius for analysis should not exceed the smallest half-dimension
    max_r_analysis = min(h // 2, w // 2, len(radii) -1, len(profile) -1)


    # Define frequency bands (adjust radii as needed for your image_shape)
    # Example bands: Low (0-10% of max_r_analysis), Mid (10%-40%), High (40%-100%)
    # These percentages are illustrative.
    low_freq_max_radius = int(0.1 * max_r_analysis)
    mid_freq_max_radius = int(0.4 * max_r_analysis)

    low_freq_mask = (radii >= 0) & (radii <= low_freq_max_radius)
    mid_freq_mask = (radii > low_freq_max_radius) & (radii <= mid_freq_max_radius)
    high_freq_mask = (radii > mid_freq_max_radius) & (radii <= max_r_analysis)
    
    # Ensure masks are within the profile length
    profile_len_for_mask = min(len(profile), max_r_analysis + 1)
    low_freq_mask = low_freq_mask[:profile_len_for_mask]
    mid_freq_mask = mid_freq_mask[:profile_len_for_mask]
    high_freq_mask = high_freq_mask[:profile_len_for_mask]
    
    current_profile = profile[:profile_len_for_mask]


    power_low = np.sum(current_profile[low_freq_mask]) if np.any(low_freq_mask) else 0
    power_mid = np.sum(current_profile[mid_freq_mask]) if np.any(mid_freq_mask) else 0
    power_high = np.sum(current_profile[high_freq_mask]) if np.any(high_freq_mask) else 0
    total_power = np.sum(current_profile[:max_r_analysis+1])

    if total_power == 0:
        ratio_low = ratio_mid = ratio_high = 0.0
    else:
        ratio_low = power_low / total_power
        ratio_mid = power_mid / total_power
        ratio_high = power_high / total_power

    return {
        "image_shape_h": h,
        "image_shape_w": w,
        "max_analysis_radius": max_r_analysis,
        "total_power_in_analyzed_radii": total_power,
        f"power_low_freq_radii(0-{low_freq_max_radius})": power_low,
        f"power_mid_freq_radii({low_freq_max_radius+1}-{mid_freq_max_radius})": power_mid,
        f"power_high_freq_radii({mid_freq_max_radius+1}-{max_r_analysis})": power_high,
        "ratio_low_freq": ratio_low,
        "ratio_mid_freq": ratio_mid,
        "ratio_high_freq": ratio_high,
    }

def plot_radial_profile(radii, profile, output_path, title="1D Radially Averaged PSD", max_plot_radius=None):
    """Saves a plot of the 1D radial profile (log-log scale)."""
    if profile is None or radii is None or len(profile) == 0 or len(radii) == 0:
        logging.warning(f"Cannot plot None or empty profile/radii: {output_path}")
        return
    try:
        plt.figure(figsize=(8, 5))
        
        # Determine valid range for plotting
        # Exclude DC component (index 0) for log-log if it exists and is problematic
        valid_indices = radii > 0 
        
        # Ensure radii and profile are sliced consistently if max_plot_radius is defined
        current_radii = radii
        current_profile = profile

        if max_plot_radius is not None:
            plot_mask = radii <= max_plot_radius
            current_radii = radii[plot_mask]
            current_profile = profile[plot_mask]
            valid_indices = valid_indices[plot_mask[:len(valid_indices)]] # Adjust valid_indices mask as well


        # Further ensure valid_indices matches the (potentially sliced) current_radii/profile
        valid_indices = valid_indices[:min(len(current_radii), len(current_profile))]
        final_radii_to_plot = current_radii[valid_indices]
        final_profile_to_plot = current_profile[valid_indices]


        if len(final_radii_to_plot) > 0 and len(final_profile_to_plot) > 0:
            # Filter out non-positive values in profile for log scale
            positive_profile_mask = final_profile_to_plot > 0
            if np.any(positive_profile_mask):
                plt.loglog(final_radii_to_plot[positive_profile_mask], final_profile_to_plot[positive_profile_mask])
                plt.xlabel("Spatial Frequency (Radius from center, k)")
                plt.ylabel("Average Power (Log Scale)")
                plt.title(title)
                plt.grid(True, which="both", ls="--", alpha=0.5)
                plt.savefig(output_path, bbox_inches='tight')
                logging.info(f"Saved radial profile plot to {output_path}")
            else:
                logging.warning(f"No positive profile data found to plot (after filtering DC and non-positive) for {output_path}")
        else:
            logging.warning(f"No valid data points found to plot (after filtering DC) for {output_path}")
        
        plt.close() # Close the figure to free memory

    except Exception as e:
        logging.error(f"Failed to save radial profile plot {output_path}: {e}")
        plt.close()


def main(args):
    if not os.path.exists(args.psd_npy_file):
        logging.error(f"Input PSD file not found: {args.psd_npy_file}")
        print(f"Error: Input PSD file '{args.psd_npy_file}' not found.")
        return

    logging.info(f"Loading 2D PSD from: {args.psd_npy_file}")
    try:
        avg_psd_2d = np.load(args.psd_npy_file)
        logging.info(f"Loaded PSD with shape: {avg_psd_2d.shape}")
    except Exception as e:
        logging.error(f"Error loading .npy file {args.psd_npy_file}: {e}")
        print(f"Error loading .npy file: {e}")
        return

    radii, radial_profile = calculate_radial_profile(avg_psd_2d)

    if radial_profile is None:
        logging.error("Failed to calculate radial profile.")
        print("Failed to calculate radial profile.")
        return

    print("\n--- Radial Profile Analysis ---")
    print(f"File: {args.psd_npy_file}")

    summary_stats = analyze_psd_profile(radii, radial_profile, avg_psd_2d.shape)

    if summary_stats:
        print("\nSummary Statistics from Radial Profile:")
        for key, value in summary_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4e}")
            else:
                print(f"  {key}: {value}")
    else:
        print("Could not calculate summary statistics for the radial profile.")

    output_dir = os.path.dirname(args.psd_npy_file) # Save outputs in the same dir as the .npy
    base_name = os.path.splitext(os.path.basename(args.psd_npy_file))[0]

    if args.save_plot:
        plot_filename = f"{base_name}_radial_profile.png"
        output_plot_path = os.path.join(output_dir, plot_filename)
        print(f"\nSaving radial profile plot to: {output_plot_path}")
        # Plot up to nyquist frequency, essentially min(H,W)/2
        max_plot_rad = min(avg_psd_2d.shape[0] // 2, avg_psd_2d.shape[1] // 2)
        plot_radial_profile(radii, radial_profile, output_plot_path, 
                              title=f"Radial PSD - {base_name}", max_plot_radius=max_plot_rad)

    if args.save_data:
        data_filename = f"{base_name}_radial_profile_data.csv"
        output_data_path = os.path.join(output_dir, data_filename)
        print(f"\nSaving radial profile data to: {output_data_path}")
        try:
            # Save only up to max_plot_rad to keep data consistent with plot
            valid_range = radii <= max_plot_rad
            profile_data = np.vstack((radii[valid_range], radial_profile[valid_range])).T
            np.savetxt(output_data_path, profile_data, delimiter=',', header='Radius,AveragePower', comments='')
            logging.info(f"Saved radial profile data to {output_data_path}")
        except Exception as e:
            logging.error(f"Failed to save radial profile data: {e}")
            print(f"Error saving radial profile data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a saved 2D PSD .npy file by calculating its 1D radial profile and summary statistics.")
    parser.add_argument("psd_npy_file", type=str, help="Path to the average 2D PSD .npy file to analyze.")
    parser.add_argument("--save_plot", action="store_true", help="Save a plot of the 1D radial profile (log-log scale).")
    parser.add_argument("--save_data", action="store_true", help="Save the calculated 1D radial profile data (Radius, Power) to a CSV file.")
    args = parser.parse_args()
    main(args)