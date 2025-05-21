import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_radial_profile(psd_2d):
    """Calculates the radially averaged 1D PSD profile."""
    h, w = psd_2d.shape
    center_y, center_x = h // 2, w // 2

    # Create index grids and calculate distance from center
    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_int = r.astype(int) # Use integer radii for binning

    # Calculate sum of PSD values and counts for each integer radius
    try:
        # Ensure ravel order matches: (values, weights)
        tbin = np.bincount(r_int.ravel(), psd_2d.ravel())
        nr = np.bincount(r_int.ravel())
    except ValueError as e:
         logging.error(f"Error in bincount, possibly due to indices: {e}")
         return None, None

    # Calculate radial average, handle division by zero
    radial_profile = np.zeros_like(tbin, dtype=float)
    valid_bins = nr > 0
    radial_profile[valid_bins] = tbin[valid_bins] / nr[valid_bins]

    # The index of radial_profile corresponds to the integer radius
    radii = np.arange(len(radial_profile))

    return radii, radial_profile

def analyze_psd_profile(radii, profile, image_shape):
    """Calculates summary statistics from the 1D radial profile."""
    if profile is None:
        return {}

    h, w = image_shape
    max_radius = min(h // 2, w // 2) # Max meaningful radius

    # Define frequency bands (adjust radii as needed for 256x256)
    # Example bands: Low (0-10), Mid (11-50), High (51-max_radius)
    low_freq_mask = (radii >= 0) & (radii <= 10)
    mid_freq_mask = (radii > 10) & (radii <= 50)
    high_freq_mask = (radii > 50) & (radii <= max_radius)

    # Ensure masks are within the profile length
    profile_len = len(profile)
    low_freq_mask = low_freq_mask[:profile_len]
    mid_freq_mask = mid_freq_mask[:profile_len]
    high_freq_mask = high_freq_mask[:profile_len]

    # Calculate total power in each band (summing average power * count, roughly)
    # A simpler approach: sum the profile values in the band
    # Note: This isn't strictly "total power" but a proportional measure
    power_low = np.sum(profile[low_freq_mask])
    power_mid = np.sum(profile[mid_freq_mask])
    power_high = np.sum(profile[high_freq_mask])
    total_power = np.sum(profile[:max_radius+1]) # Sum up to max radius

    # Avoid division by zero if total power is zero
    if total_power == 0:
        ratio_low = ratio_mid = ratio_high = 0
    else:
        ratio_low = power_low / total_power
        ratio_mid = power_mid / total_power
        ratio_high = power_high / total_power

    return {
        "total_power_approx": total_power,
        "power_in_low_freq_(0-10)": power_low,
        "power_in_mid_freq_(11-50)": power_mid,
        "power_in_high_freq_(51-max)": power_high,
        "ratio_low_freq": ratio_low,
        "ratio_mid_freq": ratio_mid,
        "ratio_high_freq": ratio_high,
    }

def plot_radial_profile(radii, profile, output_path, title="1D Radially Averaged PSD"):
    """Saves a plot of the 1D radial profile (log-log scale)."""
    if profile is None or radii is None:
        logging.warning(f"Cannot plot None profile: {output_path}")
        return
    try:
        plt.figure(figsize=(8, 5))
        # Plot only up to max radius, skip DC component (index 0) for log-log
        max_radius = min(profile.shape[0] // 2, radii.shape[0] // 2) # Sensible max extent
        valid_indices = radii > 0 # Exclude DC
        valid_indices = valid_indices[:len(profile)] # Ensure mask matches profile length

        if np.any(valid_indices):
             plt.loglog(radii[valid_indices], profile[valid_indices])
             plt.xlabel("Spatial Frequency (Radius from center, arbitrary units)")
             plt.ylabel("Average Power (Log Scale)")
             plt.title(title)
             plt.grid(True, which="both", ls="--", alpha=0.5)
             plt.savefig(output_path, bbox_inches='tight')
             plt.close()
             logging.debug(f"Saved radial profile plot to {output_path}")
        else:
             logging.warning(f"No valid data points found to plot for {output_path}")
             plt.close()

    except Exception as e:
        logging.error(f"Failed to save radial profile plot {output_path}: {e}")
        plt.close()


def main(args):
    if not os.path.exists(args.psd_npy_file):
        logging.error(f"Input PSD file not found: {args.psd_npy_file}")
        return

    logging.info(f"Loading 2D PSD from: {args.psd_npy_file}")
    try:
        avg_psd_2d = np.load(args.psd_npy_file)
        logging.info(f"Loaded PSD with shape: {avg_psd_2d.shape}")
    except Exception as e:
        logging.error(f"Error loading .npy file {args.psd_npy_file}: {e}")
        return

    # --- Calculate Radial Profile ---
    radii, radial_profile = calculate_radial_profile(avg_psd_2d)

    if radial_profile is None:
        logging.error("Failed to calculate radial profile.")
        return

    print("\n--- Radial Profile Analysis ---")
    print(f"File: {args.psd_npy_file}")

    # --- Calculate Summary Stats ---
    summary_stats = analyze_psd_profile(radii, radial_profile, avg_psd_2d.shape)

    if summary_stats:
        print("Summary Statistics:")
        for key, value in summary_stats.items():
            print(f"  {key}: {value:.4e}") # Use scientific notation for potentially large/small numbers
    else:
        print("Could not calculate summary statistics.")

    # --- Optional: Save/Plot Radial Profile ---
    if args.save_plot:
        # Construct output path based on input filename
        base_name = os.path.splitext(os.path.basename(args.psd_npy_file))[0]
        plot_filename = f"{base_name}_radial_profile.png"
        # Save in the same directory as the .npy file by default
        output_plot_path = os.path.join(os.path.dirname(args.psd_npy_file), plot_filename)

        print(f"\nSaving radial profile plot to: {output_plot_path}")
        plot_radial_profile(radii, radial_profile, output_plot_path, title=f"Radial PSD - {base_name}")

    # --- Optional: Save radial profile data ---
    if args.save_data:
        data_filename = f"{base_name}_radial_profile_data.csv"
        output_data_path = os.path.join(os.path.dirname(args.psd_npy_file), data_filename)
        print(f"\nSaving radial profile data to: {output_data_path}")
        try:
            # Combine radii and profile into a 2-column array and save as CSV
            profile_data = np.vstack((radii, radial_profile)).T
            np.savetxt(output_data_path, profile_data, delimiter=',', header='Radius,AveragePower', comments='')
        except Exception as e:
            logging.error(f"Failed to save radial profile data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a saved 2D PSD .npy file by calculating its 1D radial profile and summary statistics.")
    parser.add_argument("psd_npy_file", type=str, help="Path to the average 2D PSD .npy file to analyze.")
    parser.add_argument("--save_plot", action="store_true", help="Save a plot of the 1D radial profile (log-log scale).")
    parser.add_argument("--save_data", action="store_true", help="Save the calculated 1D radial profile data (Radius, Power) to a CSV file.")
    args = parser.parse_args()
    main(args)