README

************************************
** WEATHER
************************************
get_weather.py
    - scrape weather underground website for data
    - dates hardcoded between 5/2/2016 and 6/8/2016 (daily)
    - data in file "../output/weather/wu_{date}.csv"
    - ~15 minute intervals containing
        - temp, humid, pressure, dew point, wind speed, precipitation

parse_aq.py
    - raw air quality files in "../output/air_quality/StationDate {*}.csv"
    - sourced from NY State Dpt of Environmental Conservation? (***UNKNOWN ORIGIN)
    - there are 38 files, 1 hour time resolution
    - contain O3, PM2.5, CO, NOx, and SO2 (only O3 and PM2.5 are extracted)
    - files read, cleaned, and parsed into "../output/aq_vs_time.csv"

convert_weather_aq.py
    - Purpose: clean and put all weather data into csv file
    - Reads in time of scans from "../output/scan_times.csv" (***UNKNOWN ORIGIN)
    - each scan name has a scan time associated
    - Attributes: temp(F), dew point, pressure, humidity, precipitation, PM2.5, O3
    - For each scan find time closest to scan time in "../output/weather/wu_{date}.csv"
    - record temp(F), dew point, pressure, humidity, precipitation for each scan
    - from "../output/aq_vs_time.csv", find closest time and record PM2.5 and O3
    - put all weather data for all scans in "../output/scan_conditions.csv"

************************************
** SPECTRA
************************************
plot_fullscan.py
    - plots an RGB corrected image from veg_* cube

project.py
    - performs k-means clustering on scan veg_00055
    - 10 clusters, fit with reduced spatial dimensions (every 16th pixel)
    - result of km fit saved in "../output/km_00055_ncl10_seed314.pkl"
    - veg_spectrum is from labels 2 and 5 from kmeans
    - sky_spectrum from first 700 rows averaged
    - reflectance = (veg-veg.min) / (sky-sky.min)

process_scans.py
    - creates separate files with classes for each scan in "../data/veg_*.raw"
    - Output files:
        - vegetation: "../output/veg_specs/veg_specs_*.npy"
            - from kmeans file "../output/km_00055_ncl10_see314_labs_full.npy"
            - labels 2 or 5
        - sky (mean): "../output/sky_specs/sky_spec_*.npy"
            - average spectra of 10% of image (rows 0-160)
        - saturated (#): "../output/saturated/nsat_*.npy"
            - number of pixels where the saturation value is 2^12 - 1
        - buildings: "../output/bld_specs/bld_specs_*.npy"
          buildings (mean)" "../output/bld_specs/bld_specs_avg_*.npy"
            - rows 990-1034
            - columns 799-956
        - alternate building: "../output/alt_specs/alt_specs_*.npy"
          alternate building (mean): "../output/alt_specs/alt_specs_avg_*.npy"
          - 3 regions:
            - r1r = [933,970], r1c = [344,364]
            - r2r = [970,1000], r2c = [352,364]
            - r3r = [931,962], r3c = [455,477]
        - new building: "../output/new_specs/new_specs_*.npy"
          new building (mean): "../output/new_specs/new_specs_avg_*.npy"
          - rows 1125-1137
          - columns 790-829

count_clip.py
    - identifies which scans have clipped sky spectra
        - clipped is defined as having pixels with 2**12 (4096) amplitude
    - it counts the number of pixels equal to 2**12 (clipped) for each pixel for each scan
    - it then writes the results to file "../output/clipped/nclip_*.npy"

check_sky_sat.py
    - takes input files from "../output/clipped/nclip_*.npy"
    - defines a 'good' scan as one where <5% of pixels have <5% of wavelengths saturated
    - write names of unsaturated scans to "../output/scans_unsaturated.npy"

remove_bad_scans.py
    - loads unsaturated scans "../output/scans_unsaturated.npy"
    - remove bad scans from "../data/veg_*.hdr"
        - bad scans defined as not having 1601 or 1600 columns
    - write good scans to "../output/good_scans.npy"

************************************
** ANALYZE SPECTRA & WEATHER
************************************
plt_scan_times.py
    - plots "../output/scan_times.png" of 'good' scans
    - uses "../output/scan_conditions.csv" for weather conditions

sky_corr.py & sky_corr_ref.py
    - Calculate the differences between sky from one scan to another
    - load the sky spectra from "../output/sky_specs/*.npy"
    - select good ones from "../output/good_scans.npy"

analyze_specs.py
    - get blt and veg spectra for good scans, calculate ratio of ratios, calculate NDVI,
      find/plot correlation with env. conditions
    - good scans obtained from "../output/good_scans.npy"
        - defined in remove_bad_scans.py
        - good scan = <5% of pixels have <5% of wavelengths saturated
    - comparison set (blt spectra) obtained from:
        - "../output/new_specs/new_specs_avg*.npy" (***)
        - (***) can't find how those are produced
        - limited to the "good" scans
        - spectra of blt normalized in array "norm"
        - ratio: rat=norm/norm[0]
    - vegetation spectra (veg) obtained from: 
        - "../output/veg_patch_specs.npy"
        - produced in sky_corr.py
            - only "good" scans
            - hard coded rectangle with kmeans labels 2 or 5
            - rows 1050-1300, columns 650-1150
        - spectra of veg normalized in array "vnorm"
        - ratio: vrat=vnorm/vnorm[0]
    - ratio of ratio calculated (brat = vrat/rat)
    - PCA, Factor Analysis, ICA set to not run (but can if needed)
    - NDVI is obtained 
        - using reflectances (ref=(vegs-vegs.min)/(skys-skys.min))
        - NIR at 860nm, and Red(visible) at 670nm
        - ndvi=(ref_nir-ref_red)/(ref_nir+ref_red)
    - sky conditions from "../output/scan_conditions.csv" for "good" scans only
        - P = {temps, humidity, pm2.5, O3}
        - time obtained in seconds
    - call ratio_of_ratios at 750nm (brat750)
    - call ratio_of_ratios at 1000nm (brat1000)
    - brightness = brat700 / brat1000
    - using least square linear fit brightness=w.P

analyze_specs_bldcomp.py
    - same as analyze_specs.py except ratio of ratios is that of the buildings' spectra
      split into right and left
    - right buildings used instead of blt, and left buildings instead of veg
    - right buildings spectra from "../output/blds_right.npy"
    - left buildings spectra from "../output/blds_left.npy"

brightness_ndvi.py
    - looks for variation in buildings spectra normalized 395.46 to 463.93micron
    - loads buildings' mean spectra from "../output/bld_specs_avg/*.npy"
        - spectra of blt normalized in array "norm"
        - ratio: rat=norm/norm[0]
    - loads vegetation spectra from "../output/veg_patch_specs.npy"
        - spectra of veg normalized in array "vnorm"
        - ratio: vrat=vnorm/vnorm[0]
    - ratio of ratio calculated (norm = vrat/rat)
    - brightness claculated (brgt = norm.mean(1))
    - loads sky spectra from "../output/sky_specs/*.npy"
    - estimate ndvi using 860 and 670 wavelengths
    - reject outliers with brightness < 2.0
    - read weather conditions "../output/scan_conditions.csv"
    - plot brightness vs ndvi ("../output/brightness_vs_ndvi.png")
    - plot median ratio of ratios as function of wavelengths 
        - ("../output/brightness_median_stddev.png")
        - ("../output/brightness_median_ex.png")

************************************
** EXTRA ANALYSIS
************************************
project2.py
    - opens km pkl file
    - normalizes spectra, and km.predict the normalized cube of veg_00000
    - plots all labels map
    - plots only vegetation
    - plots reflectance (from mean sky (first 700 rows))

project3.py
    - opens veg_00055
    - reads km pkl file
    - tags km on full cube
    - saves full cube prediction to "../output/km_00055_ncl10_see314_labs_full.npy"
    - extracts veg spectra and normalizes
    - extracts sky (top 700 rows) and averages
    - plot reflectances superimposed

project4.py
    - opens veg_00055 and veg_00309 (1 week separation)
    - generates luminosity image from both
    - poly-fits the offset and amplitude
    - plots the difference of scans in "../output/scan_diff_00055_00309.eps"

project5.npy
    - plots "../output/reflectance_time.png"
    - uses outputs from:
        - "../output/veg_spec_avg.npy"
        - "../output/veg_spec_sig.npy"
        - "../output/sky_specs/sky_spec_{*}.npy"

project6.py
    - gets veg pixels as produced by project3.py
    - plot spectra
    - read sky from "../output/sky_specs/sky_spec_00055.npy"
    - plot reflectance
    - plot residual
    
project7.py
    - gets veg and sky pixels, but doesn't do anything


************************************
** MISC
************************************
hyss_util.py
    - used to read raw and header scan files

run_obs.py
    - script to take scan with camera
    
unpack_vegs.py
    - nothing (almost empty)

gradient_boosted_regressor.py
    - seems incomplete
    - perform sklearn GBR to predict NDVI
    - plot measured vs predicted NDVI ("../output/predict_ndvi.pdf")
    - plot feature importances ("../output/predict_ndvi_fi.pdf")



