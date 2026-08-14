import settings
import utilities
import image_utilities
import image_manipulation
import calibration

import os
import time
import numpy as np
import cv2 as cv  # Import OpenCV library
import colour  # Import Colour Science library

from matplotlib import pyplot as plt
import colour_checker_detection   # Identifying the colorchecker
import copy


max_val = (pow(2, 8) - 1, pow(2, 16) - 1)  # 8bit: 0-255 - 16bit: 0-65635
# Color models corresponding to color spaces in settings.py
color_model = (colour.models.RGB_COLOURSPACE_sRGB, colour.models.RGB_COLOURSPACE_ADOBE_RGB1998,
               colour.models.RGB_COLOURSPACE_PROPHOTO_RGB, colour.models.RGB_COLOURSPACE_ACES2065_1)
bit_type = ('uint8', 'uint16')


def main():
    print()

    utilities.create_directories(settings.main_directory)  # Create imaging system directory structure
    start_time = 0
    end_time = 0

    # Start in main menu
    script_mode = utilities.int_prompt("Script mode (0: Apply corrections, 1: Rename carousel timelapse files, 2: Crop images, 3: Measure series color, " \
    "4: Calibrate, 5: Measure Colorchecker sample values,  6: Measure line / area data from image, ENTER: Exit): ",
                                       (0, 7))
    if script_mode is None:
        print("Closing program.")
        return

    if script_mode == 0:  # Apply corrections
        while True:
            focus_input = input("Name of the correction profile? ")
            if focus_input is not None:
                focus_input = focus_input.strip()

            profile_data = image_utilities.read_profile(focus_input)
            if profile_data is not None:
                break
            else:
                print("Invalid correction profile.")
        print()

        start_time = time.perf_counter()

        # Sort files alphabetically
        file_names = utilities.get_files('Exported Images', match_extension=settings.input_extension)
        study_names = []
        gray_files = []
        print("Correcting gray references ...")
        print()

        for file_name in file_names:
            if len(file_name.split('-')) > 1 and file_name.split('-')[1].split('_')[0] == "gray":
                # File name corresponds to ref. gray
                if study_names.count(file_name.split('_')[0]) < 1:
                    study_names.append(file_name.split('_')[0])

                if settings.prompt_margin_utilization:
                    use_margins = utilities.yes_no_prompt("Use safety margins?")
                else:
                    use_margins = True
                if use_margins:
                    # Get area in safety margins
                    img_gray = image_utilities.get_safe_area(image_utilities.read_image(file_name))
                else:
                    # Use full image
                    img_gray = image_utilities.read_image(file_name)

                # Correct ref. gray, write result
                image_utilities.write_image(image_manipulation.adjust_color(img_gray, profile_data[0]),
                                            file_name.split('.')[0], image_utilities.sample_path(file_name),
                                            '_cc.' + settings.output_extension)
                gray_files.append(os.path.join(image_utilities.sample_path(file_name),
                                               file_name.split('.')[0] + '_cc.' + settings.output_extension))
                os.remove(os.path.join(settings.main_directory, 'Exported Images', file_name))  # Remove original

        # Crop ref. grays
        for study_name in study_names:
            image_manipulation.crop_samples(study_name, ref_gray=True)

        if not settings.save_gray_images:
            # Remove ref. gray images
            for gray_file in gray_files:
                os.remove(os.path.join(settings.main_directory, gray_file))

        file_names = utilities.get_files('Exported Images', match_extension=settings.input_extension)
        sample_names = []
        est_data = [time.perf_counter(), 0]
        image_part_crop_settings = None
        use_image_part = None
        last_sample = None
        for i in range(len(file_names)):
            img = image_utilities.read_image(file_names[i])

            # Settings.check_capture_settings is now set to False so that recalibration doesn't need to be done if not needed
            # Since we now have a script for checking whether recalibration should be done based on color difference
            # It is the responsibility of the user to see that they use the right correction profile
            if settings.check_capture_settings and not image_utilities.compare_settings(img[1][1], profile_data[2]): 
                print("Skipping", file_names[i], "(Capture settings mismatch)")
                continue

            print("Correcting", file_names[i], '...')

            path = image_utilities.sample_path(file_names[i])
            if not os.path.exists(os.path.join(settings.main_directory, path)):
                os.makedirs(os.path.join(settings.main_directory, path))

            measurement_gray = image_utilities.read_crop(file_names[i].split('-')[0] + "-gray", True)
            if measurement_gray is None or int(file_names[i].split('.')[0].split('_')[1]) not in measurement_gray:

                if last_sample != file_names[i].split('_')[0]:  # Reset settings when moving to next sample
                    image_part_crop_settings = None
                    use_image_part = None
                if image_part_crop_settings is None:
                    utilities.print_color("Warning: Ref. gray not found!", 'warning')
                    if use_image_part is None:
                        use_image_part = utilities.yes_no_prompt("Use part of image for ref. grays?")
                if image_part_crop_settings is not None or (use_image_part is not None and use_image_part):
                    if settings.prompt_margin_utilization:
                        use_margins = utilities.yes_no_prompt("Use safety margins?")
                    else:
                        use_margins = True
                    if use_margins:
                        # Get area in safety margins
                        img_gray = image_utilities.get_safe_area(img)
                    else:
                        # Use full image
                        img_gray = img.copy()

                    # Correct ref. gray, write result
                    gray_file_name = file_names[i].split('-')[0] + '-gray_' + file_names[i].split('_')[1]
                    image_utilities.write_image(image_manipulation.adjust_color(img_gray, profile_data[0]),
                                                gray_file_name.split('.')[0],
                                                image_utilities.sample_path(gray_file_name),
                                                '_cc.' + settings.output_extension)
                    gray_file = os.path.join(image_utilities.sample_path(gray_file_name),
                                             gray_file_name.split('.')[0] + '_cc.' + settings.output_extension)

                    crop_settings = image_manipulation.crop_samples(gray_file_name.split('_')[0], ref_gray=True,
                                                                    crop_settings=image_part_crop_settings)
                    if image_part_crop_settings is None and utilities.yes_no_prompt(
                            "Use similar crop for rest of sample images with no ref. gray found?"):
                        image_part_crop_settings = crop_settings

                    if not settings.save_gray_images:
                        os.remove(os.path.join(settings.main_directory, gray_file))  # Remove ref. gray image

                print()

            measurement_gray = image_utilities.read_crop(file_names[i].split('-')[0] + "-gray", True)
            if measurement_gray is None:
                utilities.print_color("Warning: No ref. gray correction used!", 'warning')
                print()
            else:
                # Read ref. gray values
                if int(file_names[i].split('.')[0].split('_')[1]) in measurement_gray:
                    measurement_gray = measurement_gray[int(file_names[i].split('.')[0].split('_')[1])]
                else:
                    # Read the earliest ref. gray values after measurement or latest ref. gray values if not numbered
                    latest_gray = list(measurement_gray.keys())[-1]
                    for j in range(len(measurement_gray)):
                        latest_gray = list(measurement_gray.keys())[j]
                        try:
                            if int(list(measurement_gray.keys())[j]) > int(file_names[i].split('.')[0].split('_')[1]):
                                break
                        except ValueError:
                            latest_gray = list(measurement_gray.keys())[-1]
                            break

                    measurement_gray = measurement_gray[latest_gray]
                    utilities.print_color("Warning: Ref. gray not found!" + " Using ref. gray from measurement '"
                                          + str(latest_gray) + "'.", 'warning')
                    print()
                ref_ciede = colour.delta_E(profile_data[1], measurement_gray, method='CIE 2000').round(2)
                if ref_ciede >= settings.gray_warning_limit:
                    utilities.print_color(f"Warning: Ref. gray dE is {ref_ciede}!", 'warning')
                else:
                    utilities.print_color(f"Ref. gray dE OK, {ref_ciede}.", 'green')
                print()

            # Write corrected image
            image_utilities.write_image(image_manipulation.adjust_color(img, profile_data[0],
                                                                        gray_refs=(profile_data[1], measurement_gray)),
                                        file_names[i].split('.')[0], path, '_cc.' + settings.output_extension)
            os.remove(os.path.join(settings.main_directory, 'Exported Images', file_names[i]))  # Remove original

            if sample_names.count(file_names[i].split('_')[0]) == 0:
                sample_names.append(file_names[i].split('_')[0])
            last_sample = file_names[i].split('_')[0]

            est_data[1] += 1
            if est_data[1] == 1:
                # Estimate total processing time based on first image
                utilities.print_estimate(est_data[0], est_data[1] / len(file_names))

        if len(sample_names) > 0:
            print()
            if utilities.yes_no_prompt("Crop corrected images?"):
                for sample_name in sample_names:
                    image_manipulation.crop_samples(sample_name)

        end_time = time.perf_counter()

    elif script_mode == 1:  # Rename carousel timelapse files
        sample_name = None
        sample_found = False
        while not sample_found:
            # Prompt for sample name
            sample_name = input("Name of series: ")
            if sample_name.strip() != '':
                for file_name in utilities.get_files("Exported Images", match_extension=settings.input_extension):
                    if file_name.split('_')[0] == sample_name.split('_')[0]:
                        sample_found = True
                        break
            if not sample_found:
                print("Invalid sample.")

        print()
        cells = None
        while cells is None:
            cells = utilities.int_prompt("Number of cells in timelapse: ", (0, 8))
        start_time = time.perf_counter()

        cell_i = 1
        measurement_i = 1   # Start indexing from 1 to match Nova output
        for file_name in utilities.get_files("Exported Images", match_extension=settings.input_extension):
            if file_name.split('_')[0] == sample_name.split('_')[0]:
                new_file_name = (file_name.split('-')[0] + '-' + str(cell_i) + '_' + str(measurement_i)
                                 + '.' + file_name.split('.')[1])
                os.rename(os.path.join(settings.main_directory, rf"Exported Images\{file_name}"),
                          os.path.join(settings.main_directory, rf"Exported Images\{new_file_name}"))
                cell_i += 1
                if cell_i > cells:
                    cell_i = 1
                    measurement_i += 1

        end_time = time.perf_counter()

    elif script_mode == 2:  # Crop images
        while True:
            # Prompt for sample name
            sample_name = input("Name of series or sample: ")
            if len(sample_name.split('_')[0].split('-')) > 1:
                path = rf"Corrected Images\{sample_name.split('_')[0].split('-')[0]}\{sample_name}"
            else:
                path = rf"Corrected Images\{sample_name}"
            if sample_name.strip() != '' and os.path.exists(os.path.join(settings.main_directory, path)):
                break
            else:
                print("Invalid sample.")
        print()

        adjusting = utilities.yes_no_prompt("Adjust crop?")  # If not, images with existing crop data will be skipped

        start_time = time.perf_counter()
        image_manipulation.crop_samples(sample_name, adjusting)  # Crop sample image(s)

        end_time = time.perf_counter()

    elif script_mode == 3:  # Measure series color
        measure_mode = utilities.int_prompt("Measuring mode (0: Measure area, 1: Measure along line"
                                            ", ENTER: Main menu): ", (0, 1))
        if measure_mode is None:
            main()  # Return to main menu
            return

        while True:
            # Prompt for sample name
            img_name = input("Name of sample: ")
            if len(img_name.split('_')[0].split('-')) > 1:
                path = rf"Corrected Images\{img_name.split('_')[0].split('-')[0]}\{img_name}"
            else:
                path = rf"Corrected Images\{img_name.split('_')[0]}"
            if img_name.strip() != '' and os.path.exists(os.path.join(settings.main_directory, path + r"\Cropped")):
                path = path + r"\Cropped"
                break
            else:
                print("Invalid sample.")

        while True:
            # Prompt for image to use as dE reference
            ref_name = input("Reference image name (leave empty for first in series): ")
            if ref_name == '':
                ref_name = utilities.get_files(path, match_extension=settings.output_extension)[0].split('.')[0]
                print("Reference image:", ref_name)

            if os.path.exists(os.path.join(settings.main_directory, path, ref_name + '.' + settings.output_extension)):
                break
            else:
                print("Invalid image.")

        while True:
            # Prompt for name of measurement
            measurement_name = input("Color measurement name: ")
            if measurement_name.strip() != '' and measurement_name == measurement_name.strip():
                break
            else:
                print("Invalid name. Don't include spaces.")

        start_time = time.perf_counter()

        image_utilities.measure_series(path, ref_name, measure_mode, measurement_name)  # Run measurement process

        end_time = time.perf_counter()

    elif script_mode == 4:  # Calibration
        calib_mode = utilities.int_prompt("Measuring mode (0: Create calibration profile, 1: Measure image uniformity, 2: Check the need for recalibration"
                                          ", ENTER: Main menu): ", (0, 2))
        if calib_mode is None:
            main()  # Return to main menu
            return

        if calib_mode == 0:  # Create calibration profile
            while True:
                # Prompt for color target type
                ref_name = settings.reference_types[0]
                ref_data = None
                if len(settings.reference_types) > 1:
                    ref_name = input("Reference type (leave empty for default): ")
                    if ref_name.strip() == '':
                        # Set default target
                        ref_name = 'it87'
                    ref_data = image_utilities.read_reference(ref_name)
                if ref_data is not None:
                    break
                else:
                    print("Invalid reference.")

            calib_files = []
            while True:
                # Prompt for focus height of calibration image
                focus_height = input("Height - focus height (mm): ")
                if focus_height is not None:
                    focus_height = focus_height.strip()

                for file_name in utilities.get_files(r'Calibration\Calibration Images',
                                                     match_extension=settings.input_extension):
                    print(file_name)
                    if file_name.split('_')[0] == 'calib-' + ref_name and file_name.split('_')[1] == focus_height:
                        calib_files.append(file_name)
                    elif file_name == 'calib-' + ref_name + '_' + focus_height + '.' + settings.input_extension:
                        calib_files = [file_name]
                        break

                if len(calib_files) == 0:
                    print("Calibration image not found.")
                else:
                    break

            if settings.prompt_margin_utilization and not utilities.yes_no_prompt("Use safety margins?"):
                use_margins = False
            else:
                use_margins = True

            start_time = time.perf_counter()

            # Get reference gray image
            gray_img = image_utilities.read_image('calib-gray_' + focus_height + '.' + settings.input_extension,
                                                  r'Calibration\Calibration Images')

            if gray_img is None:
                print()
                utilities.print_color("Warning: Ref. gray not found!", 'warning')
                print()
            else:
                if use_margins:
                    gray_img = image_utilities.get_safe_area(gray_img)  # Crop to safety margins
                ref_crop = image_manipulation.match_crop(gray_img, 0)  # Prompt for crop
                lt_corner = image_utilities.cvt_point(ref_crop[1][0], -1, gray_img[0].shape)  # Upper left corner

                # Rotate and crop as selected
                gray_img = image_utilities.get_roi(image_manipulation.rotate_image(gray_img, ref_crop[0]), 1,
                                                   in_roi=(lt_corner[0], lt_corner[1],
                                                           abs(ref_crop[1][1][0] - ref_crop[1][0][0]),
                                                           abs(ref_crop[1][1][1] - ref_crop[1][0][1])))[0]

            calib_image_data = image_utilities.read_image(calib_files[0], r'Calibration\Calibration Images')[1]
            target_template = image_utilities.read_image(ref_name + '.jpg', 'Reference Values', absolute_path=True)
            target_c = (np.zeros((target_template[0].shape[0], target_template[0].shape[1], 3)), calib_image_data)

            img = None
            for calib_file in calib_files:
                img = image_utilities.read_image(calib_file, r'Calibration\Calibration Images')
                if use_margins:
                    img = image_utilities.get_safe_area(img)  # Crop to safety margins

                # Crop color target
                img_overlay, offset = image_manipulation.crop_target(img, target_template)
                cv.imshow('Overlay image', image_manipulation.convert_color(img_overlay,'show')[0])  # Show overlay image
                cv.waitKey(0)
                lu_corner_crop = (-min(0, offset[0]), -min(0, offset[1]))
                offset = np.sum([offset, lu_corner_crop], axis=0)
                img_overlay = (img_overlay[0][lu_corner_crop[1]:][lu_corner_crop[0]:], img_overlay[1])
                rd_corner_crop = (max(0, (offset[0] + img_overlay[0].shape[1]) - target_c[0].shape[1]),
                                  max(0, (offset[1] + img_overlay[0].shape[0]) - target_c[0].shape[0]))
                cv.imshow('Overlay image', image_manipulation.convert_color(img_overlay,'show')[0])  # Show overlay image
                cv.waitKey(0)
                img_overlay = (img_overlay[0][:-(rd_corner_crop[1] + 1), :-(rd_corner_crop[0] + 1)], img_overlay[1])

                cv.imshow('Overlay image', image_manipulation.convert_color(img_overlay,'show')[0])  # Show overlay image
                cv.waitKey(0)
                layers = np.stack((target_c[0][offset[1]:offset[1] + img_overlay[0].shape[0],
                                   offset[0]:offset[0] + img_overlay[0].shape[1]], img_overlay[0]), axis=-1)

                target_c[0][offset[1]:offset[1] + img_overlay[0].shape[0],
                            offset[0]:offset[0] + img_overlay[0].shape[1]] = np.ma.average(layers, axis=-1,
                                                                                           weights=layers.astype(bool)
                                                                                           ).filled(0)

                # Show intermediate state of target_c
                image_utilities.show_image("Intermediate target_c", target_c, save_to_disk=True)

            correction_lut = calibration.color_calibration(target_c, ref_data)[0]  #  3D LUT
            print('lut created') #testi
            target_a = image_manipulation.adjust_color(target_c, correction_lut)  # Correct color target

            # Show corrected color target and get accuracy
            fit_data, sample_data = calibration.color_calibration(target_a, ref_data, True)[1:3]

            # Get ref. gray LAB values
            gray_lab = (0, 0, 0)
            if gray_img is not None:
                gray_avg = image_manipulation.adjust_color((image_utilities.get_average_color(gray_img), gray_img[1]),
                                                           correction_lut)
                gray_lab = tuple(elem for elem in gray_avg[0])

                print(f"Ref. gray LAB ({settings.output_illuminant}):"
                      f"{image_manipulation.convert_color(gray_avg, 'LAB')}")
                print()

            print("Correction LUT:", correction_lut)

            end_time = time.perf_counter()
            key_pressed = image_utilities.wait_key()
            cv.destroyAllWindows()

            if key_pressed != 'escape':
                # Save 3D LUT
                image_utilities.write_profile(focus_height, correction_lut, gray_lab, img[1],
                                              fit_data, sample_data, ref_data[0][0])
            else:
                print("Discarding profile data.")

        elif calib_mode == 1:  # Measure image uniformity
            use_margins = utilities.yes_no_prompt("Use safety margins?")

            while True:
                # Prompt for file to measure
                file_name = input("Image file name: ")

                # Get path of file
                path = image_utilities.sample_path(file_name)
                if os.path.exists(os.path.join(settings.main_directory, path)):
                    for file in os.listdir(os.path.join(settings.main_directory, path)):
                        file = str(file)

                        if file_name == file.split('.')[0]:
                            file_name = file
                            break
                        elif len(file_name.split('_')) <= len(file.split('_')):
                            is_same = True
                            for i in range(len(file_name.split('_'))):
                                if file_name.split('_')[i] != file.split('_')[i]:
                                    is_same = False

                            if is_same:
                                file_name = file
                                break

                    img = image_utilities.read_image(file_name, path)
                    if img is not None:
                        break

                print("Invalid image.")

            print()

            start_time = time.perf_counter()

            if use_margins:
                img = image_utilities.get_safe_area(img)  # Crop to safety margins
            cropped_img = image_utilities.get_roi(img)[0]
            img_uniformity = calibration.image_uniformity(cropped_img)

            image_utilities.show_image("Image uniformity", image_manipulation.scale_image(img_uniformity[:2])[0],
                                       convert=False)

            end_time = time.perf_counter()
            key_pressed = image_utilities.wait_key()
            cv.destroyAllWindows()

            if key_pressed != 'escape':
                # Save uniformity image after closing window
                image_utilities.write_image(img_uniformity[:2], file_name.split('.')[0], r'Calibration\Image Uniformity',
                                            '_uniformity.jpg', True, convert=False)
                print("Uniformity image saved.")

            else:
                print("Discarding uniformity image.")

            # The error due to nonuniform lightning 
            while True:
                try:
                    max_ciede = float(input("Maximum color error allowed due to nonuniform lightning: "))
                    if isinstance(max_ciede, int) or isinstance(max_ciede, float):
                        break
                except ValueError:
                    print("Invalid input. Only use numbers")
                    continue
            # visualize and calculate the error
            img_error = calibration.nonuniformity_error(cropped_img, img_uniformity[2], max_ciede) 

        elif calib_mode ==2: # Check if recalibration is necessary

            while True:
                # Give the correction profile without the .cube extension
                focus_input = input("Name of the correction profile to be used? ") 
                if focus_input is not None:
                    focus_input = focus_input.strip()

                profile_data = image_utilities.read_profile(focus_input)
                if profile_data is not None:
                    break
                else:
                    print("Invalid correction profile.")
            print()

            while True:
                # Prompt for file to measure. The gray image to be used should be corrected with the calibration profile already,
                # that is left to be the user's responsibility
                file_name = input("Gray image file name: ")

                # Get path of file
                # remember that the gray image used in the correction is not automatically saved 
                # So, the easiest scenario is to take two gray pictures, one named like Severi instructed as "A-gray_B"
                # (for the color correction of the other image) and the other as something else.
                path = image_utilities.sample_path(file_name)
                if os.path.exists(os.path.join(settings.main_directory, path)):
                    for file in os.listdir(os.path.join(settings.main_directory, path)):
                        file = str(file)

                        if file_name == file.split('.')[0]:
                            file_name = file
                            break
                        elif len(file_name.split('_')) <= len(file.split('_')):
                            is_same = True
                            for i in range(len(file_name.split('_'))):
                                if file_name.split('_')[i] != file.split('_')[i]:
                                    is_same = False

                            if is_same:
                                file_name = file
                                break

                    img = image_utilities.read_image(file_name, path)
                    if img is not None:
                        break

                print("Invalid image.")

            print()

            crop = utilities.yes_no_prompt("Crop to safety margins?")
            if crop:
                cropped_img0 = image_utilities.get_safe_area(img)
                cropped_img = image_utilities.get_roi(cropped_img0)[0]
            else:
                cropped_img = image_utilities.get_roi(img)[0]

            print("Measuring image uniformity...")

            img_uniformity = calibration.image_uniformity(cropped_img)

            image_utilities.show_image("Image uniformity", image_manipulation.scale_image(img_uniformity[:2])[0],
                                       convert=False)

            key_pressed = image_utilities.wait_key()
            cv.destroyAllWindows()

            if key_pressed != 'escape':
                # Save uniformity image after closing window
                image_utilities.write_image(img_uniformity[:2], file_name.split('.')[0], r'Calibration\Image Uniformity',
                                            '_uniformity.' + settings.output_extension, True, convert=False)
                print("Uniformity image saved.")
            else:
                print("Discarding uniformity image.")

            # visualize and calculate the colro error sue to nonuniform lighting
            while True:
                try:
                    max_ciede = float(input("Maximum color error allowed due to nonuniform lightning: "))
                    if isinstance(max_ciede, int) or isinstance(max_ciede, float):
                        break
                except ValueError:
                    print("Invalid input. Only use numbers")
                    continue

            img_error = calibration.nonuniformity_error(cropped_img, img_uniformity[2], max_ciede) # img_a, avg_col

            print()
            print("Measuring color difference between grays...")
            print()

            # calculate color difference between calibration gray and measured gray
            ref_gray = profile_data[1]
            avg_col = img_uniformity[2]
            delta_E = colour.delta_E(ref_gray, avg_col, method='CIE 2000') 

            print("The reference gray CIELAB: (" + str(ref_gray[0]) + ", " + str(ref_gray[1]) + ", " + str(ref_gray[2]) + ")")
            print("The measured gray CIELAB: (" + str(avg_col[0]) + ", " + str(avg_col[1]) + ", " + str(avg_col[2]) + ")")
            print()
            
            if delta_E <= 1.5:
                print("The color difference between the gray images is okay.")
                print("CIEDE2000 =  " + str(delta_E) + ".")
                print("The correction profile '" + focus_input + "' can be used, no need for recalibration.")
            else:
                print("Warning! The color difference between the gray images is significant!")
                print("CIEDE2000 =  " + str(delta_E) + ".")
                print("The correction profile '" + focus_input + "' should not be used! Recalibration is recommended!")

    elif script_mode == 5: # Detect and measure ColorChecker sample colors from picture
        # Measured values of a plain ColorChecker (these are used for sample color error measurements)
        #ref_lab0 =
        
        #ref_lab = np.array(ref_lab0)
  
        ref_data = image_utilities.read_reference(settings.reference_types[0]) # Reference ColorChecker data

        while True:
            # Prompt for sample name (seriesname-samplename_measurement)
            img_name = input("Name of sample: ")
            if len(img_name.split('_')[0].split('-')) > 1:
                path = rf"Corrected Images\{img_name.split('_')[0].split('-')[0]}\{img_name.split('_')[0]}"
                in_img = image_utilities.read_image(img_name + '_cc.' + settings.output_extension, path)
            else:
                path = rf"Corrected Images\{img_name.split('_')[0]}"
                in_img = image_utilities.read_image(img_name + '_cc.' + settings.output_extension, path)
            if img_name.strip() != '' and os.path.exists(os.path.join(settings.main_directory, path + rf"\Cropped")):
                path = path + rf"\Cropped"
                in_img = image_utilities.read_image(img_name + '_cc_cropped.' + settings.output_extension, path)
            if in_img is not None:
                break
            else:
                print("Invalid sample.")

        print("Reading values from " + img_name +"...")
        print()
        
        # The function expects color space values as (0-1) linear RGB, so first we need to convert LAB D50 to sRGB
        RGB_values = image_manipulation.convert_color(in_img, 'RGB') # Non-linear sRGB
        linear_RGB = np.clip(color_model[0].cctf_decoding(RGB_values[0]), 0, 1) # Linearization
 
        # Detecting the colorchecker from the image, choosing the region from where to calculate the average RGB for each sample
        detected = colour_checker_detection.detect_colour_checkers_segmentation(linear_RGB, samples=100, additional_data=True)
        if not detected:
            print("No ColorChecker detected.") 
            main()
        rgb = detected[0].swatch_colours  # Avg. RGB of each sample

        # Conversion to LAB D50
        # RGB D65 -> XYZ D65
        xyz_d65 = colour.RGB_to_XYZ(rgb, colourspace=color_model[0], 
                                    illuminant=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
        # XYZ D65 -> XYZ D50 (using Bradford model)
        xyz_vals = colour.adaptation.chromatic_adaptation_VonKries(xyz_d65, colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']),
                                                          colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']), transform='Bradford')         
        # XYZ D50 (0 - 1) -> LAB D50 (0 - 100, -100 - 100, -100 - 100)
        lab_vals = colour.XYZ_to_Lab(xyz_vals, illuminant=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])

        # Conversion to right shape for deltaE calculation
        ref_lab = ref_data[1].transpose(1,0,2).reshape(-1,3) 
        delta_E = colour.delta_E(ref_lab, lab_vals, method='CIE 2000')
        min_dE, max_dE = np.min(delta_E), np.max(delta_E)
        avg_dE = np.average(delta_E)
        avg_dL = np.mean(np.abs(lab_vals[:,0]-ref_lab[:,0]))
        avg_da = np.mean(np.abs(lab_vals[:,1]-ref_lab[:,1]))
        avg_db = np.mean(np.abs(lab_vals[:,2]-ref_lab[:,2]))

        # Writing a CSV file for information, will be saved in path 
        headers = ('Patch', 'L', 'a', 'b', 'dE', 'min. dE', 'max. dE', 'Avg. dE', 'Avg dL','Avg. da', 'Avg. db') # headers for CSV file       
        rows = []
        for i in range(24):
            rows.append([i+1, lab_vals[i][0], lab_vals[i][1], lab_vals[i][2], delta_E[i], '', '', '', '', '', '']) # Going through each color sample
        rows.append(['', '', '', '','', min_dE, max_dE, avg_dE, avg_dL, avg_da, avg_db])
        out_data = [headers] + rows
        print("Writing a CSV file...")
        utilities.write_csv(out_data, img_name + 'ref', path)
        print()
        print("CSV file written!")

        # Draw red squares to border the area that was used to calculate Avg. RGB-values. Present to the user
        # Using the aligned ColorChecker from the detect_color_checkers_segmentation, since it is scaled correctly with respect to swatch_masks

        # RGB D65 -> XYZ D65
        xyz_d652 = colour.RGB_to_XYZ(detected[0].colour_checker, colourspace=color_model[0],
                                      illuminant=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
        # XYZ D65 -> XYZ D50
        xyz_vals2 = colour.adaptation.chromatic_adaptation_VonKries(xyz_d652, colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']),
                                                          colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']), transform='Bradford')         
        # XYZ D50 (0 - 1) -> LAB D50 (0 - 100, -100 - 100, -100 - 100)
        lab_vals2 = colour.XYZ_to_Lab(xyz_vals2, illuminant=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])
        out_img = image_manipulation.convert_color((lab_vals2, in_img[1]), 'show')

        rec_corners = detected[0].swatch_masks   # Corners of the area from which the avg. RGB is calculated from
        for i in range(24):
            top_left = (rec_corners[i][2], rec_corners[i][0]) # Rec_corners is [y1,y2,x1,x2] in corner coordinates
            bottom_right = (rec_corners[i][3], rec_corners[i][1])
            out_img = (cv.rectangle(out_img[0], top_left, bottom_right, (0, 0, 255), 2), out_img[1]) # Drawing the rectangles

        image_utilities.show_image("The area used for avg. RGB calculation.", image_manipulation.scale_image(out_img)[0], False) 
        cv.waitKey(0)
        cv.destroyAllWindows   

    elif script_mode == 6: # Measure line/area values from picture

        measure_mode = utilities.int_prompt("Measuring mode (0: Measure 2D line data, 1: Measure area CIELAb averages, ENTER: Main menu): ", (0, 1))
        if measure_mode is None:
            main()  # Return to main menu
            return
        
        while True:
                # Prompt for file to measure
                file_name = input("Image file name: ")

                # Get path of file
                path = image_utilities.sample_path(file_name)
                if os.path.exists(os.path.join(settings.main_directory, path)):
                    for file in os.listdir(os.path.join(settings.main_directory, path)):
                        file = str(file)

                        if file_name == file.split('.')[0]:
                            file_name = file
                            break
                        elif len(file_name.split('_')) <= len(file.split('_')):
                            is_same = True
                            for i in range(len(file_name.split('_'))):
                                if file_name.split('_')[i] != file.split('_')[i]:
                                    is_same = False

                            if is_same:
                                file_name = file
                                break

                    img = image_utilities.read_image(file_name, path)
                    if img is not None:
                        break

                print("Invalid image.")

        while True:
                # Prompt for name of measurement
                measurement_name = input("Name of measurement (leave empty for default): ")
                if measurement_name.strip() == '':
                    # Set default name
                    measurement_name = file_name.split('.')[0]
                measurement_name = str(measurement_name)
                if measurement_name is not None:
                    break

        if measure_mode == 0:

            while True:
                # Prompt for image pixel scale
                px_scale = input("Pixel scale (µm/px): ")
                try:
                    px_scale = float(px_scale.strip())
                except ValueError:
                    pass
                if px_scale is not None:
                    break
                else:
                    print("Invalid scale, only input numbers using '.' as decimal separator.")

            print()
            print("Cropping the image...")

            img = image_utilities.get_safe_area(img)  # Crop to safety margins
            cropped_img = image_utilities.get_roi(img)[0]
            arrow_number = 0
            out_data = []

            # Create the figures for plotting
            # CIEDE2000
            figure_CIEDE, axe_CIEDE = plt.subplots()
            axe_CIEDE.set_xlabel('Distance from arrow start (mm)')
            axe_CIEDE.set_ylabel('CIEDE2000')
            # L*
            figure_L, axe_L = plt.subplots()
            axe_L.set_xlabel('Distance from arrow start (mm)')
            axe_L.set_ylabel('L*')
            # a*
            figure_a, axe_a = plt.subplots()
            axe_a.set_xlabel('Distance from arrow start (mm)')
            axe_a.set_ylabel('a*')
            # b*
            figure_b, axe_b = plt.subplots()
            axe_b.set_xlabel('Distance from arrow start (mm)')
            axe_b.set_ylabel('b*')
            axes = (axe_CIEDE, axe_L, axe_a, axe_b)

            titles= ["CIEDE2000", "L*", "a*", "b*"]
            for ax, title in zip(axes, titles):
                ax.set_title(title)

            # Create copies of the image for selecting the line and visualization
            og_img = (cropped_img[0].copy(), copy.deepcopy(cropped_img[1]))
            annotated_img = (cropped_img[0].copy(), copy.deepcopy(cropped_img[1]))
            
            print("Measuring modes:")
            print("0 = Average color calculated for each arrow separately and used as reference for CIEDE2000.")
            print("1 = The starting point of each arrow will be the reference color.")
            print("2 = Average color of the whole input image used as reference for all arrows.")
            measure_mode = utilities.int_prompt("Select measuring mode: ", (0,2))

            print("The data points will be averages of multiple pixels.")
            print("Bigger number = fewer data points, clearer visualization. Smaller number = more data points, unclear visualization.")
            avg_window = utilities.int_prompt("Average of how many pixels (1-20)? ", (1,20))

            # Ask the user if they want to draw another line after each one
            while True:
                select_line = utilities.yes_no_prompt("Select line for data?")
                if select_line == True:
                    arrow_number += 1
                    data = image_utilities.measure_2D(og_img, file_name, arrow_number, px_scale, measure_mode, avg_window)
                    image_utilities.add_plot(axes, data[0], data[1], data[4], arrow_number)
                    if not out_data:
                        out_data.append(['file and number', 'x', 'CIEDE2000', 'L*', 'a*', 'b*'])

                    for i in range(len(data[0][1][1])):
                        out_data.append([data[0][1][0], data[0][1][1][i], data[0][1][2][i],
                                        data[0][1][3][i], data[0][1][4][i], data[0][1][5][i]])

                    # Visualization, update figures 
                    visual_img = image_utilities.draw_arrow(annotated_img, data[2], data[3], arrow_number)
                    image_utilities.show_image("Arrows drawn so far", visual_img)
                    if arrow_number == 1:
                        plt.show(block=False)
                        plt.pause(2)
                    else:
                        for ax in axes:
                            ax.figure.canvas.draw()
                        plt.pause(2)
                if select_line == False:
                    if arrow_number == 0:
                        print("No arrows drawn. No data saved.")
                        break
                    else:
                        # Save the data and the image with arrows
                        directory = os.path.join(settings.main_directory, image_utilities.sample_path(file_name), "2D")
                        os.makedirs(directory, exist_ok=True)
                        print("Saving the data as an CSV file...")
                        utilities.write_csv(out_data, measurement_name + "2D_data", directory)
                        print("Writing the arrow image...")
                        image_utilities.write_image(annotated_img,
                            measurement_name + "_arrows", directory, '.jpg', True, True)
                        print("Arrow image written!")
                        figure_CIEDE.savefig(os.path.join(directory, f"CIEDE2000.{measurement_name}.png"))
                        figure_L.savefig(os.path.join(directory, f"L.{measurement_name}.png"))
                        figure_a.savefig(os.path.join(directory, f"a.{measurement_name}.png"))
                        figure_b.savefig(os.path.join(directory, f"b.{measurement_name}.png"))
                        plt.close(figure_CIEDE)
                        plt.close(figure_L)
                        plt.close(figure_a)
                        plt.close(figure_b)
                        break

        if measure_mode == 1:
            # Measure area values from image and draw heatmap(?) of values    

            print()
            print("Cropping the image...")

            crop = utilities.yes_no_prompt("Crop to safety margins?")
            if crop:
                img = image_utilities.get_safe_area(img)  # Crop to safety margins
                
            cropped_img = image_utilities.get_roi(img)[0]
            rec_number = 0
            out_data = []
            og_img = (cropped_img[0].copy(), copy.deepcopy(cropped_img[1]))
            annotated_img = (cropped_img[0].copy(), copy.deepcopy(cropped_img[1]))
            colors = []
            draw_img = (np.zeros((settings.max_window[1], settings.max_window[0], 3), dtype=np.float32), cropped_img[1])

            # Ask the user if they want to draw another rectangle after each one
            while True:
                select_rec = utilities.yes_no_prompt("Select rectangle for data?")
                if select_rec == True:
                    rec_number += 1
                    data = image_utilities.measure_3D(og_img, file_name, rec_number)
                    if rec_number == 1:
                        out_data.append(['file and number', 'L*', 'a*', 'b*'])
                    out_data.append([data[1][1][0], data[1][1][1], data[1][1][2], data[1][1][3]])

                    # Visualization as a separate image
                    colors.append(data[3])

                    # Original picture with the rectangles
                    visual_img = image_utilities.draw_rec(annotated_img, data[0], data[2], rec_number)
                    cv.namedWindow("Rectangles drawn so far", cv.WINDOW_NORMAL)
                    cv.imshow("Rectangles drawn so far", image_manipulation.convert_color(visual_img,'show')[0])
                    cv.waitKey(500)
    
                    save_img = image_utilities.area_color_image(colors, draw_img) # lab image
                    cv.namedWindow("Visualization of measured colors.", cv.WINDOW_NORMAL)
                    cv.imshow("Visualization of measured colors.", image_manipulation.convert_color(save_img,'show')[0])
                    cv.resizeWindow("Visualization of measured colors.", save_img[0].shape[1], save_img[0].shape[0])
                    cv.waitKey(500)
                if select_rec == False:
                    if rec_number == 0:
                        print("No rectangles drawn. No data saved.")
                        break
                    else:
                        # Save the data and the image with arrows
                        directory = os.path.join(settings.main_directory, image_utilities.sample_path(file_name), "Area")
                        os.makedirs(directory, exist_ok=True)
                        print("Saving the data as an CSV file...")
                        utilities.write_csv(out_data, measurement_name + "area_data", directory)
                        print("Writing the images...")
                        image_utilities.write_image(annotated_img,
                            measurement_name + "_rectangles", directory, '.jpg', True, True)
                        image_utilities.write_image(save_img, measurement_name + "_visualization", directory, '.jpg', True, True)
                        print("Images written!")
                        break
            cv.destroyAllWindows()
            
    elif script_mode == 7:
        while True:
            # Prompt for file to measure
            file_name = input("Image file name: ")

            # Get path of file
            path = image_utilities.sample_path(file_name)
            if os.path.exists(os.path.join(settings.main_directory, path)):
                for file in os.listdir(os.path.join(settings.main_directory, path)):
                    file = str(file)

                    if file_name == file.split('.')[0]:
                        file_name = file
                        break
                    elif len(file_name.split('_')) <= len(file.split('_')):
                        is_same = True
                        for i in range(len(file_name.split('_'))):
                            if file_name.split('_')[i] != file.split('_')[i]:
                                is_same = False

                        if is_same:
                            file_name = file
                            break

                img = image_utilities.read_image(file_name, path)
                if img is not None:
                    break

            print("Invalid image.")

        calib_image_data = img[1]
        target_template = image_utilities.read_image('it87.jpg', 'Reference Values', absolute_path=True)
        target_c = (np.zeros((target_template[0].shape[0], target_template[0].shape[1], 3)), calib_image_data)

        # Crop color target
        img_overlay, offset = image_manipulation.crop_target(img, target_template)
        lu_corner_crop = (-min(0, offset[0]), -min(0, offset[1]))
        offset = np.sum([offset, lu_corner_crop], axis=0)
        img_overlay = (img_overlay[0][lu_corner_crop[1]:][lu_corner_crop[0]:], img_overlay[1])
        rd_corner_crop = (max(0, (offset[0] + img_overlay[0].shape[1]) - target_c[0].shape[1]),
                            max(0, (offset[1] + img_overlay[0].shape[0]) - target_c[0].shape[0]))
        img_overlay = (img_overlay[0][:-(rd_corner_crop[1] + 1), :-(rd_corner_crop[0] + 1)], img_overlay[1])

        layers = np.stack((target_c[0][offset[1]:offset[1] + img_overlay[0].shape[0],
                            offset[0]:offset[0] + img_overlay[0].shape[1]], img_overlay[0]), axis=-1)

        target_c[0][offset[1]:offset[1] + img_overlay[0].shape[0],
                    offset[0]:offset[0] + img_overlay[0].shape[1]] = np.ma.average(layers, axis=-1,
                                                                                    weights=layers.astype(bool)
                                                                                    ).filled(0)

        # Show corrected color target and get accuracy
        ref_data = image_utilities.read_reference('it87')
        fit_data, sample_data, average = calibration.compare_it87(target_c, ref_data, True)

        key_pressed = image_utilities.wait_key()
        cv.destroyAllWindows()

        if key_pressed != 'escape':
            # Save 3D LUT
            channels, width, height = sample_data.shape
            header = ['SampleIndex'] + ['CIEDE2000']+ ['L difference'] + ['a* difference'] + ['b* difference']
            averages = ['Averages'] + [str(average[0])] + [str(average[1])] + [str(average[2])] + [str(average[3])]

            # Flatten sample_data into rows: each row = one sample (x,y) with all channels
            rows = []
            for i in range(width):
                for j in range(height):
                    row = [f'{i}_{j}']  # sample index label
                    for c in range(channels):
                        row.append(sample_data[c, i, j])
                    rows.append(row)

            # Combine header + rows into one 2D list
            data_to_write = [header] + rows + [averages] 
            utilities.write_csv(data_to_write, file_name, path)
        else:
            print("Discarding data.")


    else:
        print("Invalid script mode.")
        return

    print()
    print(f"Finished in {round(end_time - start_time, 2)} s.")

    main()  # Restart program at main menu


if __name__ == '__main__':
    main()
