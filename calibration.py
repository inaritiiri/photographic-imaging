import settings
import utilities
import image_utilities
import main_script
import image_manipulation

import time
import numpy as np
import cv2 as cv
import colour
from matplotlib import pyplot as plt
from matplotlib import colors as col
import matplotlib.cm as cm
import os
plt.rcParams["figure.dpi"] = 200


def color_calibration(in_img, ref_data, show_img=False):
    """Create 3D LUT from input image and reference data"""
    rec = image_manipulation.convert_color(in_img, 'show')

    # Base for color plot
    colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
        main_script.color_model[settings.output_color_space], show=False)

    # Calculate sample average colors
    fit_data = np.zeros(6)
    sample_data = np.zeros((4, ref_data[0][0][0], ref_data[0][0][1]))
    lab_vals = [[[], []], [[], []], [[], []]]
    lab_range = [[100, 100, 100], [0, -100, -100]]
    labels = ('L*', 'a*', 'b*', 'CIELAB')
    for l_y in range(ref_data[0][0][1]):
        for l_x in range(ref_data[0][0][0]):
                       # Find top left corner of sample
            top_left = (round((ref_data[0][1][0][0] + ref_data[0][1][1][0] * l_x)
                              * (in_img[0].shape[1] / ref_data[0][0][0])),
                        round((ref_data[0][1][0][1] + ref_data[0][1][1][1] * l_y)
                              * (in_img[0].shape[0] / ref_data[0][0][1])))

            # Find bottom right corner of sample
            bottom_right = (round(top_left[0] + ref_data[0][1][2][0] * (in_img[0].shape[1] / ref_data[0][0][0])),
                            round(top_left[1] + ref_data[0][1][2][1] * (in_img[0].shape[0] / ref_data[0][0][1])))
            rec = (cv.rectangle(rec[0], top_left, bottom_right, (0, 0, 255), 2), rec[1])        # Draw rectangle
            grid_sample = in_img[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]   # Crop sample
            avg_col = image_utilities.get_average_color((grid_sample, in_img[1]))               # Sample average color
            sample_data[0, l_x, l_y] = colour.delta_E(avg_col, ref_data[1][l_x][l_y], method='CIE 2000')
            print(f'({l_x + 1}, {l_y + 1})', f"Average color - LAB ({settings.output_illuminant}):",
                  image_manipulation.convert_color((avg_col, in_img[1]), 'LAB')[0].round(3))
            print("Reference CIEDE2000:", "{:.3f}".format(sample_data[0, l_x, l_y]))

            ref_xy = image_manipulation.convert_color((ref_data[1][l_x][l_y], in_img[1]), 'xy')[0]  # Reference xy
            avg_xy = image_manipulation.convert_color((avg_col, in_img[1]), 'xy')[0]                # Sample xy

            # Plot line from reference point
            plt.plot((ref_xy[0], avg_xy[0]), (ref_xy[1], avg_xy[1]), 'k')
            plt.plot(ref_xy[0], ref_xy[1], '.w')

            # Plot separate points
            # plt.plot(ref_xy[0], ref_xy[1], '.k')
            # plt.plot(avg_xy[0], avg_xy[1], '.w')

            for i in range(3):
                # Append sample color
                lab_vals[i][0].append(avg_col[i])
                lab_vals[i][1].append(ref_data[1][l_x][l_y][i])

                # Sample deviation from reference data
                sample_data[i + 1, l_x, l_y] = np.subtract(lab_vals[i][0][-1], lab_vals[i][1][-1])

                # Save LAB ranges
                lab_range[0][i] = min(lab_range[0][i], lab_vals[i][0][-1])
                lab_range[1][i] = max(lab_range[1][i], lab_vals[i][0][-1])

    for i in range(3):
        # Calculate average differences
        fit_data[i + 3] = np.average(np.average(np.abs(sample_data[i + 1])))

        # Check if LAB are out of domain (without considering color ranges for specific luminance)
        if lab_range[0][i] < settings.input_lab_domain[0][i] or lab_range[1][i] > settings.input_lab_domain[1][i]:
            utilities.print_color("Input " + labels[i] + " out of domain: (" + str(lab_range[0][i]) + ', ' +
                                  str(lab_range[1][i]) + ')!', 'error')

    fit_data[0] = np.average(sample_data[0])            # Average CIEDE2000
    fit_data[1] = sample_data[0].min(initial=1000.0)    # Min. CIEDE2000
    fit_data[2] = sample_data[0].max(initial=0.0)       # Max. CIEDE2000

    print()
    print("Avg. luminance difference:", fit_data[3].round(3))
    print("Avg. a difference:", fit_data[4].round(3))
    print("Avg. b difference:", fit_data[5].round(3))

    print("Ref. average CIEDE2000:", "{:.3f}".format(fit_data[0]))
    print("Ref. CIEDE2000 range: [", "{:.3f}".format(fit_data[1]), "-",
          "{:.3f}".format(fit_data[2]), "]")

    # Show color plot
    plt.title(f'CIEDE2000: {fit_data[0].round(3)} [{fit_data[2].round(3)}]')
    plt.show()

    correction_lut = None

    if show_img:
        # Show corrected color target
        image_utilities.show_image("Corrected target (Press Esc to discard, or any other key to save)",
                                   image_manipulation.scale_image(rec)[0], False)
    else:
        s_time = time.perf_counter()
        # Create 3D LUT
        correction_lut = utilities.create_lut((lab_vals[0][0], lab_vals[1][0], lab_vals[2][0]),
                                              (lab_vals[0][1], lab_vals[1][1], lab_vals[2][1]),
                                              settings.input_lab_domain)

        print('LUT creation time:', time.perf_counter() - s_time, '(s)')

        # Plot calibration result
        plot_domain = ((0, -60, -60), (100, 60, 60))    # Plot input range
        edges = []
        step_size = 2   # Plot step size

        # Create grid for plotting color values
        for o in range(3):
            edges.append(np.linspace(plot_domain[0][o], plot_domain[1][o],
                                     round((plot_domain[1][o] - plot_domain[0][o]) / step_size)))

        lab_i = np.array(np.meshgrid(edges[0], edges[1], edges[2]))
        lab_i = np.array((lab_i[0].flatten(), lab_i[1].flatten(), lab_i[2].flatten()))
        res_i = correction_lut.apply(lab_i.T)   # Calculate color values at grid points

        # Plot 3D LUT cubes
        for i in range(4):
            fig = plt.figure()
            plt.title(labels[i])
            ax = fig.add_subplot(111, projection='3d')
            if i < 3:
                # Plot single value cubes
                fig.colorbar(ax.scatter(lab_i[1], lab_i[2], lab_i[0], c=res_i.take(i, axis=1), cmap=plt.viridis()),
                             shrink=0.9, pad=0.1)
            else:
                # Plot cube with mapped colors
                ax.scatter(lab_i[1], lab_i[2], lab_i[0],
                           c=image_manipulation.convert_color((res_i, in_img[1]), 'RGB')[0])
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b + 0.05 * h, w, h * 0.9])
            ax.set_xlabel('a*')
            ax.set_ylabel('b*')
            ax.set_zlabel('L*')
            ax.zaxis.labelpad = 0
            plt.show()

    print()
    return correction_lut, fit_data, sample_data

def image_uniformity(in_img):
    """Calculate image color uniformity"""
    image_utilities.show_image("Selected area", image_manipulation.scale_image(in_img)[0])

    # Calculate average color
    avg_col = image_utilities.get_average_color(in_img)
    print(f"Average color - LAB ({settings.output_illuminant}):",
          image_manipulation.convert_color((avg_col, in_img[1]), 'LAB')[0].round(3))

    # Downscale image for calculations
    img_u = image_manipulation.scale_image(in_img, settings.uniformity_scale)[0]
    # Calculate CIEDE2000 values for scaled image
    deltas = colour.delta_E(avg_col, img_u[0], method='CIE 2000')

    print("Average CIEDE2000:", "{:.3f}".format(np.average(deltas)))
    print("CIEDE2000 range: [", "{:.3f}".format(deltas.min(initial=1000.0)), "-",
          "{:.3f}".format(deltas.max(initial=0.0)), "]")

    px_range = [0, 0]
    img_d = np.zeros((img_u[0].shape[0], img_u[0].shape[1], 3))
    for y in range(img_u[0].shape[0]):
        for x in range(img_u[0].shape[1]):
            # Map CIEDE2000 to pixel value
            px_brightness = round(np.interp(deltas[y, x], (0, settings.ciede_max),
                                            (0, main_script.max_val[0])))
            if img_u[0][y, x][0] > avg_col[0]:
                img_d[y, x] = (0, 0, px_brightness)  # Red: Brighter
                px_range[1] = max((px_range[1], px_brightness))
            else:
                img_d[y, x] = (px_brightness, 0, 0)  # Blue: Darker
                px_range[0] = min((px_range[0], -px_brightness))
    img_d = img_d.astype(main_script.bit_type[0])   # Save as 8bit
    print("Image range:", px_range)
    return img_d, ((0, 0), {}), avg_col

def illumination_color_error_areas(in_img, rows = 8, cols = 12):
    """Calculates the color error and average color of different areas of image"""
    
    img = in_img[0]
    h, w = img.shape[:2]
    small_area_height = h // rows # area height
    small_area_width = w // cols # area width

    area_average_colors = []
    area_pixels = []

    for i in range(rows):
        areas = []
        for j in range(cols):
            y1, y2 = i * small_area_height, (i+1) * small_area_height # y-coordinates
            x1, x2 = j * small_area_width, (j+1) * small_area_width # x-coordinates
            area_pixels.append(img[y1:y2, x1:x2]) # storing the size of the areas
            areas.append(image_utilities.get_average_color((img[y1:y2, x1:x2], in_img[1]))) # average colors of the areas
        area_average_colors.append(areas)

    return area_average_colors, area_pixels, rows, cols

def nonuniformity_error(in_img, avg_col, max_ciede):
    # Calculates the error due to nonuniform lightning and presnts it to the user

    data = illumination_color_error_areas(in_img) # area average colors, area pixels, rows, cols
    plotting_deltas = colour.delta_E(avg_col, data[0]) # color differences in different parts of the image
    heatmap = np.zeros((data[2], data[3], 3), dtype=np.uint8) * 255

    for y in range(data[2]):
        for x in range(data[3]):
            # scaling the color difference of each block to get the right color
            px_brightness = round(np.interp(plotting_deltas[y, x], (0, max_ciede), 
                                            (0, main_script.max_val[0])))

            if data[0][y][x][0] > avg_col[0]: # red = brighter than average
                r = 255+px_brightness*((1/2.55)-1)
                g = 255-px_brightness
                b = 255-px_brightness
            else:
                r = 255-px_brightness # blue = darker than average
                g = 255-px_brightness
                b = 255+px_brightness*((1/2.55)-1)
            heatmap[y][x] = (r, g, b) # assigning the color to the heatmap for visualization

    heatmap_float = heatmap / 255.0 #scaling the values to 0.0-1.0

    #fig, ax = plt.subplots()
    #ax.imshow(heatmap_float, aspect="equal")
    fig = plt.gcf()  # get current figure or create one with plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # smaller area for heatmap (left, bottom, width, height)
    ax.imshow(heatmap_float, aspect="equal")
    for i in range(data[2]):
        for j in range(data[3]):
            # error values bigger than allowed will be drawn with light brown
            if plotting_deltas[i][j] >= max_ciede:
                font_color = (193/255.0, 154/255.0, 107/255.0)
            else:
                font_color = (0,0,0)
            # adding the ciede2000 of each area to the figure
            ax.text(j, i, f"{plotting_deltas[i][j]:.3f}", ha = 'center', va = 'center', color = font_color, fontsize=8)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title(f"CIEDE2000 in different areas. Maximum allowed difference = {max_ciede}", fontsize =10)

    scale = (1 / 2.55) - 1
    num_colors = 256
    blue_colors = [( (255 - 255*i)/255, (255 - 255*i)/255, min((255 + 255*i*scale)/255, 1.0)) for i in np.linspace(0, 1, num_colors)]
    red_colors = [( min((255 + 255*i*scale)/255,1.0), (255 - 255*i)/255, (255 - 255*i)/255 ) for i in np.linspace(0, 1, num_colors)]

    blue_cmap = col.LinearSegmentedColormap.from_list("white_to_blue", blue_colors)
    red_cmap = col.LinearSegmentedColormap.from_list("white_to_red", red_colors)
    norm = col.Normalize(vmin=0, vmax=max_ciede)
    blue_sm = cm.ScalarMappable(cmap=blue_cmap, norm=norm)
    red_sm = cm.ScalarMappable(cmap=red_cmap, norm=norm)

    fig = plt.gcf()

    # Create axes for the two colorbars side by side (adjust [x,y,width,height] as needed)
    ax_blue = fig.add_axes([0.8, 0.2, 0.015, 0.6])  # left colorbar
    ax_red = fig.add_axes([0.83, 0.2, 0.015, 0.6])   # right colorbar

    # Draw blue colorbar without ticks (no scale)
    cb_blue = plt.colorbar(blue_sm, cax=ax_blue, orientation='vertical')
    cb_blue.ax.yaxis.set_visible(False)  # Hide ticks and labels

    cb_red = plt.colorbar(red_sm, cax=ax_red, orientation='vertical')
    cb_red.set_label("CIEDE2000 Deviation\nBlue - darker than average\n Red- brighter than average", fontsize=8)

    plt.show()

    #creating the colorbar
    #combined_colors = [(1.0 + ((1/2.55)-1), 0.0, 0.0), 
    #                   (1.0, 1.0, 1.0), 
    #                   (0.0, 0.0, 1.0 + ((1/2.55)-1))]
    
    #colormap = col.LinearSegmentedColormap.from_list("red_white_blue", combined_colors, N=255)
    #norm = col.Normalize(vmin = 0, vcenter=0, vmax = max_ciede)
    #sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    #cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.05)
    #cbar.set_label("Blue - darker than average, Red - brighter than average", fontsize=6)
    #plt.show()

def compare_it87(in_img, ref_data, show_img=True):
    rec = image_manipulation.convert_color(in_img, 'show')

    # Calculate sample average colors
    fit_data = np.zeros(6)
    sample_data = np.zeros((4, ref_data[0][0][0], ref_data[0][0][1]))
    lab_vals = [[[], []], [[], []], [[], []]]
    lab_range = [[100, 100, 100], [0, -100, -100]]
    labels = ('L*', 'a*', 'b*', 'CIELAB')
    for l_y in range(ref_data[0][0][1]):
        for l_x in range(ref_data[0][0][0]):
                       # Find top left corner of sample
            top_left = (round((ref_data[0][1][0][0] + ref_data[0][1][1][0] * l_x)
                              * (in_img[0].shape[1] / ref_data[0][0][0])),
                        round((ref_data[0][1][0][1] + ref_data[0][1][1][1] * l_y)
                              * (in_img[0].shape[0] / ref_data[0][0][1])))

            # Find bottom right corner of sample
            bottom_right = (round(top_left[0] + ref_data[0][1][2][0] * (in_img[0].shape[1] / ref_data[0][0][0])),
                            round(top_left[1] + ref_data[0][1][2][1] * (in_img[0].shape[0] / ref_data[0][0][1])))
            rec = (cv.rectangle(rec[0], top_left, bottom_right, (0, 0, 255), 2), rec[1])        # Draw rectangle
            grid_sample = in_img[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]   # Crop sample
            avg_col = image_utilities.get_average_color((grid_sample, in_img[1]))               # Sample average color
            sample_data[0, l_x, l_y] = colour.delta_E(avg_col, ref_data[1][l_x][l_y], method='CIE 2000')
            print(f'({l_x + 1}, {l_y + 1})', f"Average color - LAB ({settings.output_illuminant}):",
                  image_manipulation.convert_color((avg_col, in_img[1]), 'LAB')[0].round(3))
            print("Reference CIEDE2000:", "{:.3f}".format(sample_data[0, l_x, l_y]))

            for i in range(3):
                # Append sample color
                lab_vals[i][0].append(avg_col[i])
                lab_vals[i][1].append(ref_data[1][l_x][l_y][i])

                # Sample deviation from reference data
                sample_data[i + 1, l_x, l_y] = np.subtract(lab_vals[i][0][-1], lab_vals[i][1][-1])

                # Save LAB ranges
                lab_range[0][i] = min(lab_range[0][i], lab_vals[i][0][-1])
                lab_range[1][i] = max(lab_range[1][i], lab_vals[i][0][-1])

    for i in range(3):
        # Calculate average differences
        fit_data[i + 3] = np.average(np.average(np.abs(sample_data[i + 1])))

        # Check if LAB are out of domain (without considering color ranges for specific luminance)
        if lab_range[0][i] < settings.input_lab_domain[0][i] or lab_range[1][i] > settings.input_lab_domain[1][i]:
            utilities.print_color("Input " + labels[i] + " out of domain: (" + str(lab_range[0][i]) + ', ' +
                                  str(lab_range[1][i]) + ')!', 'error')

    fit_data[0] = np.average(sample_data[0])            # Average CIEDE2000
    fit_data[1] = sample_data[0].min(initial=1000.0)    # Min. CIEDE2000
    fit_data[2] = sample_data[0].max(initial=0.0)       # Max. CIEDE2000

    print()
    print("Avg. luminance difference:", fit_data[3].round(3))
    print("Avg. a difference:", fit_data[4].round(3))
    print("Avg. b difference:", fit_data[5].round(3))

    print("Ref. average CIEDE2000:", "{:.3f}".format(fit_data[0]))
    print("Ref. CIEDE2000 range: [", "{:.3f}".format(fit_data[1]), "-",
          "{:.3f}".format(fit_data[2]), "]")

    if show_img:
        # Show corrected color target
        image_utilities.show_image("Corrected target (Press Esc to discard, or any other key to save)",
                                   image_manipulation.scale_image(rec)[0], False)

    print()
    return fit_data, sample_data, (fit_data[0], fit_data[3], fit_data[4], fit_data[5])


