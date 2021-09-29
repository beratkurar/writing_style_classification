import numpy as np
import os
import random
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm
import wandb
from scipy.ndimage import label as bwlabel
from scipy.signal import find_peaks
from skimage.measure import regionprops
import statistics
from skimage.filters import *
from scipy.signal import savgol_filter
import _thread


def number_of_peaks(data, thresh):
    
    data = savgol_filter(data, 51, 5)
    
    peaks, _ = find_peaks(data, prominence=1, width=20, distance=20)
    
    return peaks.shape[0]


def is_valid_patch(patch, binary_patch, x_var_thresh=500, y_var_thresh=500, y_peaks_thresh=220, alpha=0.01, beta=0.7, 
                   validate_based_on_cc=True, cc_thresh=30):
    inv_patch = ((255 - patch)/255)
    x_profile = np.sum(inv_patch, axis=0)
    y_profile = np.sum(inv_patch, axis=1)
    
    x_r, x_l = np.sum(x_profile[:x_profile.shape[0]//2]), np.sum(x_profile[x_profile.shape[0]//2:])
    y_r, y_l = np.sum(y_profile[:y_profile.shape[0]//2]), np.sum(y_profile[y_profile.shape[0]//2:])
    
    x_var = x_profile.var() # centered lines
    y_var = y_profile.var() # number of lines
    
    
    f_is_valid = beta * (patch.shape[0]*patch.shape[1]) > np.count_nonzero(binary_patch) > alpha * (patch.shape[0]*patch.shape[1])
    
    
    lines_number = number_of_peaks(y_profile, y_peaks_thresh)
    
    
#     print(y_var, x_var, lines_number, x_r/x_l, y_r/y_l, (1.5 > x_r/x_l > 0.5), (1.5 > y_r/y_l > 0.5), f_is_valid, (np.count_nonzero(binary_patch)/(patch.shape[0]*patch.shape[1])))

    cc_valid = True
    
    if True: #validate_based_on_cc:
        contours, hierarchy = cv2.findContours(255-patch.astype(np.uint8), 1, 2)

        
        boxes = []
        conts_bbs = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            conts_bbs.append(np.asarray([[x, y],[x+w, y],[x+w, y+h],[x, y+h]], dtype=np.int))

            if 0.001*patch.shape[0]*patch.shape[1] < area < 0.2*patch.shape[0]*patch.shape[1]:

                box = np.asarray([[x, y],[x+w, y],[x+w, y+h],[x, y+h]], dtype=np.int)
                boxes.append(box)
        
        
#         d_img = cv2.cvtColor(patch.astype(np.uint8).copy(),cv2.COLOR_GRAY2RGB)
#         import matplotlib.pyplot as plt
#         c_img = cv2.drawContours(d_img,boxes,-1,(0,255,0),10)
#         plt.imshow(c_img)
#         plt.show()
            
        cc_valid = len(boxes) > cc_thresh
        
    
    is_valid = f_is_valid and y_var > y_var_thresh and x_var < x_var_thresh and lines_number > 0 and \
    (1.5 > x_r/x_l > 0.5) and (1.5 > y_r/y_l > 0.5) and patch.min() != patch.max() and cc_valid
    
    
    return is_valid, lines_number

def calculate_average_number_of_line_in_patch(img, patch_size=(600,600), samples_number=20, max_iterations=10000):
    number_of_lines = 0
    sampled_patches = 0
    
    iterations = 0
    
    while sampled_patches < samples_number:
        x,y = get_random_patch_location(img)
        patch = get_patch(img, x, y, src_patch_size=patch_size, dst_patch_size=(350,350))
        
        iterations += 1
        
        if iterations > max_iterations:
            return -1
        
        if patch.min() == patch.max():
            continue
        
        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(patch, selem)
        threshold_global_otsu = threshold_otsu(patch)
        bin_patch = 255* (patch >= threshold_global_otsu)

        is_valid, lines = is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=500, y_peaks_thresh=80, alpha=0.01, cc_thresh=10)
        #is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=0.1)
        

        if is_valid:
            sampled_patches += 1
            number_of_lines += lines
            
    return number_of_lines/sampled_patches




def get_valid_patch(img, src_patch_size=(900,900), dst_patch_size=(350,350), margine_p_x = 0.1, margine_p_y = 0.1, 
                    max_iterations=10000, validate_based_on_cc=True):
    iterations = 0
    
    is_valid = False
    while not is_valid:
        x,y = get_random_patch_location(img)
#         x,y = get_random_patch_location_in_range(img, (margine_p_x*img.shape[0], img.shape[0]*(1-margine_p_x)),
#                                                        (margine_p_y*img.shape[1], img.shape[1]*(1-margine_p_y)))
        
        patch = get_patch(img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)
        
        iterations += 1
        
        if patch.min() == patch.max():
            continue

        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(patch, selem)
        threshold_global_otsu = threshold_otsu(patch)
        bin_patch = 255* (patch >= threshold_global_otsu)

        
        is_valid, lines = is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=500, 
                                         y_peaks_thresh=80, alpha=0.01, validate_based_on_cc=validate_based_on_cc, cc_thresh=30)
        #is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=0.1)
        
        if iterations > max_iterations:
            return None, -1
        
    return patch, lines


def get_valid_patch_with_info(img, src_patch_size=(900,900), dst_patch_size=(350,350), margine_p_x = 0.1, margine_p_y = 0.1, 
                    max_iterations=10000, validate_based_on_cc=True):
    iterations = 0
    
    is_valid = False
    while not is_valid:
#         x,y = get_random_patch_location(img)
        x,y = get_random_patch_location_in_range(img, (margine_p_x*img.shape[0], img.shape[0]*(1-margine_p_x)),
                                                       (margine_p_y*img.shape[1], img.shape[1]*(1-margine_p_y)))
        
        patch = get_patch(img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)
        
        iterations += 1
        
        if patch.min() == patch.max():
            continue

        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(patch, selem)
        threshold_global_otsu = threshold_otsu(patch)
        bin_patch = 255* (patch >= threshold_global_otsu)

        
        is_valid, lines = is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=500, 
                                         y_peaks_thresh=80, alpha=0.01, validate_based_on_cc=validate_based_on_cc, cc_thresh=30)
        #is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=0.1)
        
        if iterations > max_iterations:
            return None, -1
        
    return patch, lines, (x,y)



def get_page_heat_map(page_img, number_of_patches=5, number_of_lines_per_patch = 5, patch_size = (350, 350), 
                                  margine_p_x = 0.1, margine_p_y = 0.1, validate_based_on_cc=False):
    init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

    avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)
    src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))
    
    heat_map = np.zeros_like(page_img)

    for i in tqdm(range(number_of_patches), desc='sampling patches for page'):
        
        patch, lines, (x,y) = get_valid_patch_with_info(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size, 
                                   validate_based_on_cc=validate_based_on_cc)
        
        heat_map[x:x+src_patch_size[0], y:y+src_patch_size[1]] += 1

    return heat_map


    

def estimate_source_patch_size(img, src_patch_size=(900,900), number_of_lines=6):
            
    avg_lines_num = calculate_average_number_of_line_in_patch(img, patch_size=src_patch_size)
    
    if avg_lines_num < 0:
        return None
    
    src_patch_size = (int(src_patch_size[0]*(number_of_lines/avg_lines_num)), int(src_patch_size[1]*(number_of_lines/avg_lines_num)))
    
    return src_patch_size
    
    
def get_random_image_path(dir):
    pages=os.listdir(dir)
    if len(pages) == 0:
        print('Warning: there is no images in {}'.format(style_path))
        return None
    
    selected_index=random.randint(0, len(pages) - 1)
    selected_page_name=pages[selected_index]
    selected_page_path=os.path.join(dir,selected_page_name)
    
    return selected_page_path

def get_random_image(dir):
    pages=os.listdir(dir)
    if len(pages) == 0:
        print('Warning: there is no images in {}'.format(style_path))
        return None
    
    selected_index=random.randint(0, len(pages) - 1)
    selected_page_name=pages[selected_index]
    selected_page_path=os.path.join(dir,selected_page_name)
    
    print(selected_page_path)

    selected_page_image = cv2.imread(selected_page_path, 0)
    
    return selected_page_image


def get_random_patch_location(img, patch_size=(350,350)):
    rows, cols = img.shape
    x = random.randint(0, rows - patch_size[0])
    y = random.randint(0, cols - patch_size[1])
    
    return x,y


def get_random_patch_location_in_range(img, x_range, y_range, patch_size=(350,350)):
    rows, cols = img.shape[:2]
    
    if x_range[1] > rows or y_range[1] > cols:
        print('Error: image out of range')
        return None, None
    
    x = random.randint(int(x_range[0]), int(x_range[1]-1) - patch_size[0])
    y = random.randint(int(y_range[0]), int(y_range[1]-1) - patch_size[1])
    
    return x, y


def get_patch(img, x, y, src_patch_size=(350,350), dst_patch_size=(350,350)):
    patch = img[x:x + src_patch_size[0], y:y + src_patch_size[1]]
    # print(x, x + src_patch_size[0], y, y + src_patch_size[1], patch.shape, src_patch_size)
    patch = cv2.resize(patch, dst_patch_size)
    return patch


def generate_patches_for_style(style_path, output_class_folder, number_of_patches_per_page, patch_size=(350,350), 
                               number_of_lines_per_patch=5, margine_p_x = 0.1, margine_p_y = 0.1):
        
    curr_patch_id = 0

    for page in tqdm(os.listdir(style_path)):
        page_path=os.path.join(style_path,page)
        page_img = cv2.imread(page_path, 0)
        init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

        avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)
        
        if avg_lines_num < 0:
            print('Error couldn\'t calculate avg number of line in page', page_path)
            continue

        src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))


        for i in tqdm(range(number_of_patches_per_page), desc='generating patches for page {}'.format(page)):

            patch, _ = get_valid_patch(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size, margine_p_x = 
                                       margine_p_x, margine_p_y = margine_p_y)
            if patch is None:
                print('Error: unable to generate patches from page {}'.format(page_path))
                break
                      
            cv2.imwrite(output_class_folder+'/'+str(curr_patch_id)+'.png',patch)
            curr_patch_id += 1
            
def multithread_generate_patches_for_style(style_path, output_class_folder, number_of_patches_per_page, patch_size=(350,350), number_of_lines_per_patch=8):
        
    curr_patch_id = 0

    for page in tqdm(os.listdir(style_path)):
        page_path=os.path.join(style_path,page)
        page_img = cv2.imread(page_path, 0)
        init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

        avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)
        
        if avg_lines_num < 0:
            print('Error couldn\'t calculate avg number of line in page', page_path)
            continue

        src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))


        for i in range(number_of_patches_per_page):
            x,y = get_random_patch_location(page_img, patch_size=src_patch_size)

            patch, _ = get_valid_patch(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size)
            
            cv2.imwrite(output_class_folder+'/'+str(curr_patch_id)+'.png',patch)
            curr_patch_id += 1
            
def generate_patches(data_folder, output_folder, patch_size=(350,350), number_of_patches_per_style=5000, skip_to=None):

    labels = os.listdir(data_folder)
    output_folder = '{}_{}x{}'.format(output_folder, patch_size[0], patch_size[1])

    pages = []
    pages_labels = []

    os.makedirs(output_folder, exist_ok=True)
    
    skip = skip_to is not None 

    for style_folder in os.listdir(data_folder):
        print(style_folder)
        
        if skip:
            if style_folder == skip_to:
                skip = False
            else:
                continue
        
        style_path=os.path.join(data_folder,style_folder)
        output_class_folder=os.path.join(output_folder,style_folder)
        os.makedirs(output_class_folder, exist_ok=True)
        
        number_of_patches_per_page = number_of_patches_per_style//len(os.listdir(style_path))
        
        print(number_of_patches_per_page)
        
        generate_patches_for_style(style_path, output_class_folder, number_of_patches_per_page, patch_size=patch_size)
        
        
def multithread_generate_patches(data_folder, output_folder, patch_size=(350,350), number_of_patches_per_style=5000):

    labels = os.listdir(data_folder)
    output_folder = '{}_{}x{}'.format(output_folder, patch_size[0], patch_size[1])

    pages = []
    pages_labels = []

    os.makedirs(output_folder, exist_ok=True)


    for style_folder in os.listdir(data_folder):
        print(style_folder)
        
        style_path=os.path.join(data_folder,style_folder)
        output_class_folder=os.path.join(output_folder,style_folder)
        os.makedirs(output_class_folder, exist_ok=True)
        
        number_of_patches_per_page = number_of_patches_per_style//len(os.listdir(style_path))
        
        print(number_of_patches_per_page)
        
        _thread.start_new_thread(multithread_generate_patches_for_style, (style_path, output_class_folder, number_of_patches_per_page, patch_size))
        
def sample_patches_from_page(page_img, number_of_patches=5, number_of_lines_per_patch = 5, patch_size = (350, 350), 
                                  margine_p_x = 0.1, margine_p_y = 0.1, validate_based_on_cc=True):

    init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

    avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)
    src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))
    
    patches = []

    for i in tqdm(range(number_of_patches), desc='generating patches for page'):
#         x,y = get_random_patch_location(page_img, patch_size=src_patch_size)
#         x,y = get_random_patch_location_in_range(page_img, (margine_p_x*page_img.shape[0], page_img.shape[0]*(1-margine_p_x)),
#                                                        (margine_p_y*page_img.shape[1], page_img.shape[1]*(1-margine_p_y)))
        
        patch, _ = get_valid_patch(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size, 
                                   validate_based_on_cc=validate_based_on_cc)
        patches.append(patch)

    return patches


def sample_patches_from_page_path(page_path, number_of_patches=5, number_of_lines_per_patch = 5, patch_size = (350, 350), 
                                  margine_p_x = 0.1, margine_p_y = 0.1):
    page_img = cv2.imread(page_path)

    init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

    avg_lines_num = calculate_average_number_of_line_in_patch(page_img, patch_size=init_patch_size)
    src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))
    
    patches = []

    for i in tqdm(range(number_of_patches), desc='generating patches for page'):
        #x,y = get_random_patch_location(page_img, patch_size=src_patch_size)
        
        
        x,y = get_random_patch_location_in_range(page_img, (margine_p_x*img.shape[0], img.shape[0]*(1-margine_p_x)),
                                                       (margine_p_y*img.shape[1], img.shape[1]*(1-margine_p_y)))
        
        patch, _ = get_valid_patch(page_img, src_patch_size=src_patch_size, dst_patch_size=patch_size)
        patches.append(patch)

    return patches


def sample_patches_from_page_w_binary(page_img, binary_page_img, number_of_patches=5, number_of_lines_per_patch = 5, 
                                      patch_size = (350, 350), margine_p_x = 0.1, margine_p_y = 0.1):

    init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

    avg_lines_num = calculate_average_number_of_line_in_patch_w_binary(page_img, binary_page_img, patch_size=init_patch_size)
    src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))
    
    patches = []

    for i in tqdm(range(number_of_patches), desc='generating patches for page'):
        patch, _ = get_valid_patch_w_binary(page_img, binary_page_img, src_patch_size=src_patch_size, 
                                                dst_patch_size=patch_size, margine_p_x = margine_p_x, margine_p_y = margine_p_y)
        if patch is not None:
            patches.append(patch)

    return patches

def get_valid_patch_w_binary_w_info(img, b_img, src_patch_size=(900,900), dst_patch_size=(350,350), 
                             margine_p_x = 0.1, margine_p_y = 0.1, max_iterations=10000):
    iterations = 0
    
    is_valid = False
    while not is_valid:
#         x,y = get_random_patch_location(img)
        x,y = get_random_patch_location_in_range(img, (margine_p_x*img.shape[0], img.shape[0]*(1-margine_p_x)),
                                                       (margine_p_y*img.shape[1], img.shape[1]*(1-margine_p_y)))
        
        patch = get_patch(img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)
        bin_patch = get_patch(b_img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)
        
        bin_patch = 255 - bin_patch
        iterations += 1
        
        if bin_patch.min() == bin_patch.max():
            continue
        
        is_valid, lines = is_valid_patch(bin_patch, 1-(bin_patch >= 100),x_var_thresh=1500, y_var_thresh=500, y_peaks_thresh=80, alpha=0.01, cc_thresh=20)
        #is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=0.1)
        
        if iterations > max_iterations:
            return None, -1, (-1,-1)
        
    return patch, lines, (x,y)

def get_page_heat_map_w_binary(page_img, binary_page_img, number_of_patches=5, number_of_lines_per_patch = 5, 
                                      patch_size = (350, 350), margine_p_x = 0.1, margine_p_y = 0.1):

    init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)

    avg_lines_num = calculate_average_number_of_line_in_patch_w_binary(page_img, binary_page_img, patch_size=init_patch_size)
    if avg_lines_num == -1:
        print('Error: calculating average line numbers')
        return None
    
    src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))
    
    patches = []
    
    
    heat_map = np.zeros_like(page_img)

    for i in tqdm(range(number_of_patches), desc='sampling patches for page'):
        
        patch, lines, (x,y) = get_valid_patch_w_binary_w_info(page_img, binary_page_img, src_patch_size=src_patch_size, 
                                                dst_patch_size=patch_size, margine_p_x = margine_p_x, margine_p_y = margine_p_y)
        
        if patch is None:
            print('Error: unable to generate pache from page')
            return heat_map
        
        heat_map[x:x+src_patch_size[0], y:y+src_patch_size[1]] += 1


    return heat_map

def get_valid_patch_w_binary(img, b_img, src_patch_size=(900,900), dst_patch_size=(350,350), 
                             margine_p_x = 0.1, margine_p_y = 0.1, max_iterations=1000):
    iterations = 0
    # print(img.shape, b_img.shape, src_patch_size, '\n\n')

    is_valid = False
    while not is_valid:
        # x,y = get_random_patch_location(img)
        x,y = get_random_patch_location_in_range(img, (margine_p_x*img.shape[0], img.shape[0]*(1-margine_p_x)),
                                                       (margine_p_y*img.shape[1], img.shape[1]*(1-margine_p_y)), patch_size=src_patch_size)
        
        patch = get_patch(img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)
        bin_patch = get_patch(b_img, x, y, src_patch_size=src_patch_size, dst_patch_size=dst_patch_size)
        
        bin_patch = 255 - bin_patch
        iterations += 1
        
        if bin_patch.min() == bin_patch.max():
            continue
        
        is_valid, lines = is_valid_patch(bin_patch, 1-(bin_patch >= 100),x_var_thresh=1500, y_var_thresh=500, y_peaks_thresh=80, alpha=0.01, cc_thresh=10)
        #is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=0.1)
        
        if iterations > max_iterations:
            return None, -1
        
    return patch, lines



def calculate_average_number_of_line_in_patch_w_binary(img, b_img, patch_size=(600,600), samples_number=20, 
                                                       margine_p_x = 0.1, margine_p_y = 0.1, max_iterations=5000):
    number_of_lines = 0
    sampled_patches = 0
    
    iterations = 0
    
    while sampled_patches < samples_number:
        
        iterations += 1
        
        if iterations > max_iterations:
            return -1
        
#         x,y = get_random_patch_location(img)
        x,y = get_random_patch_location_in_range(img, (margine_p_x*img.shape[0], img.shape[0]*(1-margine_p_x)),
                                                       (margine_p_y*img.shape[1], img.shape[1]*(1-margine_p_y)), patch_size=patch_size)
        # print(img.shape)
        # print(b_img.shape)
        patch = get_patch(img, x, y, src_patch_size=patch_size, dst_patch_size=(350,350))
        
        bin_patch = get_patch(b_img, x, y, src_patch_size=patch_size, dst_patch_size=(350, 350))
        
        bin_patch = 255 - bin_patch
        
        if patch.min() == patch.max():
            continue
            

        
        is_valid, lines = is_valid_patch(bin_patch, 1-(bin_patch >= 100),x_var_thresh=1500, y_var_thresh=500, y_peaks_thresh=80, alpha=0.01, cc_thresh=10)
        #is_valid_patch(bin_patch, 1-(patch >= threshold_global_otsu),x_var_thresh=1500, y_var_thresh=1500, y_peaks_thresh=80, alpha=0.1)
        

        if is_valid:
            sampled_patches += 1
            number_of_lines += lines
            
    return number_of_lines/sampled_patches

def generate_patches_for_style_w_binary(style_path, binary_style_path, output_class_folder, number_of_patches_per_page, 
                                        patch_size=(350,350), number_of_lines_per_patch=5, margine_p_x = 0.1, margine_p_y = 0.1):
        
    curr_patch_id = 0

    for page in tqdm(os.listdir(style_path)):
        page_path=os.path.join(style_path,page)
        binary_page_path=os.path.join(binary_style_path,f'{page.split(".")[0]}.png')
        
        ext = page_path.split('.')[-1]
        
        if ext.lower() not in ['png', 'jpg', 'jpeg', 'tiff']:
            continue
        
        print(page_path)
        page_img = cv2.imread(page_path, 0)
        binary_page_img = cv2.imread(binary_page_path, 0)
        
        init_patch_size = (page_img.shape[0]//10,page_img.shape[0]//10)
        print(page_path, 'calculating pach size')
        avg_lines_num = calculate_average_number_of_line_in_patch_w_binary(page_img, binary_page_img, patch_size=init_patch_size)

        if avg_lines_num == -1:
            print('Error: unable to calculating pach size from page {}'.format(page_path))
            continue
        
        src_patch_size = (int(init_patch_size[0]*(number_of_lines_per_patch/avg_lines_num)), int(init_patch_size[1]*(number_of_lines_per_patch/avg_lines_num)))


        for i in tqdm(range(number_of_patches_per_page), desc='generating patches for page {}'.format(page)):

            patch, _ = get_valid_patch_w_binary(page_img, binary_page_img, src_patch_size=src_patch_size, 
                                                dst_patch_size=patch_size, margine_p_x = margine_p_x, margine_p_y = margine_p_y)
            if patch is None:
                print('Error: unable to generate patches from page {}'.format(page_path))
                break
                      
            cv2.imwrite(output_class_folder+'/'+str(curr_patch_id)+'.png',patch)
            curr_patch_id += 1
            

def generate_patches_with_binary(data_folder, binary_data_folder, output_folder, patch_size=(350,350), 
                                 number_of_patches_per_style=5000, skip_to=None):

    labels = os.listdir(data_folder)
    output_folder = '{}_{}x{}_fb'.format(output_folder, patch_size[0], patch_size[1])

    pages = []
    pages_labels = []

    os.makedirs(output_folder, exist_ok=True)
    
    skip = skip_to is not None 

    for style_folder in os.listdir(data_folder):
        print(style_folder)
        
        if skip:
            if style_folder == skip_to:
                skip = False
            else:
                continue
        
        style_path=os.path.join(data_folder,style_folder)
        binary_style_path=os.path.join(binary_data_folder,style_folder)
        output_class_folder=os.path.join(output_folder,style_folder)
        os.makedirs(output_class_folder, exist_ok=True)
        
        number_of_patches_per_page = number_of_patches_per_style//len(os.listdir(style_path))
        
        print(number_of_patches_per_page)
        
        generate_patches_for_style_w_binary(style_path, binary_style_path, output_class_folder, number_of_patches_per_page, patch_size=patch_size)
