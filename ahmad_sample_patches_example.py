from util import *
import matplotlib.pyplot as plt
        
page_path= 'dataset/PALEOGRAPHY_TRAIN_AND_TEST_v3/TEST/orientalsquare/0129_FL12951770.jpg'
patches = sample_patches_from_page(page_path, number_of_patches=5)

for patch in patches:
    plt.imshow(patch)
    plt.show()

