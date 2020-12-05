import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

from PIL import Image

pic = Image.open('./assets/images/bk.jpg')

pic_array = np.asarray(pic)

# plt.imshow(pic_array[:,:,1],'gray')
# plt.imshow(pic_array[:,:,1])

pic_red = pic_array.copy()

pic_red[:,:,1:3] = 0

plt.imshow(pic_red)

plt.savefig('myfig.png')

plt.show()
