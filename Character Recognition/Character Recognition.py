from PIL import *
from PIL import Image
import numpy as np
import math
from PIL import Image, ImageDraw
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance


def main():
    img = Image.open('sayılar.jpg')
    img_gray = img.convert('L') # converts the image to grayscale image
    img_bin = img.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    img_gray.show()
    ONE = 150
    a = np.asarray(img_gray) #from PIL to np array
    a_bin = threshold(a,100,ONE,0)
    im = Image.fromarray(a_bin) # from np array to PIL format
    #im.show()

    #a_bin = binary_image(100,100, ONE)   #creates a binary image
    label_im, label_im_color = blob_coloring_8_connected(a_bin, ONE)
    np2PIL(label_im_color).show()
    bounding_rectangles, label_set = find_rectangles(label_im)
    print(len(label_set))
    draw_rectangles(label_im_color, bounding_rectangles, label_set)





def binary_image(nrow,ncol,Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow,ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10


    for i in range (50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i-20][90-i+1] = 1
        mask_lines[i-20][90-i+2] = 1
        mask_lines[i-20][90-i+3] = 1


    #mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
    #mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    #mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    #mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    #imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    #imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge

def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def threshold(im,T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    print("nrow, ncol", nrow, ncol)
    im = np.zeros(shape=(nrow,ncol), dtype = int)
    a = np.zeros(shape=max_label, dtype = int)
    a = np.arange(0,max_label, dtype = int)
    color_map = np.zeros(shape = (max_label,3), dtype= np.uint8)
    color_im = np.zeros(shape = (nrow, ncol,3), dtype= np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][1] = np.random.randint(0,255,1,dtype = np.uint8)
        color_map[i][2] = np.random.randint(0,255,1,dtype = np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c   = bim[i][j]
                l   = bim[i][j - 1]
                u   = bim[i - 1 ][j]
                label_u  = im[i -1][j]
                label_l  = im[i][j - 1]
                label_ul = im[i-1][j-1]
                label_ur = im[i-1][j+1]
                im[i][j] = max_label
                if c == ONE:
                    min_label = min( label_u, label_l,label_ul,label_ur)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label  :
                            update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label  :
                            update_array(a, min_label, label_l)

                        if min_label != label_ur and label_ur != max_label:
                             update_array(a, min_label, label_ur)

                        if min_label != label_ul and label_ul != max_label:
                            update_array(a, min_label, label_ul)

                else :
                    im[i][j] = max_label
    # final reduction in label array
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    #second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j],0]
                color_im[i][j][1] = color_map[im[i][j],1]
                color_im[i][j][2] = color_map[im[i][j],2]
    return im, color_im

def update_array(a, label1, label2) :
    index = lab_small = lab_large = 0
    if label1 < label2 :
        lab_small = label1
        lab_large = label2
    else :
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else: #a[index] == lab_small
            break

    return

def find_rectangles(im):
    nrow = im.shape[0]
    ncol = im.shape[1]
    k_max = 0
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] == 10000:
                continue
            myLabel = im[i][j]
            if myLabel > k_max:
                 k_max = myLabel
    bounding_rectangles = np.zeros([k_max+1, 4])
    mini = np.full(k_max+1, np.inf)
    minj = np.full(k_max+1, np.inf)
    maxi = np.zeros(k_max+1)
    maxj = np.zeros(k_max+1)
    label_set = set()
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] == 10000:
                continue
            myLabel = im[i][j]
            label_set.add(myLabel)
            if i < mini[myLabel]:
                mini[myLabel] = i
            if i > maxi[myLabel]:
                maxi[myLabel] = i
            if j < minj[myLabel]:
                minj[myLabel] = j
            if j > maxj[myLabel]:
                maxj[myLabel] = j
    for i in range(k_max + 1):
        bounding_rectangles[i] = (mini[i], minj[i], maxi[i], maxj[i])

    return bounding_rectangles, label_set

def draw_rectangles(color_img_array, rectangles, label_set):
    color_img = np2PIL(color_img_array)
    img_draw = ImageDraw.Draw(color_img)
    for rect in range(len(rectangles)):
        if rect in label_set:
            miny, minx, maxy, maxx = rectangles[rect]
            img_draw.rectangle([minx, miny, maxx, maxy], outline='red')
            cropped =color_img.crop([minx, miny, maxx, maxy])
            resized_image = cropped.resize((21,21))
            moment(resized_image)
            resized_image.show()
    color_img.show()
    return cropped

def moment(cropped):
    f=np.asarray(cropped)
    print(f)
    nrow =f.shape[0]
    ncol =f.shape[1]

    m = [[0,0],[0,0]]
    mu = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] #central moments
    n = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] # normalized central moments

    for p in range(2):
        for q in range(2):
            for x in range(ncol):
                for y in range(nrow):
                    m[p][q] = m[p][q] + ((x**p)*(y**q))*(f[x][y])



    x0=((m[1][0])/(m[0][0]))
    y0=(m[0][1]/(m[0][0]))

    for p in range(4):
        for q in range(4):
            ye = ((p + q) / 2) + 1
            for x in range(ncol):
                for y in range (nrow):
                    mu[p][q] = mu[p][q] +((x-x0)**p)*((y-y0)**q)*f[x][y]
                    n[p][q] = (mu[p][q])/(mu[0][0]**ye)


    h1 = n[2][0] + n[0][2]
    h2 = ((n[2][0] + n[0][2])**2) + (4*(n[1][1])**2)
    h3 = ((n[3][0] - n[1][2])**2) + ((n[2][1]-n[0][3])**2)
    h4 = ((n[3][0] + n[1][2])**2) + ((n[2][1]+n[0][3])**2)
    #h5 = ((n[3][0]- (3*(n[1][2]))*(n[3][0]+n[1][2])*[((n[3][0]+n[1][2])**2)-(3*((n[2][1]+n[0][3])**2))]+(3*(n[2][1]-n[0][3)**2)+(3(n[1][2]-n[3][0])*(n[2][1]+n[0][3])*[3((n[3][0]+n[1][2])**2)-(n[2][1]+n[0][3])**2)])
    h6 = (n[2][0]-n[0][2]*[((n[3][0]+n[1][2])**2)-((n[2][1]+n[0][3])**2)]) + 4*n[1][1]*(n[3][0]+n[1][2])*((n[2][1]+n[0][3]**2))
    #h7 = ((3 * n[2][1] - n[0][3]) * (n[3][0] + n[1][2]) * [((n[3][0] + n[1][2]) ** 2) - (3 * ((n[2][1] + n[0][3]) ** 2))] + (3 * n[1][2] - n[3][0]) * ( n[2][1] + n[0][3]) * [3((n[3][0] + n[1][2]) ** 2) - ((n[2][1] + n[0][3]) ** 2)]



    r1 = pow(h2,0.5)/h1
    r2 = h1 + pow(h2,0.5)/h1-pow(h2,0.5)
    r3 = pow(h3,0.5)/pow(h4,0.5)
    #r4 = pow(h3,0.5)/pow(abs(h5,0.5))
    #r5 = pow(h4,0.5)/pow(abs(h5,0.5))
    r6 = abs(h6)/h1*h3
    #r7 = abs(h3)/h1*pow(abs(h5,0.5))
    r8 = abs(h6)/h3*pow(h2,0.5)
    #r9 = abs(h6)/pow(h2*abs(h5),0.5)
    #r10 = abs(h5)/h3*h4








if __name__=='__main__':
    main()
