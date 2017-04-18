
# **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note** If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
from __future__ import division
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
```

    ('This image is:', <type 'numpy.ndarray'>, 'with dimesions:', (540, 960, 3))





    <matplotlib.image.AxesImage at 0x7f52df1d18d0>




![png](output_3_2.png)


**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(line_image, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point 
    once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    '''
        for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    '''
 
    '''
    Finding the first point corresponding to left and right track
    '''
    i = 0
    h = 0
    j = 0

    prev_x2_l = 0
    prev_y2_l = 0
    prev_x2_r = 0
    prev_y2_r = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            p = ((y2-y1)//(x2-x1))
            if p == 0 and h != 1:
                prev_x2_l = lines[i,0,0]
                prev_y2_l = lines[i,0,1]
                h = 1

            if p == -1 and j != 1:
                prev_x2_r = lines[i,0,0]
                prev_y2_r = lines[i,0,1]
                j = 1

            i = i+1

    end_x2_l  = 0
    end_y2_l  = 0
    end_x2_r  = 0
    end_y2_r  = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            p = ((y2-y1)//(x2-x1))
            if p == -1:          
                if x1 != prev_x2_r:
                    cv2.line(line_image,(prev_x2_r,prev_y2_r),(x1,y1),(255,0,0),10)
                    prev_x2_r = x1
                    prev_y2_r = y1

                    if x2 > end_x2_r:
                        end_x2_r  = x2
                        end_y2_r  = y2
                else:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                    prev_x2_r = x2
                    prev_y2_r = y2

            elif p == 0:
                if x1 != prev_x2_l:
                    cv2.line(line_image,(prev_x2_l,prev_y2_l),(x1,y1),(255,0,0),10)
                    prev_x2_l = x1
                    prev_y2_l = y1

                    if x2 > end_x2_l:
                        end_x2_l  = x2
                        end_y2_l  = y2
                else:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
                    prev_x2_l = x2
                    prev_y2_l = y2


    cv2.line(line_image,(prev_x2_l,prev_y2_l),(end_x2_l,end_y2_l),(255,0,0),10)
    cv2.line(line_image,(prev_x2_r,prev_y2_r),(end_x2_r,end_y2_r),(255,0,0),10)


def draw_lines1(line_image, lines, color=[255, 0, 0], thickness=2):
    i = 0
    h = 0
    j = 0


    prev_x2_l = 0
    prev_y2_l = 0
    prev_x2_r = 0
    prev_y2_r = 0



    for line in lines:
        for x1,y1,x2,y2 in line:
            p = (y2-y1)/(x2-x1)

            if p < -0.3 and h != 1:
                prev_x2_l = lines[i,0,0]
                prev_y2_l = lines[i,0,1]
                h = 1

            if p >= 0.2 and p < 0.6 and j != 1:
                prev_x2_r = lines[i,0,0]
                prev_y2_r = lines[i,0,1]
                j = 1

            i = i+1

    prev_x2_l1 = lines[2,0,0]
    prev_y2_l1 = lines[2,0,1]
    prev_x2_r1 = lines[0,0,0]
    prev_y2_r1 = lines[0,0,1]


    '''
    print prev_x2_l
    print prev_y2_l
    print prev_x2_l1
    print prev_y2_l1
    print prev_x2_r
    print prev_y2_r
    print prev_x2_r1
    print prev_y2_r1
    '''

    '''
    Extrapolating between the points in 
    '''


    end_x2_l  = 0
    end_y2_l  = 0
    end_x2_r  = 0
    end_y2_r  = 0



    for line in lines:
        for x1,y1,x2,y2 in line:
            p = ((y2-y1)/(x2-x1))

            if p <= -0.3:          
                if x1 != prev_x2_r:
                    cv2.line(line_image,(prev_x2_r,prev_y2_r),(x1,y1),(0,255,0),10)
                    prev_x2_r = x1
                    prev_y2_r = y1

                    end_x2_r  = x2
                    end_y2_r  = y2

                else:
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
                    prev_x2_r = x2
                    prev_y2_r = y2

            elif p >= 0.2 and p < 0.6:
                if x1 != prev_x2_l:
                    cv2.line(line_image,(prev_x2_l,prev_y2_l),(x1,y1),(0,255,0),10)
                    prev_x2_l = x1
                    prev_y2_l = y1

                    if x2 > end_x2_l:
                        end_x2_l  = x2
                        end_y2_l  = y2
                else:
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
                    prev_x2_l = x2
                    prev_y2_l = y2

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hough_lines1(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines1(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)
```

## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    
    # Define our parameters for Canny and apply
    low_threshold = 100
    high_threshold = 150
    kernel_size = 5
    
    # Convert image to grayscale
    gray = grayscale(image)
    
    # Get the image verticies
    
    imshape = gray.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    
    # Smooth the gray image
    blur_gray = gaussian_blur(gray, kernel_size)
    
    # Compute the edges
    edges = canny( blur_gray, low_threshold, high_threshold)
    
    # Mask
    mask = region_of_interest(edges, vertices)

    #Hough transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 2  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    
    lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)
    
    result = weighted_img(lines, image, alpha=0.8, beta=1., gamma=0.)
    
    return result
```


```python
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video white.mp4
    [MoviePy] Writing video white.mp4


    100%|█████████▉| 221/222 [00:04<00:00, 51.90it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: white.mp4 
    
    CPU times: user 2.08 s, sys: 160 ms, total: 2.24 s
    Wall time: 4.91 s

**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
def process_image1(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    kernel_size = 5
    
    # Convert image to grayscale
    gray = grayscale(image)
    
    # Get the image verticies
    
    imshape = gray.shape
    vertices = np.array([[(190,520),(430, 330), (600, 330), (830,520)]], dtype=np.int32)
    
    # Smooth the gray image
    blur_gray = gaussian_blur(gray, kernel_size)
    
    # Compute the edges
    edges = canny( blur_gray, low_threshold, high_threshold)
    
    # Mask
    mask = region_of_interest(edges, vertices)
    
    #Hough transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 60 #minimum number of pixels making up a line
    max_line_gap = 40    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)
    
    result = weighted_img(lines, image, alpha=0.8, beta=1., gamma=0.)
    
    return result

```


```python
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image1)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video yellow.mp4
    [MoviePy] Writing video yellow.mp4


    100%|█████████▉| 681/682 [00:14<00:00, 47.20it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: yellow.mp4 
    
    CPU times: user 6.97 s, sys: 636 ms, total: 7.61 s
    Wall time: 15.1 s



```
```





<video width="460" height="440" controls>
  <source src="yellow.mp4">
</video>




## Reflections

Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?

Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!


## Submission

If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.


### My Reflections:

#### This project gave me a great insight into the applications of different computer vision techniques for lane detection. My implementation would fail in the presence of shadows, and more curved roads as shown in the challenge question down below. Also this implementation requires the vertices to  be changed constantly, which means there must be a more robust method to automatically adjust the field-of-view. By employing more modern computer vision techniques that we are going to learn as the course progresses will help us improve, and make the lane detection system more robust. My optional challenge does not work.
