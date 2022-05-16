#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2

color = [255,0,0] # BGR 순서 ; 파란색
pixel = np.uint8([[color]]) # 한 픽셀로 구성된 이미지로 변환

# BGR -> HSV
hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
print(hsv, 'shape:', hsv.shape )

# 픽셀값만 가져오기 
hsv = hsv[0][0]

print("bgr: ", color)
print("hsv: ", hsv) # +_ 10


# In[2]:


print(pixel)


# In[5]:


import numpy as np
import cv2

img_color = cv2.imread('test_img.png') # 이미지 파일을 컬러로 불러옴
print('shape: ', img_color.shape)

height, width = img_color.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환

lower_blue = (120-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (120+10, 255, 255)
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색

# 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득
img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)

cv2.imshow('img_origin', img_color)
cv2.imshow('img_mask', img_mask)
cv2.imshow('img_color', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


print(img_hsv.shape)


# In[13]:


from PIL import Image

im = Image.open('test_img.png')
pix = np.array(im)


# In[19]:


im_x = im.size[0]
im_y = im.size[1]

print(im_x, im_y)


# In[3]:


from PIL import Image
import numpy as np
import cv2

im_name = input("검출할 사진의 이름을 입력해 주세요. >> ")

full_name = im_name+".png"

im = Image.open(full_name)
pix = np.array(im)

image = cv2.imread('test_img.png')

im_x = im.size[1]
im_y = im.size[0]

print("좌표 영역 끝값은 ", im_x, im_y)

# 색상 지정
find_color = [0, 0, 0]
find_color1 = [18,0,253]
find_color2 = [255, 127, 39]
find_color3 = [237, 28, 36]

chose_color = int(input("어떤 색을 찾을까요? 1. 파랑, 2. 주황, 3. 빨강 (숫자를 입력해 주세요)>>"))

if chose_color == 1:
    find_color = find_color1
elif chose_color == 2:
    find_color = find_color2
elif chose_color == 3:
    find_color = find_color3
else :
    print("에러를 입력했습니다.")
    

# 색상 범위 설정
high_color = [find_color[0]+10, find_color[1]+10, find_color[2]+10]
low_color = [find_color[0]-10, find_color[1]-10, find_color[2]-10]

if low_color[1] < 0:
    low_color[1] = 0
    print("low_color 값 조정중...", low_color[1])
    
print("하이컬러는 ", high_color, " 로우 컬러는 ", low_color, "입니다.")

middle_x = 0
middle_y = 0

if (im_x/2)%2 == 0:
    middle_x = (im_x/2)
else:
    middle_x = (im_x//2)

if (im_y/2)%2 == 0:
    middle_y = (im_y/2)
else:
    middle_y = (im_y//2)
    
# 색상 인식 좌표범위 저장값
axis_x = [im_x, 0]
axis_y = [im_y, 0]

# for : image x와 y 좌표전체를 반복시행 ex) (1, 1), (1, 2), (1, 3)...
for x in range(im_x):
    for y in range(im_y):
        # 현재 for문으로 돌리는 좌표 픽셀값을 확인
        pix_RGB = pix[x][y]
        #print("현재 점검 좌표의 픽셀 값은 ", pix_RGB)
        
        # if : 현재 for문 위치 픽셀 좌표의 RGB값이 color범위값 사이에 있다면
        if pix_RGB[0] >= low_color[0] and pix_RGB[1] >= low_color[1] and pix_RGB[2] >= low_color[2] and pix_RGB[0] <= high_color[0] and pix_RGB[1] <= high_color[1] and pix_RGB[2] <= high_color[2]:
            tmp_x = x
            tmp_y = y
            #print("tmp_x, y 는", tmp_x, tmp_y)
            
            # tmp_x와 y값을 범위 값에 저장
            if tmp_x < axis_x[0]:
                axis_x[0] = tmp_x
                print("x의 최소값이 바뀌었습니다!")
                
            elif tmp_x > axis_x[1]:
                axis_x[1] = tmp_x
                print("x의 최대값이 바뀌었습니다!")
                
            elif tmp_y < axis_y[0]:
                axis_y[0] = tmp_y
                print("y의 최소값이 바뀌었습니다!")
                
            elif tmp_y > axis_y[1]:
                axis_y[1] = tmp_y
                print("x의 최대값이 바뀌었습니다!")


print("픽셀_RGB값은 ", pix_RGB)
print("좌표축_x 범위는 ", axis_x[0], axis_x[1])
print("좌표축_y 범위는 ", axis_y[0], axis_y[1])

imageRectangle = image.copy()
cv2.rectangle(imageRectangle, 
            (axis_x[0],axis_y[0]),
            (axis_x[1],axis_y[1]),
            (0,0,255),
            thickness=5, 
            lineType=cv2.LINE_AA)  
cv2.imshow("image", imageRectangle) 

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


print("좌표축_x 범위는 ", axis_x[0], axis_x[1])
print("좌표축_y 범위는 ", axis_y[0], axis_y[1])


# In[58]:


print(find_color)


# In[8]:


print(low_color)
print(high_color)


# In[47]:


print(pix_RGB[0] > low_color[0], pix_RGB[0] < high_color[0])


# In[4]:


print("컬러 최대값은 ", high_color)
print("컬러 색상값은 ", [18, 0, 253])


# In[5]:


pix_RGB < high_color


# In[7]:


pix_RGB[0] > low_color[0]


# In[1]:


from PIL import Image
import numpy as np
import cv2

im = Image.open('test_img.png')
pix = np.array(im)

image = cv2.imread('test_img.png')

im_x = im.size[1]
im_y = im.size[0]

print("좌표 영역 끝값은 ", im_x, im_y)

# 색상 인식 좌표범위 저장값
axis_x = [(im_x/2), (im_x/2)]
axis_y = [(im_y/2), (im_y/2)]


# In[3]:


print("axis_x의 값은 ", axis_x)
print("axis_y의 값은 ", axis_y)


# In[31]:


date()


# In[32]:


time()


# In[33]:


time(10)


# In[34]:


help(time)


# In[35]:


import time

secs = time.time()
print(secs)


# In[36]:


import time

tm = time.gmtime(1575142526.500323)
print(tm)


# In[37]:


import time

tm = time.localtime(1575142526.500323)
print("year:", tm.tm_year)
print("month:", tm.tm_mon)
print("day:", tm.tm_mday)
print("hour:", tm.tm_hour)
print("minute:", tm.tm_min)
print("second:", tm.tm_sec)


# In[67]:


from PIL import Image
import numpy as np
import cv2

im_name = input("검출할 사진의 이름을 입력해 주세요. >> ")

if im_name == "":
    im_name = "test_img"

full_name = im_name+".png"

im = Image.open(full_name)
pix = np.array(im)

image = cv2.imread('test_img.png')

im_x = im.size[0]
im_y = im.size[1]

#print("좌표 영역 끝값은 ", im_x, im_y)

# 색상 지정
find_color = [0, 0, 0]
find_color1 = [18,0,253]
find_color2 = [255, 127, 39]
find_color3 = [237, 28, 36]

chose_color = int(input("어떤 색을 찾을까요? 1. 파랑, 2. 주황, 3. 빨강 (숫자를 입력해 주세요)>>"))

if chose_color == 1:
    find_color = find_color1
elif chose_color == 2:
    find_color = find_color2
elif chose_color == 3:
    find_color = find_color3
else :
    print("에러를 입력했습니다.")
    

# 색상 범위 설정
high_color = [find_color[0]+10, find_color[1]+10, find_color[2]+10]
low_color = [find_color[0]-10, find_color[1]-10, find_color[2]-10]

if low_color[1] < 0:
    low_color[1] = 0
    print("low_color 값 조정중...", low_color[1])
    
#print("하이컬러는 ", high_color, " 로우 컬러는 ", low_color, "입니다.")
    
# 색상 영역에 포함될시 axis에 저장
axis = []
axis_len = 0

# 최종 좌표 저장위치, 디폴트는 최소값이 좌표끝값 / 최대값이 0
axis_x = [im_x, 0]
axis_y = [im_y, 0]


# for : image x와 y 좌표전체를 반복시행 ex) (1, 1), (1, 2), (1, 3)...
for x in range(im_x):
    for y in range(im_y):
        # 현재 for문으로 돌리는 좌표 픽셀값을 확인
        pix_RGB = pix[y][x]
        #print("현재 점검 좌표의 픽셀 값은 ", pix_RGB)
        
        # if : 현재 for문 위치 픽셀 좌표의 RGB값이 color범위값 사이에 있다면
        if pix_RGB[0] >= low_color[0] and pix_RGB[1] >= low_color[1] and pix_RGB[2] >= low_color[2] and pix_RGB[0] <= high_color[0] and pix_RGB[1] <= high_color[1] and pix_RGB[2] <= high_color[2]:
            axis.append([x, y])
            
#print("axis 값 출력 >>")
#print(axis)
for tmp_axis in axis:
    # 최소 위치x 찾기
    if tmp_axis[0] < axis_x[0]:
        axis_x[0] = tmp_axis[0]
        #print("최소x 변환 bool 체크 : ", bool(tmp_axis[0]==axis_x[0]), "최대 x는 ", axis_x[0], " 와 ", tmp_axis[0])
    # 최소 위치y 찾기
    if tmp_axis[1] < axis_y[0]:
        axis_y[0] = tmp_axis[1]
        #print("최소y 변환 bool 체크 : ", bool(tmp_axis[1]==axis_y[0]), "최대 y는 ", axis_y[1], " 와 ", tmp_axis[1])
                
    # 최대 위치x 찾기
    if tmp_axis[0] > axis_x[1]:
        axis_x[1] = tmp_axis[0]
        #print("최대x 변환 bool 체크 : ", bool(tmp_axis[0]==axis_x[1]), "최대 x는 ", axis_x[1], " 와 ", tmp_axis[0])

    # 최대 위치y  찾기
    if tmp_axis[1] > axis_y[1]:
        axis_y[1] = tmp_axis[1]
        #print("최대y 변환 bool 체크 : ", bool(tmp_axis[1]==axis_y[1]), "최대 y는 ", axis_y[1], " 와 ", tmp_axis[1])

#print("픽셀_RGB값은 ", pix_RGB)
#print("좌표축_x 범위는 ", axis_x[0], axis_x[1])
#print("좌표축_y 범위는 ", axis_y[0], axis_y[1])

imageRectangle = image.copy()
cv2.rectangle(imageRectangle, 
            (axis_x[0], axis_y[0]),
            (axis_x[1], axis_y[1]),
            (0,0,255),
            thickness=5, 
            lineType=cv2.LINE_AA)  
cv2.imshow("image", imageRectangle) 

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




