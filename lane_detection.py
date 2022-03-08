import cv2 
import numpy as np 
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('detected_lanes.mp4', fourcc, 20.0, (640,480))
def draw_region(image,regions):
    img=np.zeros_like(image)
    mask=cv2.fillPoly(img,[regions],(255,255,255))
    res=cv2.bitwise_and(image,mask)
    return res,mask
def draw_line(image,lines):
    img=image.copy()
    mask=np.zeros_like(img)
    for line in lines:
        for x1,y1,x2,y2 in line:
        #ROI=np.array([(x1,y1),((x1+x2)/2,(y2-y1)/2),(x2,y2)],np.int32)
        #lanes=cv2.fillPoly(mask,[ROI],(255,255,255))
            lanes=cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5,cv2.LINE_AA)
    #lanes=cv2.addWeighted(img,0.8,mask,1,0.0)
    return lanes

order=[0,3,1,2]
def steer_line(image,lines):
    theta=0
    img=image.copy()
    for line in lines:
        for x1,y1,x2,y2 in line:
            theta=theta+np.arctan2((y2-y1),(x2-x1))
    threshold=1.2
    print(theta)
    if theta>threshold:
        return "left"
        #return theta
           #assist=cv2.putText(img,"right",(100,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,40,(0,255,0),thickness=3)
    elif theta<-threshold:
        return "right"
        #return theta
           #assist=cv2.putText(img,"left",(100,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,40,(0,255,0),thickness=3)
    elif abs(theta)<=threshold:
        return "straight"
        #return theta
           #assist=cv2.putText(img,"stright",(100,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,40,(0,255,0),thickness=3)
def poly_zone(image,lines):
    img=image.copy()
    h,w=img.shape[:2]
    mask=np.zeros_like(img)
    left_side=[]
    right_side=[]
    points=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            m=(y2-y1)/(x2-x1)
            c=y-m*x1
            if m<0:
                left_side.append((m,c))
            if m>0:
                right_side.append((m,c))
            for m,c in left_side:
                xx1=int(h-c)/m
                yn=int(h*0.7)
                xn1=(yn-c)/m
                ROI=np.array([(xx1,h),(xn1,yn),(xn1+100,yn),(xx1+100,h)],np.int32)
                mask=cv2.fillPoly(mask,[ROI],(0,255,255))
            for m,c in right_side:
                xx1=int(h-c)/m
                yn=int(h*0.75)
                xn1=(yn-c)/m
                ROI=np.array([(xx1,h),(xn1,yn),(xn1+100,yn),(xx1+100,h)],np.int32)
                mask=cv2.fillPoly(mask,[ROI],(0,255,255))
    im=cv2.addWeighted(img,1,mask,0.5,0)
    return im

video_dir='Lane detect test data.mp4'
video=cv2.VideoCapture(video_dir)
ret,frame1=video.read()
#bboxes=cv2.selectROI(frame1,False)
#x1,y1,x2,y2=bboxes
y=frame1.shape[0]
x=frame1.shape[1]
ROI=np.array([(0,y-100),(x/2,y/2-100),(x/2+40,y/2-100),(x-100,y-100)],np.int32)
#print(bboxes)
while (video.isOpened()): 
    ret,frame=video.read()
    h,w=frame.shape[:2]
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    #ret,gray=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    canny=cv2.Canny(gray,50,255,apertureSize=3)
    roi,mask=draw_region(canny,ROI)
    #cv2.imshow("roi",roi)
    lines=cv2.HoughLinesP(roi,1,np.pi/180,threshold=100, lines=np.array([]), minLineLength=10, maxLineGap=100)
    #print(lines)
    if lines is not None:
        lanes=draw_line(frame, lines)
        assist=steer_line(lanes,lines)
        #anes=poly_zone(lanes, lines)
        cv2.putText(lanes,assist,(int(x/2)-100,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,0),thickness=4)
        #out.write(lanes)
        cv2.imshow("lanes",lanes)
        cv2.imwrite("lanes.jpg",lanes)
    cv2.imwrite("canny_edge.jpg",canny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
#out.release()
cv2.destroyAllWindows()