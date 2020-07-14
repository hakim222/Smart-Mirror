import cv2,utils,time
import numpy as np
from scipy import signal
from PIL import ImageFont, ImageDraw, Image  
from datetime import datetime

#%% User Settings
use_prerecorded		= False
fs					= 30  # Sampling Frequency

#%% Parameters

font1 = ImageFont.truetype("Roboto-Regular.ttf", 80)
font2 = ImageFont.truetype("Roboto-Regular.ttf", 60)
font3 = ImageFont.truetype("Roboto-Regular.ttf", 35)

haar_cascade_path 	= "haarcascade_frontalface_default.xml"
face_cascade 		= cv2.CascadeClassifier(haar_cascade_path)
tracker 			= cv2.TrackerMOSSE_create()
cap 				= utils.RecordingReader() if use_prerecorded else cv2.VideoCapture(0) 


window				= 300 # Number of samples to use for every measurement
skin_vec            = [0.3841,0.5121,0.7682]
B,G,R               = 0,1,2

found_face 	            = False
initialized_tracker		= False
face_box            	= []
mean_colors             = []
timestamps 	            = []

mean_colors_resampled   = np.zeros((3,1))

bpm = 0

#%% Main loop

while True: 
	
	ret, frame = cap.read() 
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# Try to update face location using tracker		
	if found_face and initialized_tracker :
		print("Tracking")
		found_face,face_box = tracker.update(frame)
		if not found_face:
			print("Lost Face")
	
	# Try to detect new face		
	if not found_face:
		initialized_tracker = False
		print("Redetecing")
		faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
		found_face = len(faces) > 0

	# Reset tracker
	if found_face and not initialized_tracker:			
		face_box = faces[0]
		tracker = cv2.TrackerMOSSE_create()
		tracker.init(frame,tuple(face_box))			
		initialized_tracker = True

	# Measure face color
	if found_face:
		face = utils.crop_to_boundingbox(face_box,frame)
		if face.shape[0] > 0 and face.shape[1]>0:
			
			mean_colors += [face.mean(axis=0).mean(axis=0)] 
			timestamps  +=  [ret] if use_prerecorded else [time.time()]
			utils.draw_face_roi(face_box,frame)
			t = np.arange(timestamps[0],timestamps[-1],1/fs)
			mean_colors_resampled = np.zeros((3,t.shape[0]))
			
			for color in [B,G,R]:
				resampled = np.interp(t,timestamps,np.array(mean_colors)[:,color])
				mean_colors_resampled[color] = resampled

	# Perform chrominance method
	if mean_colors_resampled.shape[1] > window:

		col_c = np.zeros((3,window))
        
		for col in [B,G,R]:
			col_stride 	= mean_colors_resampled[col,-window:]# select last samples
			y_ACDC 		= signal.detrend(col_stride/np.mean(col_stride))
			col_c[col] 	= y_ACDC * skin_vec[col]
            
		X_chrom     = col_c[R]-col_c[G]
		Y_chrom     = col_c[R] + col_c[G] - 2* col_c[B]
		Xf          = utils.bandpass_filter(X_chrom) 
		Yf          = utils.bandpass_filter(Y_chrom)
		Nx          = np.std(Xf)
		Ny          = np.std(Yf)
		alpha_CHROM = Nx/Ny
        
		x_stride   				= Xf - alpha_CHROM*Yf
		amplitude 				= np.abs( np.fft.fft(x_stride,window)[:int(window/2+1)])
		normalized_amplitude 	= amplitude/amplitude.max() #  Normalized Amplitude
		
		frequencies = np.linspace(0,fs/2,int(window/2) + 1) * 60
		bpm_index = np.argmax(normalized_amplitude)
		bpm       = frequencies[bpm_index]
		snr       = utils.calculateSNR(normalized_amplitude,bpm_index)
		utils.put_snr_bpm_onframe(bpm,snr,frame)

	# cv2.imshow('Camera',frame) 
	
    # get the current time
	now = datetime.now()
	current_date = now.strftime("%d %b %Y")
	current_time = now.strftime("%H:%M %p")
	current_day = now.strftime("%A")

	# get bpm and temperature
	bpm = int(bpm)
	degree = u'\u00B0C'
	temperature = "29.5" + degree

	# Create a black image
	img = np.zeros((768,1366,3), np.uint8)
	
	# convert to pil for custom font and draw it on the black screen window
	pil_img = Image.fromarray(img)
	draw = ImageDraw.Draw(pil_img)
	draw.text((10, 50), current_day, font=font1)
	draw.text((25, 130), current_date, font=font3)
	draw.text((15, 180), current_time, font=font2)
	
	# positioning the bpm according to its digit number
	if bpm < 10:
		bpm = str(bpm)
		draw.text((1150, 120), bpm, font=font2)
		draw.text((1195, 120), " bpm", font=font2)
	elif bpm > 9 and bpm < 100:
		bpm = str(bpm)
		draw.text((1115, 120), bpm, font=font2)
		draw.text((1195, 120), " bpm", font=font2)
	else:
		bpm = str(bpm)
		draw.text((1080, 120), bpm, font=font2)
		draw.text((1195, 120), " bpm", font=font2)

	draw.text((1150, 180), temperature, font=font2)
	img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  

	# Make the display window fullscreen
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	
	# Display on screen
	cv2.imshow('frame', img)

	# Exit by pressing q key
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Close all windows after q key being pressed
cap.release() 
cv2.destroyAllWindows() 

#plt.figure()


# %%
