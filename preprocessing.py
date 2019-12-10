import cv2
import numpy as np
import math
from music21 import *

#Helper functions
def grayscaleToBw(img, whiteThreshold = 250):
	'''
	turns img with pixels 0-255 to either 0 or 255 based on threshold
	'''
	height, width = img.shape
	bwImage = img.copy()

	count = 0
	count2 = 0
	for x in range(width):
		for y in range(height):
			color = img[y][x]
			if color < whiteThreshold:
				count += 1
				bwImage[y][x] = 0
			else:
				count2 +=1 
				bwImage[y][x] = 255
	if whiteThreshold == 180:
		print("here: ", count, count2)
	return bwImage

def getStaffHeight(staffLineWidth, staffSpaceWidth):
	# 5 lines with 4 spaces between
	return 5 * staffLineWidth + 4 * staffSpaceWidth


# Core Function steps
def compute_skew(img):
	'''
	Returns skew, new image with horizontal lines draw
	'''
	acceptableAngleLowerBound = -45
	acceptableAngleUpperBound = 45
	lineImg = img.copy()
	# Convert the image to gray-scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray = grayscaleToBw(gray, whiteThreshold = 110) # in case need to convert to bw
	cv2.imwrite("somethingMaybe.png", gray)
	# Find the edges in the image using canny detector
	edges = cv2.Canny(gray, 50, 200, apertureSize=3)
	cv2.imwrite("edgesSkew.png", edges)
	# Detect points that form a line
	lines = cv2.HoughLines(edges,1,np.pi/180,400)

	if lines is None:
		return 0, lineImg

	# Draw lines on the image
	angleSum = 0.0 # in degrees
	numHorizLines = 0
	for line in lines:
		rho,theta = line[0]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		angleRad = np.arctan2(y2 - y1, x2 - x1) # in radians
		angleDeg = angleRad*180/np.pi # radians --> degrees
		if angleDeg >= acceptableAngleLowerBound and angleDeg <= acceptableAngleUpperBound:
			angleSum += angleDeg
			numHorizLines += 1
			cv2.line(lineImg,(x1,y1),(x2,y2),(0,0,255),2)

	# for line in lines: # use this if using HoughLinesP
	# 	x1, y1, x2, y2 = line[0]
	# 	cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
	# 	angle += np.arctan2(y2 - y1, x2 - x1)

	if numHorizLines == 0:
		skew = 0
	else:
		skew = angleSum/numHorizLines 
	return skew, lineImg

def deskew(img, angle):# rotate the image to deskew it
	'''
	returns a new rotated image
	'''
	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, skew, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return rotated

def removeStaffLines(img):
	'''
	creates grayscale image without staff lines
	returns tuple: that image, list of tuples of vertical position of lines inclusive (start,end), avg width of lines, avg space between lines
	lines lines tuples in order from top to bottom
	'''
	# Convert the image to gray-scale
	im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	noStaffLineImg = im.copy()
	height, width = im.shape

	# Invert
	im = 255 - im

	# Change scale from 0-255 --> 0-1
	im = im/255

	# Calculate horizontal projection
	proj = np.sum(im,1) # sum of grey pixels which is range 0 * 255
	m = np.max(proj)


	# Create output image same height as text, 500 px wide
	m = np.max(proj)
	result = np.zeros((proj.shape[0],width))
	# Draw a line for each row
	for row in range(im.shape[0]):
	   cv2.line(result, (0,row), (int(proj[row]*width/m),row), (255,255,255), 1)
	# # Save result
	# cv2.imwrite('result.png', result)
	# cv2.line(img,(0,46),(0,51),(0,0,255),2)
	# cv2.imwrite('result2.png', img)

	# find staff lines
	# .7 from magical music decoder (using .5)
	lineLocs = []
	percent_width = .2
	firstSeen = None
	for i in range(len(proj)):
		if proj[i] >= percent_width * width:
			# loc.append(i)
			if firstSeen is None:
				firstSeen = i
		else:
			if not firstSeen is None:
				lineLocs.append((firstSeen, i-1)) # shows inclusive
				firstSeen = None

	# find widths of lines
	totalLineWidths = 0
	for i in range(len(lineLocs)):
		top, bot = lineLocs[i]
		totalLineWidths += (bot-top)
	if len(lineLocs) == 0:
		avgLineWidth = 0
	else:
		avgLineWidth = math.ceil(totalLineWidths/len(lineLocs))
	bufferAmount = math.ceil(avgLineWidth/4)
	distanceCheckAroundLine = math.ceil(avgLineWidth/2)

	# find widths of spaces between lines
	avgSpaceWidth = 0
	if len(lineLocs) > 0:
		totalSpaceWidth = 0
		spacesCounted = 0
		for i in range(len(lineLocs) - 1):
			# dont consider the gap between staff lines 
			if (i+1)%5 != 0:
				spacesCounted += 1
				thisLineTop, thisLineBot = lineLocs[i]
				nextLineTop, nextLineBot = lineLocs[i+1]
				spaceWidth = nextLineBot - thisLineTop
				totalSpaceWidth += spaceWidth
		if spacesCounted == 0:
			avgSpaceWidth = 0
		else:
			avgSpaceWidth = totalSpaceWidth/spacesCounted


	# remove staff lines 
	grey_threshold = 50/255
	for x in range(width): 
		for line in lineLocs:
			top, bot = line # top is lower 
			shouldEliminateTop = False
			# check that avgLineWidth in both directions past the line 
			topStart = top-1-bufferAmount
			for u in range(topStart, topStart-distanceCheckAroundLine-1, -1):
				if u < 0:
					break
				if im[u][x] < grey_threshold:
					shouldEliminateTop = True
					break
			shouldEliminateBot = False
			botStart = bot+1+bufferAmount
			for l in range(botStart, botStart+distanceCheckAroundLine+1):
				if l >= height:
					break
				if im[l][x] < grey_threshold:
					shouldEliminateBot = True
					break
			if shouldEliminateTop and shouldEliminateBot:
				# currently, don't white out the buffer
				for y in range(top-1-bufferAmount, bot+bufferAmount+1):
					noStaffLineImg[y][x] = 255

	return noStaffLineImg, lineLocs, avgLineWidth, avgSpaceWidth

def getBarlines(img, staffLineWidth, staffSpaceWidth, imgRemoveFrom=None):
	'''	
	img is greyscale with staff lines removed 
	really only works if there is only one line of music

	where a bar line looks like 

	top
	|
	|
	|
	bot

	# https://pdfs.semanticscholar.org/f4f5/1cffaa1b6661e135aa3dedc26e5561e66578.pdf

	# imgRemoveFrom must be greyscale too 

	returns verticalProjectionImg, potentialBarLineImage, BarlineImage, RemovedBarlineImage, barlines

	where barlines is a list of (start,end) inclusive start and end horizontal positions of bar lines

	'''
	staffHeight = getStaffHeight(staffLineWidth, staffSpaceWidth)
	tolerance = staffSpaceWidth
	noteSize = staffSpaceWidth

	# Convert the image to gray-scale
	# im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	noBarLine = img.copy()
	height, width = img.shape
	bwImg = grayscaleToBw(img)

	# Invert
	bwImg = 255 - bwImg

	# Change scale from 0-255 --> 0-1
	bwImg = bwImg/255


	# Calculate vertical projection
	proj = np.sum(bwImg,0) # sum of grey pixels which is range 0 * 255
	m = np.max(proj)

	# Create output image same height as text, 500 px wide
	verticalProjection = np.zeros((height, proj.shape[0]))
	# Draw a line for each row
	for col in range(img.shape[1]):
		# start from height and go down 
		cv2.line(verticalProjection, (col,height), (col, height - int(proj[col]*height/m)), (255,255,255), 1)


	# find potential lines (long enough)
	longVerticalLines = []
	
	markPossibleBarLines = img.copy()
	firstSeen = None
	for i in range(len(proj)):
		# print(proj[i])
		if proj[i] >= staffHeight - tolerance:
			# show the potential lines
			cv2.line(markPossibleBarLines,(i,0),(i,255),(0,0,255),2)
			if firstSeen is None:
				firstSeen = i
		else:
			if not firstSeen is None:
				longVerticalLines.append((firstSeen, i-1)) # shows inclusive
				firstSeen = None

	# elimate lines with note head on either side of it (it's a note stem, not bar line)
	# note head seen as vertical projection of atleast half staff line space 
	# for about staff line space to either left or right
	barLines = []

	distanceToCheckAroundLine = max(1, math.ceil(noteSize)) # make sure checking at least some distance
	minAverageNoteHeight = noteSize/2

	for line in longVerticalLines:
		leftPosition, rightPosition = line 
		leftHasNoteHead = False
		rightHasNoteHead = True
		totLeft = 0
		leftCounted = 0
		# note dont' check the positions of the vertical lines themselves 
		# --> this will bring the avg up when it shouldnt 
		for i in range(leftPosition - distanceToCheckAroundLine, leftPosition):
			if i < 0:
				break
			leftCounted += 1
			totLeft += proj[i]


		totRight = 0
		rightCounted = 0
		for i in range(rightPosition + 1, rightPosition + distanceToCheckAroundLine + 1):
			if i >= width:
				break
			rightCounted += 1
			totRight += proj[i]

		if leftCounted < distanceToCheckAroundLine or rightCounted < distanceToCheckAroundLine:
			barLines.append(line)
		else:
			leftAverage = totLeft/leftCounted
			rightAverage = totRight/rightCounted
			# no note on either side (not enough vertical projection)
			if leftAverage < minAverageNoteHeight and rightAverage < minAverageNoteHeight:
				barLines.append(line)

	# show barlines and remove barlines
	removedBarlinesImg = img.copy()
	if not removedBarlinesImg is None:
		removedBarlinesImg = removedBarlinesImg

	barlinesMarked = img.copy()
	for line in barLines:
		left, right = line
		for i in range(left, right +1):
			cv2.line(barlinesMarked,(i,0),(i,255),(0,0,255),2)
			for y in range(height):
					removedBarlinesImg[y][i] = 255


	return verticalProjection, markPossibleBarLines, barlinesMarked, removedBarlinesImg, barLines

def isolateSymbols(img, staffLineWidth, staffSpaceWidth, *, grayImg = None, minArea = 0, maxArea = None, breakBeams = True, invert = True):
	'''
	takes in a b/w inage of music without staff lines
	uses the grayImg if provided to show the bounding boxes

	Finds music symbols
	Returns tuple (imgWithBoundingBoxes, list of image tuple lists [(image, (x,y,w,h)), (image, (x,y,w,h)), (image, (x,y,w,h))]
	each list inside the inner list are beamed together

	where image has bounding box that looks like
	(x,y)----*
	|        |
	|        | h
	|        |
	*--------*
	    w

	code from: https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python

	'''
	if not grayImg is None:
		boundingBoxesImg = grayImg.copy()
	else:
		boundingBoxesImg = img.copy()

	if maxArea is None:
		width,height = img.shape
		maxArea = .9 * width * height

	symbols = []

	savingDir = "boxes/"



	# contours, hierarchy = cv2.findContours(img,mode=1,method=cv2.CHAIN_APPROX_SIMPLE)[-2:] # only outer bounding boxes
	contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:] # see all children, even inside
	idx = 0

	for i in range(len(contours)):
		cnt = contours[i]
		x,y,w,h = cv2.boundingRect(cnt)
		area = w*h
		if not grayImg is None:
			roi = grayImg[y:y+h,x:x+w]
		else:
			roi=img[y:y+h,x:x+w]
		if area > minArea and (area < maxArea or len(contours) ==1): # allow the case where can't segment it to have big segment
			if breakBeams:
				stemsImage, chopImage, segments = breakBeam(roi, staffLineWidth, staffSpaceWidth)
				# use the smaller images if possible
				if len(segments) > 0:
					beamed = []
					for smallRoi, smallCoord in segments:
						print("segment: ", smallRoi)
						idx += 1
						smallX, smallY, smallW, smallH = smallCoord
						# need these coord in relation to whole image
						actualX = x + smallX 
						actualY = y + smallY
						if invert:
							smallRoi = 255 - smallRoi
						cv2.imwrite(savingDir + str(idx) + '.jpg', smallRoi) #puts the images in this folder
						cv2.rectangle(boundingBoxesImg,(actualX,actualY),(actualX+smallW,actualY+smallH),(200,0,0),2)
						beamed.append((smallRoi, (actualX, actualY, smallW, smallH)))
					symbols.append(beamed)
				# when there are not smaller images, just use the whole imagee
				else:
					idx += 1
					if invert:
						roi = 255- roi
					cv2.imwrite("roiiii.png", roi)
					cv2.imwrite(savingDir + str(idx) + '.jpg', roi) #puts the images in this folder
					cv2.rectangle(boundingBoxesImg,(x,y),(x+w,y+h),(200,0,0),2)
					symbols.append([(roi, (x,y,w,h))])
			# when not breaking, using whole image only
			else:
				idx += 1
				if invert:
					roi = 255- roi
				cv2.imwrite(savingDir + str(idx) + '.jpg', roi) #puts the images in this folder
				cv2.rectangle(boundingBoxesImg,(x,y),(x+w,y+h),(200,0,0),2)
				symbols.append([(roi, (x,y,w,h))])
	# cv2.imshow('img',boundingBoxesImg)
	# cv2.waitKey(0)  
	return boundingBoxesImg, symbols

def findStems(img, staffLineWidth, staffSpaceWidth):
	'''
	returns image with stems marked and list of stem locations (x coordiantes) and vertical proj of bw image inverted
	in form (x_left, x_right) inclusive
	'''
	staffHeight = getStaffHeight(staffLineWidth, staffSpaceWidth)
	# when minNoteStemHeight = staffHeight/2, get accidentals too
	# when minNoteStemHeight = 2 * staffHeight/3 , missed some notes
	minNoteStemHeight = 7 * staffHeight/12 
	minNoteHeight = 2*staffSpaceWidth/3

	# Convert the image to gray-scale
	# im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	noBarLine = img.copy()
	height, width = img.shape
	bwImg = grayscaleToBw(img)

	# Invert
	bwImg = 255 - bwImg

	# Change scale from 0-255 --> 0-1
	bwImg = bwImg/255


	# Calculate vertical projection
	proj = np.sum(bwImg,0) # sum of grey pixels which is range 0 * 255
	m = np.max(proj)


	# Find stems
	stems = []
	markStemsImage = img.copy()
	firstSeenStem = None

	for i in range(len(proj)):
		# find stems
		if proj[i] >= minNoteStemHeight:
			# show the potential lines
			cv2.line(markStemsImage,(i,0),(i,255),(0,0,255),2)
			if firstSeenStem is None:
				firstSeenStem = i
		else:
			if not firstSeenStem is None:
				stems.append((firstSeenStem, i-1)) # shows inclusive
				firstSeenStem = None

	return markStemsImage, stems, proj

def breakBeam(img, staffLineWidth, staffSpaceWidth):
	'''
	isolating symbols 

	could have problems if the notes are too close to each other... 

	returns stemlines marked, chopped image, list of bounding boxes and images
	'''
	height, width = img.shape

	markStemsImage, stems, vertProj = findStems(img, staffLineWidth, staffSpaceWidth)

	chopImage = img.copy()
	chops = []

	# Find chops to segment between stems
	for i in range(len(stems)-1):
		leftLineLeft, leftLineRight = stems[i]
		rightLineLeft, rightLineRight = stems[i+1]
		chop = round((rightLineLeft + leftLineRight)/2)
		chops.append(chop)
		cv2.line(chopImage,(chop,0),(chop,255),(0,0,255),2)


	segments = [] # (roi, (x,y,w,h))

	# same for all
	y = 0
	h = height

	leftX = 0

	for chopX in chops:
		x = leftX
		w = chopX - leftX

		roi = img[y:y+h,x:x+w]
		segments.append((roi, (x,y,w,h)))

		leftX = chopX

	# do for last segment
	if len(chops) > 0:
		x = chops[-1]
		w = width - x
		roi = img[y:y+h,x:x+w]
		segments.append((roi, (x,y,w,h)))

	idx = 0
	for tinyImg, info in segments:
		idx +=1 
		cv2.imwrite("smallerBoxes/" + str(idx) + '.jpg', tinyImg) #puts the images in this folder

	return markStemsImage, chopImage, segments

def getNoteHead(img, staffLineWidth, staffSpaceWidth, xCoordInBiggerImg = 0):
	'''
	img is bw or greyscale and can only have one note inside

	returns the x coord of center of note head or best guess if no head is present
	when xCoordInBiggerImg provided, describes top left x coord of img in the orig image

	returns (center_x, imgWithCenterMarked)
	'''
	imgWithCenterMarked = img.copy()
	noteHeadTolerance  = (3/4) * staffSpaceWidth
	

	# check horizontal projects on the left and the right
	height, width = img.shape
	bwImg = grayscaleToBw(img)

	# Invert
	bwImg = 255 - bwImg

	# Change scale from 0-255 --> 0-1
	bwImg = bwImg/255

	# horizontal projection 
	proj = np.sum(bwImg,1) # sum of grey pixels which is range 0 * 255
	m = np.max(proj)
	

	horizProj = np.zeros((proj.shape[0],width))
	# Draw a line for each row
	for row in range(bwImg.shape[0]):
	   cv2.line(horizProj, (0,row), (int(proj[row]*width/m),row), (255,255,255), 1)
	# showing this for filled and non filled notes, and notes on top and bottom of stem could be informative
	# cv2.imwrite("horizontalProject.jpg", horizProj)


	# find note head (note the highs may not be continuous in the case of half notes)
	# notes are symmetric
	noteLocs = []
	firstSeen = None
	lastSeen = None
	for i in range(len(proj)):
		if proj[i] >= staffSpaceWidth - noteHeadTolerance:
			if firstSeen is None:
				firstSeen = i
				lastSeen = i
			else:
				lastSeen = i

	if firstSeen is None:
		# best guess if not possible to set it
		noteHeadCenter = np.argmax(proj)
	else:
		noteHeadCenter = round((firstSeen + lastSeen)/2)

	cv2.line(imgWithCenterMarked, (0, noteHeadCenter), (width, noteHeadCenter), (0,0,255), 1)

	# adjust coordinate in terms of bigger global image
	globalNoteHeadCenter = noteHeadCenter + xCoordInBiggerImg

	return globalNoteHeadCenter, imgWithCenterMarked


def getStaffLineSpacePositions(staffLines):
	'''
	finds the lines and spaces from the 5 lines positions 

	when passed in line positions from top to bottom, returned are top to bottom
	and vice versa
	'''

	lineSpacePosition = []
	for i in range(len(staffLines)):
		lineStart, lineEnd = staffLines[i]
		lineMiddle = round((lineStart + lineEnd)/2)
		lineSpacePosition.append(lineMiddle)

		if i+1 < len(staffLines): # can you get the next line
			nextLineStart, nextLineEnd = staffLines[i+1]
			spaceMiddle = round((lineEnd+nextLineStart)/2)
			lineSpacePosition.append(spaceMiddle)

	return lineSpacePosition

def drawStaffLineSpacePositions(staffLines, img):
	'''
	finds the lines and spaces from the 5 lines positions 
	
	returns line and space positions image with them drawn on 

	when passed in line positions from top to bottom, returned are top to bottom
	and vice versa
	'''
	imgCopy = rotatedImg.copy()
	staffLineSpacePositions = getStaffLineSpacePositions(staffLines)
	for p in staffLineSpacePositions:
		cv2.line(imgCopy,(0,p),(width,p),(0,0,255),1)
	return staffLineSpacePositions, imgCopy

def getNoteFromPosition(staffLineSpacePositions, lowestClefNote = "e4", highestClefNote = "f5"):
	'''
	staffLineSpacePositions are from top to bottom 
	'''
	# bass clef is g2 --> a4
	cScale = scale.MajorScale('c')
	pitches = [str(p) for p in sc1.getPitches(lowestClefNote, highestClefNote)]
	staffLineSpacePositionsBotToTop = staffLineSpacePositions[::-1] # reverse
	idx = (np.abs(staffLineSpacePositionsBotToTop - noteHeadCenterX)).argmin()
	pitch = pitches[idx]
	return note.Note(pitch)


def getMeasureNumberFromPosition(elementBoundingBox, barLines):
	'''

	'''


# Read image 
imgName = "fromPaper"
ext = ".png"
sourcePath = 'sourceImages/'
savePath = 'generatedImages/'

img = cv2.imread(sourcePath+imgName+ext, cv2.IMREAD_COLOR) 
height, width, channels = img.shape

skew, linedImg = compute_skew(img)
cv2.imwrite(savePath + imgName + '_lines' + ext, linedImg)

rotatedImg = deskew(img, skew)
cv2.imwrite(savePath + imgName + '_oriented' + ext, rotatedImg)

stafflessImg, lines, lineWidth, spaceWidth = removeStaffLines(rotatedImg)
# staffLineSpacesPositions = getStaffLineSpacePositions(lines)
staffLines, staffLinesSpacesImg = drawStaffLineSpacePositions(lines, rotatedImg)
cv2.imwrite(savePath + imgName + 'staffLineSpacesPositions' + ext, staffLinesSpacesImg)



avgNoteHeadRadius = spaceWidth/2 # spaceWidth is the diameter
minSegmentArea = spaceWidth*spaceWidth
cv2.imwrite(savePath + imgName + '_staffless' + ext, stafflessImg)

barLineResult = getBarlines(stafflessImg, lineWidth, spaceWidth)
verticalProj, possBarLineImg, barlineMarkedImg, removedBarlineImg, barlines = barLineResult
cv2.imwrite(savePath + imgName + '_barlines' + ext, barlineMarkedImg)

bwStafflessBarlessImg = grayscaleToBw(removedBarlineImg)


# segmentedImg, symbols = isolateSymbols(bwStafflessBarlessImg, lineWidth, spaceWidth, grayImg = removedBarlineImg, minArea = minSegmentArea)
segmentedImg, symbols = isolateSymbols(bwStafflessBarlessImg, lineWidth, spaceWidth, minArea = minSegmentArea)
cv2.imwrite(savePath + imgName + '_segmented' + ext, segmentedImg)

# cv2.imwrite("itworked.png", symbols[0][0][0])
# cv2.imwrite("boxing.jpg", symbols[0][0])
# for box, posInfo in symbols:
# 	x,y,w,h = posInfo



# testImg = cv2.imread(sourcePath+"26.jpg", cv2.IMREAD_GRAYSCALE) 
# invertedTestImg = 255 - testImg
# markStemsImage, chopImage, segments = breakBeam(invertedTestImg, lineWidth, spaceWidth)
# cv2.imwrite("markStems26.jpg", markStemsImage)
# cv2.imwrite("chopImag26.jpg", chopImage)

# noteHeadCenter, imgWithCenterMarked = getNoteHead(invertedTestImg, lineWidth, spaceWidth)
# cv2.imwrite("surprise26.jpg", imgWithCenterMarked)

# getLineSpaceCenters(staffLines)
