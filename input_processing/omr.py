import cv2
import numpy as np
import math
from music21 import *

#Helper functions
def grayscaleToBw(img, whiteThreshold = 230):
	'''
	turns greyscale img with pixels 0-255 to either 0 or 255 based on threshold
	below the threshold is marked black (0)
	above and including threshold is marked white(255)
	returns new image
	'''
	height, width = img.shape
	bwImage = img.copy()

	for x in range(width):
		for y in range(height):
			color = img[y][x]
			if color < whiteThreshold:
				bwImage[y][x] = 0
			else:
				bwImage[y][x] = 255
	return bwImage

def getStaffHeight(staffLineWidth, staffSpaceWidth):
	# returns total height of a staff composed of 5 lines with 4 spaces between
	return 5 * staffLineWidth + 4 * staffSpaceWidth


# Core Function steps
def compute_skew(img):
	'''
	Returns (skew angle in degrees, new image with horizontal lines draw)
	'''
	acceptableAngleLowerBound = -45
	acceptableAngleUpperBound = 45

	lineImg = img.copy()

	# Convert the image to gray-scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray = grayscaleToBw(gray, whiteThreshold = 110) # in case need to convert to bw

	# Find the edges in the image using canny detector
	edges = cv2.Canny(gray, 50, 200, apertureSize=3) # can save this image to see

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
		# only consider lines that are horizontal-ish
		if angleDeg >= acceptableAngleLowerBound and angleDeg <= acceptableAngleUpperBound:
			angleSum += angleDeg
			numHorizLines += 1
			cv2.line(lineImg,(x1,y1),(x2,y2),(0,0,255),2)

	# average skew
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
	really only works if there is only one staff line of music

	where a bar line looks like 

	top
	|
	|
	|
	bot

	# https://pdfs.semanticscholar.org/f4f5/1cffaa1b6661e135aa3dedc26e5561e66578.pdf

	# imgRemoveFrom must be greyscale. this image used for returned imgs when provided

	returns verticalProjectionImg, potentialBarLineImage, BarlineImage, RemovedBarlineImage, barlines

	where barlines is a list of (start,end) inclusive start and end horizontal positions of bar lines in img

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
	longVerticalLines = [] # holds tuples (leftx, rightx) both inclusive
	
	markPossibleBarLines = img.copy()
	firstSeen = None
	for i in range(len(proj)):
		if proj[i] >= staffHeight - tolerance:
			# show the potential lines
			cv2.line(markPossibleBarLines,(i,0),(i,255),(0,0,255),2)
			if firstSeen is None:
				firstSeen = i
		else:
			if not firstSeen is None:
				longVerticalLines.append((firstSeen, i-1)) # shows inclusive
				firstSeen = None

	# elimate lines with note head on either side of it (since those are note stems, not bar line)
	# note head seen as vertical projection of atleast half staff line space 
	# for about staff line space to either left or right
	# barlines shouldn't have much on either side of them 
	barLines = []

	distanceToCheckAroundLine = max(1, math.ceil(noteSize)) # make sure checking at least some distance
	minAverageNoteHeight = noteSize/2

	for line in longVerticalLines:
		leftPosition, rightPosition = line 

		avgLineHeight = 0
		linesCounted = 0
		for i in range(leftPosition, rightPosition +1):
			avgLineHeight += proj[i]
			linesCounted += 1
		avgLineHeight =  avgLineHeight/linesCounted

		heightTolerance = staffSpaceWidth

		totLeft = 0
		leftCounted = 0
	
		# note don't check the positions of the vertical lines themselves --> this will bring the avg up when it shouldnt 
		for i in range(leftPosition - distanceToCheckAroundLine, leftPosition):
			if i < 0:
				break
			leftCounted += 1
			h = proj[i]
			if h < avgLineHeight - heightTolerance:
				totLeft += h

		totRight = 0
		rightCounted = 0
		for i in range(rightPosition + 1, rightPosition + distanceToCheckAroundLine + 1):
			if i >= width:
				break
			rightCounted += 1
			h = proj[i]
			if h < avgLineHeight - heightTolerance:
				totRight += h

		if leftCounted < distanceToCheckAroundLine or rightCounted < distanceToCheckAroundLine:
			barLines.append(line)
		else:
			# leftCounted, rightCounted nonzero at this point bc distanceToCheckAroundLine is at least 1
			leftAverage = totLeft/leftCounted
			rightAverage = totRight/rightCounted
			# no note on either side (not enough vertical projection)
			if leftAverage < minAverageNoteHeight and rightAverage < minAverageNoteHeight:
				barLines.append(line)

			# or lots of stuff on both sides means its probably not a note
			# but this has false positives in the case of a note head on the left and a beam and slur on the right
			# elif leftAverage >= minAverageNoteHeight and rightAverage >= minAverageNoteHeight:
			# 	barLines.append(line)

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
	Finds music symbols in a bw image and saves them in the "boxes" directory 

	takes in a b/w inage of music without staff lines (ideally without barlines, but it would segment those too)
	uses the grayImg if provided to show the bounding boxes

	
	Returns tuple (imgWithBoundingBoxes,
				   list of image tuple lists [(image, invertedImg (x,y,w,h)), (image, invertedImg,  (x,y,w,h)), ...])
	each list inside the inner list are beamed together

	return symbol lists in order from left to right in the image of the top left bounding box coord
	(note beamed notes are not necessarily returned in order)

	where image has bounding box that looks like
	(x,y)----*
	|        |
	|        | h
	|        |
	*--------*
	    w

	uses code from: https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python

	minArea = minimum area to include something as a bounding box, unless breakBeams is true and it is a box inside that
	maxArea = maximum area ""
	breakBeams = whether or not to try to break beamed notes and return those a list of smaller components instead
	invert = whether images saved in boxes directory should be inverted

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
						idx += 1
						smallX, smallY, smallW, smallH = smallCoord
						# need these coord in relation to whole image
						actualX = x + smallX 
						actualY = y + smallY
						invertedSmallRoi = 255 - smallRoi
						if invert:
							saveSmallRoi = invertedSmallRoi
						else:
							saveSmallRoi = smallRoi
						cv2.imwrite(savingDir + str(idx) + '.jpg', saveSmallRoi) #puts the images in this folder
						relativeX = actualX + smallX
						relativeY = actualY + smallY
						cv2.rectangle(boundingBoxesImg,(relativeX,relativeY),(relativeX + smallW, relativeY + smallH),(200,0,0),2)
						beamed.append((smallRoi, invertedSmallRoi, (relativeX, relativeY, smallW, smallH)))
					symbols.append((beamed,x))
				# when there are not smaller images, just use the whole image
				else:
					idx += 1
					invertedRoi = 255- roi
					if invert:
						saveRoi = invertedRoi
					else:
						saveRoi = roi
					cv2.imwrite(savingDir + str(idx) + '.jpg', saveRoi) #puts the images in this folder
					cv2.rectangle(boundingBoxesImg,(x,y),(x+w,y+h),(200,0,0),2)
					symbols.append(([(roi, invertedRoi, (x,y,w,h))],x))
			# when not breaking bream, using whole bounding box
			else:
				idx += 1
				invertedRoi = 255- roi
				if invert:
					saveRoi = invertedRoi
				else:
					saveRoi = roi
				cv2.imwrite(savingDir + str(idx) + '.jpg', saveRoi) #puts the images in this folder
				cv2.rectangle(boundingBoxesImg,(x,y),(x+w,y+h),(200,0,0),2)
				symbols.append(([(roi, invertedRoi, (x,y,w,h))],x))

	sortedSymbols = sorted(symbols, key=lambda symbol: symbol[1]) # put in order left to right of boxes left coord
	returnSymbols = [s for s,x in sortedSymbols]
	return boundingBoxesImg, returnSymbols

def findStems(img, staffLineWidth, staffSpaceWidth):
	'''
	img can be bw or greyscale (not color)
	returns image with stems marked and list of stem locations (x coordinates) and vertical proj of bw image inverted
	coord in form (x_left, x_right) inclusive
	'''
	staffHeight = getStaffHeight(staffLineWidth, staffSpaceWidth)
	minNoteStemHeight = 7 * staffHeight/12 # staffHeight/2, get accidentals too, 2 * staffHeight/3 , missed some notes
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
	isolating symbols beamed together in img
	img is bw or greyscale (not color)

	might not split beam correctly if notes are too close to each other

	returns stemlines marked image, chops marked image, list of (split images, their bounding boxes)
	'''
	height, width = img.shape

	markStemsImage, stems, vertProj = findStems(img, staffLineWidth, staffSpaceWidth)

	chopImage = img.copy()
	chops = [] # horizontal x coord

	# Find chops to segment between stems
	for i in range(len(stems)-1):
		leftLineLeft, leftLineRight = stems[i]
		rightLineLeft, rightLineRight = stems[i+1]
		# chop between stems
		chop = round((rightLineLeft + leftLineRight)/2)
		chops.append(chop)
		cv2.line(chopImage,(chop,0),(chop,255),(0,0,255),2)

	# now get the bounding boxes made by the chops
	segments = [] # (roi, (x,y,w,h))

	# y, h same for all bounding boxes
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

	# uncomment these to see the chops in the "smallerBoxes/" directory
	# idx = 0
	# for tinyImg, info in segments:
	# 	idx +=1 
	# 	cv2.imwrite("smallerBoxes/" + str(idx) + '.jpg', tinyImg) #puts the images in this folder

	return markStemsImage, chopImage, segments

def getNoteHead(img, staffLineWidth, staffSpaceWidth, yCoordInBiggerImg = 0):
	'''
	img is bw or greyscale and should only have one note inside

	returns the y coord of center of note head or best guess if no head is present
	when yCoordInBiggerImg provided, y coord of center of note in terms of coord for bigger img

	returns (center_y, imgWithCenterMarked)
	'''
	imgWithCenterMarked = img.copy()
	noteHeadTolerance  = (1/2) * staffSpaceWidth
	
	# check horizontal projects on the left and the right
	height, width = img.shape
	bwImg = grayscaleToBw(img)

	# Invert
	bwImg = 255 - bwImg

	# Change scale from 0-255 --> 0-1
	bwImg = bwImg/255

	# horizontal projection 
	proj = np.sum(bwImg,1)
	m = np.max(proj)
	

	# Uncomment to see the horizontal projection of the note in the file horizontalNoteProjection.jpg
	# horizProj = np.zeros((proj.shape[0],width))
	# # Draw a line for each row
	# for row in range(bwImg.shape[0]):
	#    cv2.line(horizProj, (0,row), (int(proj[row]*width/m),row), (255,255,255), 1)
	# # showing this for filled and non filled notes, and notes on top and bottom of stem could be informative
	# cv2.imwrite("horizontalNoteProjection.jpg", horizProj)

	# find note head (note the highs may not be continuous in the case of half notes)
	# notes are symmetric since curved circles so highest point in the center and equal lower amounts above and below
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
	globalNoteHeadCenter = yCoordInBiggerImg + noteHeadCenter 

	return globalNoteHeadCenter, imgWithCenterMarked

def getStaffLineSpacePositions(staffLines):
	'''
	finds the lines and spaces from the 5 lines positions 

	when passed in line positions from top to bottom, returned are top to bottom
	and vice versa

	returns a list of their y coord
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
	
	returns a list of their y coord AND copy of img with lines marking these positions drawn on

	when passed in line positions from top to bottom, returned are top to bottom
	and vice versa
	'''
	imgCopy = rotatedImg.copy()
	staffLineSpacePositions = getStaffLineSpacePositions(staffLines)
	for p in staffLineSpacePositions:
		cv2.line(imgCopy,(0,p),(width,p),(0,0,255),1)
	return staffLineSpacePositions, imgCopy

def getNoteFromPosition(staffLineSpacePositions, noteHeadCenterY, lowestClefNote = "e4", highestClefNote = "f5"):
	'''
	returns music21 note object corresponding to noteHeadCenterY position in the staff 
	which is described by lines and spaces y position from top to bottom in staffLineSpacePositions

	default lowestClefNote and highestClefNote are for treble clef
	bass clef would be g2 --> a4
	'''

	# corresponding lists of y positions and notes 
	cScale = scale.MajorScale('c')
	pitches = [str(p) for p in cScale.getPitches(lowestClefNote, highestClefNote)][::-1]
	staffLineSpacePositionsNp = np.asarray(staffLineSpacePositions)

	# find closest match
	idx = (np.abs(staffLineSpacePositionsNp - noteHeadCenterY)).argmin()
	pitch = pitches[idx]

	return note.Note(pitch)

def getMeasureNumberFromPosition(elementBoundingBox, barLines):
	'''
	returns the measure (0 indexed) the music symbol represented at position in elementBoundingBox belongs in
	Still needs implemented
	x,y,w,h of the element elementBoundBox 
	barLines is list of barline x coord in image from left to right
	''' 
	pass


# Choose Image
imgName = "snippet"
ext = ".png"
sourcePath = 'sourceImages/'
savePath = 'generatedImages/' 

# 1. Load Image
img = cv2.imread(sourcePath+imgName+ext, cv2.IMREAD_COLOR) 
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # optional
# bw = grayscaleToBw(gray) # optional
height, width, channels = img.shape

# 2. Orient Image
skew, linedImg = compute_skew(img)
cv2.imwrite(savePath + imgName + '_lines' + ext, linedImg)
rotatedImg = deskew(img, skew)
cv2.imwrite(savePath + imgName + '_oriented' + ext, rotatedImg)

# 3. Get Staffline Information (and remove)
stafflessImg, lines, lineWidth, spaceWidth = removeStaffLines(rotatedImg)
staffLinesSpacesLocs, staffLinesSpacesImg = drawStaffLineSpacePositions(lines, rotatedImg)
cv2.imwrite(savePath + imgName + 'staffLineSpacesPositions' + ext, staffLinesSpacesImg)
avgNoteHeadRadius = spaceWidth/2 # spaceWidth is the diameter
minSegmentArea = spaceWidth*spaceWidth
cv2.imwrite(savePath + imgName + '_staffless' + ext, stafflessImg)
stafflessBw = grayscaleToBw(stafflessImg)

# 4. Get Barline Information (and remove)
barLineResult = getBarlines(stafflessImg, lineWidth, spaceWidth) # or use stafflessBw
verticalProj, possBarLineImg, barlineMarkedImg, removedBarlineImg, barlines = barLineResult
cv2.imwrite(savePath + imgName + '_barlines' + ext, barlineMarkedImg)
cv2.imwrite(savePath + imgName + '_withoutBarlines' + ext, removedBarlineImg)
bwStafflessBarlessImg = grayscaleToBw(removedBarlineImg)

# 5. Symbol Segmentation 
# segmentedImg, symbols = isolateSymbols(bwStafflessBarlessImg, lineWidth, spaceWidth, grayImg = removedBarlineImg, minArea = minSegmentArea)
segmentedImg, symbols = isolateSymbols(bwStafflessBarlessImg, lineWidth, spaceWidth, minArea = minSegmentArea)
cv2.imwrite(savePath + imgName + '_segmented' + ext, segmentedImg)

# 6. Symbol Classification and Music Reconstruction 
count = 0
saveDir = "symbols/"
s = stream.Stream()
for beamedSymbols in symbols:
	for symbol in beamedSymbols:
		symbolImg, invertedSymbolImg, coord = symbol
		x,y,w,h = coord

		# Find pitch
		noteHeadCenter, noteHeadImg = getNoteHead(symbolImg, lineWidth, spaceWidth, yCoordInBiggerImg = y)
		symbolNote = getNoteFromPosition(staffLinesSpacesLocs, noteHeadCenter, lowestClefNote = "e4", highestClefNote = "f5")
		# symbolNote.show() # show the note
		print("Symbol " + str(count) + ": ", symbolNote.pitch)

		# Here classifying using NN would happen and the correct music object would be added to the stream
		s.append(symbolNote)

		# Save images in saveDir to compare pitches later
		imgCopy = img.copy()
		cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(200,0,0),2)
		cv2.imwrite(saveDir + "invertedSymbol_" + str(count) + ".jpg", invertedSymbolImg)
		cv2.imwrite(saveDir + "symbolInContext_" + str(count) + ".jpg", imgCopy)
		
		count += 1

# 7. Save as MIDI file
s.write("midi", savePath + imgName + ".mid")
