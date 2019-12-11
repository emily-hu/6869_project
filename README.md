# 6869_project

## Training Data Preprocessing
Find the script for training data preprocessing in /classification/data_collection/muscima_parsing.py
In the script, call any of the parse_xxx() functions to extract segmented images of a certain class from muscima and save them to the train and val folders in /classification/data_collection.

## Classification
Run the jupyter notebook in /classification/final_project.ipynb

## Input Preprocessing and Postprocessing
Have cv2, numpy, music21 installed by running `pip install opencv numpy music21`

Run the script in input_processing/omr.py

Functions are at the top, and you change the input image by specifying `imgName` and `ext` and putting the image in the `sourcePath` directory, which is currently sourceImages. Change `imgName` to "example#" to view examples 1-5 from the paper. 

Input images must be color and should have only a single staff line. 

Outputted midi and images during different processing steps are saved in the `savePath` directory, which is currently generatedImages. Segmented inverted images are saved in the boxes folder, as well as in the symbols folder (both the invertedSymbol_# and symbolInContext_#). The script returns the pitch prediction for each symbol with the numbering corresponding to the symbols in the symbols folder. 
