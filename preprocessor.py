# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:28:43 2019
"""


import csv
import xlrd     #If you need to install, use "pip install xlrd"
import os
import numpy as np
import xlsxwriter

# All start and end positions for required variables in the csv.
geneStart = 46
geneEnd = 197

micStart = 19   # 0 indexing
micEnd = 45

ageIndex = 200
sexIndex = 201
siteIndex = 2
stateIndex = 6

states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'DE', 'Dgo', 'FL', 'GA', 'IA', 'IL', 'IN', 'Jaliso', 'KS', 'KY', 'LA', 
          'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'NC', 'ND', 'NE', 'New South Wales', 'NJ', 'NM', 'NY', 'OH', 'OR', 'PA', 
          'QLD', 'TN', 'TX', 'UT', 'VA', 'VIC', 'VT', 'WA', 'Western Australia', 'WI' ]

siteCodes = [2, 3, 4, 12, 13, 15, 17, 21, 24, 25, 27, 30, 39, 40, 43, 46, 48, 49, 51, 52, 57, 62, 63, 64, 65, 66, 68, 69, 75, 81, 
             86, 89, 90, 91, 96, 97, 100, 101, 102, 106, 107, 112, 113, 115, 116, 117, 120, 122, 126, 127, 129, 130, 131, 133, 134, 
             136, 137, 138, 139, 146, 148, 149, 150, 203, 215, 219, 229, 260, 262, 263, 277, 279, 280, 281, 283, 302, 303, 307, 317,
             329, 333, 336, 342, 346, 347, 349, 361, 364, 365, 370, 371, 376, 377, 379, 380, 404, 410, 411, 412, 413, 417, 420, 422, 
             425, 426, 433, 437, 442, 443, 448, 452, 453, 454, 455, 456, 457, 460, 461, 462, 464, 467, 468, 469, 470, 471, 472, 473, 
             475, 476, 477, 480, 481, 601, 603, 605, 606, 614, 616, 703, 704, 721, 726, 728, 732, 734, 739, 742, 760, 789, 792, 794, 
             800, 806, 810, 812, 814, 815 ]

#The path to Required Information for Analysis.xlsx
reqInformationPath = os.getcwd() + "/Required Information for Analysis.xlsx"

#To debug file
debug = False



def load_dataset(d, rowNum=None, predict=False, export=False):
    """
    Author Cory Kromer-Edwards
    Edits by: Andrew West
    The main method to preprocessing. This method will open the dataset csv
    file, and send each row to the preprocessing steps.
    """
    debug = d
    X = []
    Y = []
    
    with open('dataset.csv') as csvFile:
        with xlsxwriter.Workbook('formated data.xlsx') as workbook:
            formatted = workbook.add_worksheet()
            csvReader = csv.reader(csvFile, delimiter=',')
            lineCount = 0
            formatedLines = 1
            (esblNames, carbaNames, micBreakpoints, betaDrugs) = getRequiredInfo()
            for row in csvReader:
                if lineCount == 0:
                    (isolateNames, drugNames) = getNames(row)
                    formatted.write(0, 0, "Site")
                    formatted.write(0, 1, "State")
                    formatted.write(0, 2, "Gender")
                    formatted.write(0, 3, "Age")
                    formatted.write(0, 4, "MIC num undefined")
                    formatted.write(0, 5, "MIC num R")
                    formatted.write(0, 6, "MIC num I")
                    formatted.write(0, 7, "MIC num S")
                    if debug:
                        print(f'Column names are:\n{", ".join(row)}')
                    lineCount += 1
                elif lineCount > 5 and debug:
                    break
                else:
                    if testRow(row):
                        
                        if rowNum and rowNum == lineCount:
                            X = []
                            Y = []
                      
                        #Get all data for row.
                        isolate = getGenes(row)
                        (convertedEsblCarba, numCategory) = convertEsblCarba(isolateNames, esblNames, carbaNames, isolate)
                        (mics, actualMics) = getMicValues(row, micBreakpoints, betaDrugs, drugNames)

                        age = getAgeBin(row)
                        sex = 0 if row[sexIndex] == "M" else 1
                        site = siteCodes.index(int(row[siteIndex]))
                        state = states.index(row[stateIndex]) if row[stateIndex] != "" else -1
                        
                        if debug or (rowNum and rowNum == lineCount):
                            print("\nRow number " + str(lineCount))
                            print("Genes: ", isolate)
                            print("ESBL, Carba: ", convertedEsblCarba, " - ", ["ESBL", "Carba.", "both", "neither"][convertedEsblCarba])
                            print("Num genes ESBL or Carba: ", str(numCategory))
                            print("MICs: ", mics, " - ", [["NA", "R", "I", "S"][mic] + "/" + str(drugId) for mic, drugId in mics])
                            print("Actual MIC values: [", [str(actualMic) + "," for actualMic in actualMics], "]")
                            print("Age:", str(age) + "; Sex: ", str(row[sexIndex]) + "; Site: ", str(site) + "; State: ", state)
                        
                        if export:
                            
                            formatted.write(formatedLines, 0, site)
                            formatted.write(formatedLines, 1, state)
                            formatted.write(formatedLines, 2, sex)
                            formatted.write(formatedLines, 3, age)
                            undef = 0
                            R = 0
                            I = 0
                            S = 0
                            for i in range(len(mics)):
                                if mics[i][0] == 0:
                                    undef += 1
                                elif mics[i][0] == 1:
                                    R += 1
                                elif mics[i][0] == 2:
                                    I += 1
                                elif mics[i][0] == 3:
                                    S += 1
                              
                            formatted.write(formatedLines, 4, undef)
                            formatted.write(formatedLines, 5, R)
                            formatted.write(formatedLines, 6, I)
                            formatted.write(formatedLines, 7, S)
                                
                            formatedLines += 1
                          
                        else:
                            #Append X and Y for dataset
                            for i in range(len(mics)):
                                X.append([convertedEsblCarba, mics[i][1], numCategory, site, age])
                                Y.append(mics[i][0])
                                
                            if rowNum and rowNum == lineCount:
                                return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32))
                            
                        
                    lineCount += 1
                
        print(f'Processed {lineCount} lines.')
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32))

def main():
    load_dataset(False, export=True)
    
        
def testRow(row):
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    This method will take in a row and test all needed values to see if there is
    something valid in that position. If all required values have are not empty,
    then the row is valid. The method will return True iff the row is valid, 
    false otherwise.
    """
    #Make sure the row has all genes available.
    for i in range(geneStart, geneEnd):
        if row[i] == "":
            return False
          
    if row[ageIndex] == "":
        return False
      
    if row[sexIndex] == "":
        return False
      
    if row[siteIndex] == "":
        return False

    return True
  
def getNames(row):
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    Gets the gene names that then form an isolate, and drug names for MIC values. It returns lists of Stirngs.
    """
    isolate = []
    drugNames = []
    
    for i in range(geneStart, geneEnd):
        isolate.append(row[i])
        
    for i in range(micStart, micEnd):
        drugNames.append(row[i])
        
    return (isolate, drugNames)
  
def getAgeBin(row):
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    Takes the age of a subject and places them in the correct bin.
    """
    age = int(row[ageIndex])
        
    #Pick 1 of 20 bins
    if age >= 0 and age < 5:
      return 0
    elif  age >= 5 and age < 10:
      return 1
    elif  age >= 10 and age < 15:
      return 2
    elif  age >= 15 and age < 20:
      return 3
    elif  age >= 20 and age < 25:
      return 4
    elif  age >= 25 and age < 30:
      return 5
    elif  age >= 30 and age < 35:
      return 6
    elif  age >= 35 and age < 40:
      return 7
    elif  age >= 40 and age < 45:
      return 8
    elif  age >= 45 and age < 50:
      return 9
    elif  age >= 50 and age < 55:
      return 10
    elif  age >= 55 and age < 60:
      return 11
    elif  age >= 60 and age < 65:
      return 12
    elif  age >= 65 and age < 70:
      return 13
    elif  age >= 70 and age < 75:
      return 14
    elif  age >= 75 and age < 80:
      return 15
    elif  age >= 80 and age < 85:
      return 16
    elif  age >= 85 and age < 90:
      return 17
    elif  age >= 90 and age < 95:
      return 18
    elif  age >= 95 and age < 100:
      return 19
    else:
      return -1
  
def getRequiredInfo():
    """
    Author: Cory Kromer-Edwards
    Edits by: Andrew West
    Gets all required information from the Required info excel sheet. This would 
    include the ESBL group, carab. group, and MIC break points.
    """
    if debug:
        print(reqInformationPath + "\n")
        
    esbl = set()
    carba = set()
    betaDrugs = set()
    mic = dict()
    
    with xlrd.open_workbook((reqInformationPath)) as wb:
        sheet_0 = wb.sheet_by_index(0)
        sheet_2 = wb.sheet_by_index(2)
        sheet_3 = wb.sheet_by_index(3)
    
        #range is (row start, row end) of the excel sheet, and it starts at 0.
        for i in range(1, 20):
            betaDrugs.add(sheet_0.cell_value(i, 1))
            
        for i in range(1, 48):
            mic[sheet_3.cell_value(i, 0)] = (sheet_3.cell_value(i, 1), sheet_3.cell_value(i, 2))
          
        for i in range(1, 424): 
            esbl.add(sheet_2.cell_value(i, 0))
                    
        for i in range(1, 585): 
            carba.add(sheet_2.cell_value(i, 1))
                
    return (esbl, carba, mic, betaDrugs)
  
  
def convertEsblCarba(isolateNames, esbl, carba, isolate):
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    Takes in the names of the genes, sets with the genes that are esbl and 
    carba., and the [0, 1] values of the genes. It then sorts through and
    checks if the genes that are 1 are part of the esbl or carba. categories,
    and returns 0, 1, 2, or 3 accordingly.
    """
    esblBool = False
    carbaBool = False
    numGenesInGroup = 0
    for i in range(len(isolate)):
        if isolate[i] == 1:
            if isolateNames[i] in esbl:
                numGenesInGroup += 1
                esblBool = True
                if debug:
                    print(isolateNames[i])
                
            if isolateNames[i] in carba:
                numGenesInGroup += 1
                carbaBool = True
                if debug:
                    print(isolateNames[i])
                    
    if not esblBool and not carbaBool:
        return (3, numGenesInGroup)  #Neither
    elif esblBool and carbaBool:
        return (2, numGenesInGroup)  #Both
    elif esblBool:
      return (0, numGenesInGroup)    #ESBL
    else:
      return (1, numGenesInGroup)    #Carba.

def getGenes(row):
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    getIsotope takes in the row from the csv, and extracts the genes from
    the row. It then converts their neg or pos value to either a 0 or 1.
    If a value that is not either one of those appears then it will throw
    a ValueError.
    """
    genes = []
    for i in range(geneStart, geneEnd):
        if row[i] == "neg":
            genes.append(0)
        elif row[i] == "pos":
            genes.append(1)
        else:
            raise ValueError("Gene is not of type 'neg' or 'pos', but is {row[i]}")
    
    return genes

def getMicValues(row, micBreakpoints, betaDrugs, drugNames):
    """
    Author: Andrew West
    Edits by: Cory Kromer-Edwards
    getMicValues takes in the row from the csv, and extracts the mic values from
    the row. It then comapres their value value to the breakpoints for that MIC.
    It returns 3 if less than or equal to the lower breakpoint, 2 if in between
    the breakpoints, and 1 if greater than or equal to the upper breakpoint
    """
    mic = []
    actualMic = []
    for i in range(micStart, micEnd):
        if row[i] == '':
            continue
          
        micValue = float(row[i])
        
        if drugNames[i - micStart] not in betaDrugs:
            mic.append((0, i - micStart)) #NA
        elif micValue <= micBreakpoints[drugNames[i - micStart]][0]:
            mic.append((3, i - micStart)) #S
        elif micValue >= micBreakpoints[drugNames[i - micStart]][1]:
            mic.append((1, i - micStart)) #R
        else:
            mic.append((2, i - micStart)) #I
            
        actualMic.append(micValue) #math.log(micValue, 2))
    return (mic, actualMic)




if __name__ == '__main__':
    main()
