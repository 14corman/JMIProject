# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:28:43 2019
"""


import csv
import xlrd     #If you need to install, use "pip install xlrd"
import os
import numpy as np

# All start and end positions for required variables in the csv.
geneStart = 46
geneEnd = 197

micStart = 19   # 0 indexing
micEnd = 45
emptyMics = {20, 24, 28, 31, 35, 41}

#The path to Required Information for Analysis.xlsx
reqInformationPath = os.getcwd() + "/Required Information for Analysis.xlsx"

#To debug file
debug = False



def load_dataset(d):
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
        csvReader = csv.reader(csvFile, delimiter=',')
        lineCount = 0
        (esblNames, carbaNames, micBreakpoints, betaDrugs) = getRequiredInfo()
        for row in csvReader:
            if lineCount == 0:
                (isolateNames, drugNames) = getNames(row)
                if debug:
                    print(f'Column names are:\n{", ".join(row)}')
                lineCount += 1
            elif lineCount > 5 and debug:
                break
            else:
                if testRow(row):
                  
                    #Get all data for row.
                    isolate = getGenes(row)
                    convertedEsblCarba = convertEsblCarba(isolateNames, esblNames, carbaNames, isolate)
                    mics = getMicValues(row, micBreakpoints, betaDrugs, drugNames)
                    
                    #Append X and Y for dataset
                    for i in range(len(mics)):
                        X.append([*isolate, convertedEsblCarba])
                        Y.append(mics[i])
                    
                    
                    if debug:
                        print("\nRow number " + str(lineCount))
                        print("Genes: ", isolate)
                        print("ESBL, Carba: ", convertedEsblCarba, " - ", ["ESBL", "Carba.", "both", "neither"][convertedEsblCarba])
                        print("MICs: ", mics, " - ", [["NA", "R", "I", "S"][mic] for mic in mics])
                        
                    
                lineCount += 1
                
        print(f'Processed {lineCount} lines.')
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32))

def main():
    load_dataset(True)
    
        
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
          
    #Make sure the row has all MIC values available.
    for i in range(micStart, micEnd):
        if row[i] == "" and i not in emptyMics:
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
    mic = []
    
    wb = xlrd.open_workbook((reqInformationPath)) 
    sheet_0 = wb.sheet_by_index(0)
    sheet_2 = wb.sheet_by_index(2)
    sheet_3 = wb.sheet_by_index(3)

    #range is (row start, row end) of the excel sheet, and it starts at 0.
    for i in range(1, 20):
        betaDrugs.add(sheet_0.cell_value(i, 1))
        
    for i in range(1, 48):
        mic.append((sheet_3.cell_value(i, 1), sheet_3.cell_value(i, 2)))
      
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
    for i in range(len(isolate)):
        if isolate[i] == 1:
            if isolateNames[i] in esbl:
                esblBool = True
                if debug:
                    print(isolateNames[i])
                
            if isolateNames[i] in carba:
                carbaBool = True
                if debug:
                    print(isolateNames[i])
                
        if esblBool and carbaBool:
            break
    
    if not esblBool and not carbaBool:
        return 3  #Neither
    elif esblBool and carbaBool:
        return 2  #Both
    elif esblBool:
      return 0    #ESBL
    else:
      return 1    #Carba.

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
    for i in range(micStart, micEnd):
        if row[i] == '' and i not in emptyMics:
            raise ValueError("MicValue is not specified")
            continue
        if i in emptyMics:
            continue
          
        micValue = float(row[i])
        
        if drugNames[i - micStart] not in betaDrugs:
            mic.append(0) #NA
        elif micValue <= micBreakpoints[i][0]:
            mic.append(3) #S
        elif micValue >= micBreakpoints[i][1]:
            mic.append(1) #R
        else:
            mic.append(2) #I
    return mic




if __name__ == '__main__':
    main()
