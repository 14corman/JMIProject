# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:28:43 2019
"""


import csv
import xlrd     #If you need to install, use "pip install xlrd"
import os

# All start and end positions for required variables in the csv.
geneStart = 46
geneEnd = 197

#The path to Required Information for Analysis.xlsx
reqInformationPath = os.getcwd() + "/Required Information for Analysis.xlsx"

#To debug output
debug = True




def main():
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    The main method to preprocessing. This method will open the dataset csv
    file, and send each row to the preprocessing steps.
    """
    with open('dataset.csv') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        lineCount = 0
        (esblNames, carbaNames) = getEsblCarba()
        for row in csvReader:
            if lineCount == 0:
                isolateNames = getIsolate(row)
                if debug:
                    print(f'Column names are:\n{", ".join(row)}')
                lineCount += 1
            elif lineCount > 5 and debug:
                break
            else:
                if testRow(row):
                    isolate = getGenes(row)
                    convertedEsblCarba = convertEsblCarba(isolateNames, esblNames, carbaNames, isolate)
                    
                    if debug:
                        print("\nRow number " + str(lineCount))
                        print("Genes: ", isolate)
                        print("ESBL, Carba: ", convertedEsblCarba)
                        
                    
                lineCount += 1
                
        print(f'Processed {lineCount} lines.')
        
def testRow(row):
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    This method will take in a row and test all needed values to see if there is
    something valid in that position. If all required values have are not empty,
    then the row is valid. The method will return True iff the row is valid, 
    false otherwise.
    """
    for i in range(geneStart, geneEnd):
        if row[i] == "":
            return False

    return True
  
def getIsolate(row):
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    Gets the gene names that then form an isolate. It returns a list of Stirngs.
    """
    isolate = []
    for i in range(geneStart, geneEnd):
        isolate.append(row[i])
        
    return isolate
  
def getEsblCarba():
    """
    Author: Cory Kromer-Edwards
    Edits by: ...
    Gets the esbl and carba. groups from the required information excel workbook,
    and puts the categories into sets and returns those sets.
    """
    if debug:
        print(reqInformationPath + "\n")
        
    esbl = set()
    carba = set()
    wb = xlrd.open_workbook((reqInformationPath)) 
    sheet = wb.sheet_by_index(2) 
      
      
    for i in range(1, 424): 
        esbl.add(sheet.cell_value(i, 0))
                
    for i in range(1, 585): 
        carba.add(sheet.cell_value(i, 1))
                
    return (esbl, carba)
  
  
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
        return 3
    elif esblBool and carbaBool:
        return 2
    elif esblBool:
      return 0
    else:
      return 1

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
        




if __name__ == '__main__':
    main()
