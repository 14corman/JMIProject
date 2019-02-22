# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:28:43 2019
"""


import csv

# All start and end positions for required variables in the csv.
geneStart = 46
geneEnd = 197


def main(debug):
    """
    Author Cory Kromer-Edwards
    Edits by: ...
    The main method to preprocessing. This method will open the dataset csv
    file, and send each row to the preprocessing steps.
    """
    with open('dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count > 5 and debug:
                break
            else:
                if testRow(row):
                    print("Genes: ", getGenes(row))
                    
                line_count += 1
                
        print(f'Processed {line_count} lines.')
        
def testRow(row):
    """
    This method will take in a row and test all needed values to see if there is
    something valid in that position. If all required values have are not empty,
    then the row is valid. The method will return True iff the row is valid, 
    false otherwise.
    """
    for i in range(geneStart, geneEnd):
        if row[i] == "":
            return False

    return True

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
    main(True)
