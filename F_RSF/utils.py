
import csv
import os
from typing import List

def get_params(csv_file: str, client_id) -> List:
    """Read the results of current round."""
    with open(csv_file, newline='') as csvfile:
        paramReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in paramReader:
            
            #print(client_id)
            if(len(row) == 1 ):#and row[0] == client_id):
                rowlist = list(row[0].split(","))
                if rowlist[0] == str(client_id):
                    return rowlist
def read_csv(csv_file: str) -> List:
    """Read the results of current round."""
    with open(csv_file, newline='') as csvfile:
        paramReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rowlist = []
        for row in paramReader:
            if(len(row) == 1 ):#and row[0] == client_id):
                #rowlist = list(row[0].split(","))
                rowlist.append(row)
        return rowlist
