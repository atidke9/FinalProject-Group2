import os
from zipfile import ZipFile

Path = os.getcwd()

# specifying the zip file name
file_name = Path + "\\yelp-review-dataset.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
