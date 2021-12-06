import os
from zipfile import ZipFile

os.system('kaggle datasets download ilhamfp31/yelp-review-dataset' )
os.system('unzip yelp-review-dataset.zip')

# specifying the zip file name
file_name = "yelp-review-dataset.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')