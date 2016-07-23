import json
import requests
import os
import multiprocessing
import threading
import json_to_csv_converter

#use multiprocessing and threading
def load_file():
    """
    Accessing files from s3 to local and converting them from json to csv format
    """
    file_path = ['http://s3-us-west-2.amazonaws.com/yelpdatacapstone/yelp_data/yelp_dataset_business.json',
     'http://s3-us-west-2.amazonaws.com/yelpdatacapstone/yelp_data/yelp_dataset_review.json',
    'http://s3-us-west-2.amazonaws.com/yelpdatacapstone/yelp_data/yelp_dataset_user.json']

    for fp in file_path:
        file_name = fp.split('/')[-1]
        data_raw = requests.get(fp)
        with open(file_name, "w") as f:
            f.write(data_raw.text)
        os.system('python json_to_csv_converter.py %s'%(file_name))
        os.system('rm %s'%(file_name))

# def preprocessing():
#     
