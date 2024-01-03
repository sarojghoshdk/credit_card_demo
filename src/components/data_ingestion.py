import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)



class DataIngestion:
    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()


    def export_collection_as_dataframe(self,collection_name, db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)    # MongoClient(: This is a constructor from the pymongo library, which is used to create connections to MongoDB databases.

            collection = mongo_client[db_name][collection_name]     # mongo_client[db_name]: This part accesses a specific database within the MongoDB instance. 
                                                               # [collection_name]: This part further drills down to access a specific collection within the chosen database. 
            df = pd.DataFrame(list(collection.find()))  # collection.find(): This part retrieves all documents from the specified MongoDB collection. It returns a cursor object, which acts as an iterator over the results.
                                                        # list(...): This converts the cursor object into a Python list. This is necessary because the DataFrame constructor expects a list-like object as input.
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def export_data_into_feature_store_file_path(self)->pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method reads data from mongodb and saves it into artifacts. 
        
        Output      :   dataset is returned as a pd.DataFrame
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   0.1
       
        """
        try:
            logging.info("Exporting Data From MongoDB")
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path, exist_ok=True)

            card_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,
                db_name= MONGO_DATABASE_NAME
            )

            logging.info("Saving Exported Data into feature store file path: {raw_file_path}")

            feature_store_file_path = os.path.join(raw_file_path, "credit_card.csv")
            card_data.to_csv(feature_store_file_path,index=False)


            return feature_store_file_path

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_ingestion(self) -> Path:
        """
            Method Name :   initiate_data_ingestion
            Description :   This method initiates the data ingestion components of training pipeline 
            
            Output      :   train set and test set are returned as the artifacts of data ingestion components
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:

            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("Got The Data From MongoDB")


            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys) from e