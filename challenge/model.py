import pandas as pd

from typing import Tuple
from typing import Union
from typing import List

from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance

class DelayModel:

    def __init__(
        self
    ):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=4.4402380952380955)

    def get_period_day(self,date:str):
        """
        Determine the time of the day based on the input date and time.

        Args:
            date (str): A string representing the date and time in the format '%Y-%m-%d %H:%M:%S'.

        Returns:
            str: A string representing the time period of the day.
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        night_min = datetime.strptime("19:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(date_time > night_min and date_time < night_max):
            return 'noche'
    
    def is_high_season(self,date:str):
        """
        Check if the provided date falls within a high season.

        Args:
            date (str): A string representing the date and time in the format '%Y-%m-%d %H:%M:%S'.

        Returns:
            int: 1 if the date is in high season, 0 otherwise.
        """
        date_year= int(date.split('-')[0])
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = date_year)
        range1_max = datetime.strptime('3-Mar', '%d-%b').replace(year = date_year)
        range2_min = datetime.strptime('15-Jul', '%d-%b').replace(year = date_year)
        range2_max = datetime.strptime('31-Jul', '%d-%b').replace(year = date_year)
        range3_min = datetime.strptime('11-Sep', '%d-%b').replace(year = date_year)
        range3_max = datetime.strptime('30-Sep', '%d-%b').replace(year = date_year)
        
        if ((date >= range1_min and date <= range1_max) or 
            (date >= range2_min and date <= range2_max) or 
            (date >= range3_min and date <= range3_max)):
            return 1
        else:
            return 0
        
    def get_min_diff(self,data: pd.DataFrame):
        """
        Calculate the time difference in minutes between two provided dates.

        Args:
            data (dict): A dictionary containing two date and time values as strings with keys 'Fecha-O' and 'Fecha-I'.

        Returns:
            float: The time difference in minutes.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.       

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )[top_10_features]
        if target_column is not None:
            target = pd.DataFrame(data[target_column])
            return features,target
        else:
            return features
        
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, _, y_train, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)
        self._model.fit(x_train, y_train)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        xgboost_y_preds = self._model.predict(features)
        xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
        return xgboost_y_preds