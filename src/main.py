import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def clean_data(filename):
    data = pd.read_csv(filename)
    data.drop('ID', axis = 1, inplace=True)
    data.drop('Source', axis = 1, inplace=True)
    data.drop('End_Time', axis = 1, inplace=True)
    data.drop('End_Lat', axis = 1, inplace=True)
    data.drop('End_Lng', axis = 1, inplace=True)
    data.drop('Street', axis = 1, inplace=True)
    data.drop('City', axis = 1, inplace=True)
    data.drop('County', axis = 1, inplace=True)
    data.drop('Zipcode', axis = 1, inplace=True)
    data.drop('Country', axis = 1, inplace=True)
    data.drop('Timezone', axis = 1, inplace=True)
    data.drop('Airport_Code', axis = 1, inplace=True)
    data.drop('Weather_Timestamp', axis = 1, inplace=True)
    data.drop('Wind_Direction', axis = 1, inplace=True)
    data.drop('Amenity', axis = 1, inplace=True)
    data.drop('Sunrise_Sunset', axis = 1, inplace=True)
    data.drop('Nautical_Twilight', axis = 1, inplace=True)
    data.drop('Astronomical_Twilight', axis = 1, inplace=True)
    substring = 'I-'
    data['Start_Index'] = data['Description'].str.find(substring)
    data['Start_Index'] = data['Start_Index'].replace([np.inf, -np.inf, np.nan], -1).astype('int64')
    data['Freeway'] = data.apply(
        lambda row: row['Description'][row['Start_Index']:row['Start_Index'] + 5] if row['Start_Index'] != -1 else "O", 
        axis=1
    )

    data['Freeway'] = data['Freeway'].str.replace(' ', '', regex=True)
    data['Is_Freeway'] = data['Start_Index'].apply(lambda x: True if x >= 0 else False)

    data.drop('Start_Index', axis = 1, inplace=True)
    data.drop('Description', axis = 1, inplace=True)
    data['Start_Time'] = pd.to_datetime(data['Start_Time'], errors='coerce')
    data['Is_Day'] = data['Civil_Twilight'].str.contains("Day")
    data.drop('Civil_Twilight', axis = 1, inplace=True)

    data['Severity'] = data['Severity'].astype('int8')
    data['Start_Lat'] = data['Start_Lat'].astype('float16')
    data['Start_Lng'] = data['Start_Lng'].astype('float16')
    data['Distance(mi)'] = data['Distance(mi)'].astype('float16')
    data['Temperature(F)'] = data['Temperature(F)'].astype('float16')
    data['Wind_Chill(F)'] = data['Wind_Chill(F)'].astype('float16')
    data['Humidity(%)'] = data['Humidity(%)'].astype('float16')
    data['Pressure(in)'] = data['Pressure(in)'].astype('float16')
    data['Visibility(mi)'] = data['Visibility(mi)'].astype('float16')
    data['Wind_Speed(mph)'] = data['Wind_Speed(mph)'].astype('float16')
    data['Precipitation(in)'] = data['Precipitation(in)'].astype('float16')

    data.to_pickle('clean_data.pkl')
    return data


def load_data(filename):
    df = pd.read_pickle(filename)
    return df

    
if __name__ == "__main__":
    file_path = "/home/surya/Desktop/FA24/ECE143_Python/Final_Project/US_Car_Accident_Data_Analysis/Dataset/US_Accidents_March23.csv"
    df = clean_data(file_path)
    pickle_path = "/home/surya/Desktop/FA24/ECE143_Python/Final_Project/US_Car_Accident_Data_Analysis/clean_data.pkl"
    df_temp = load_data(pickle_path)
    print(df_temp.equals(df))
    x = 5



