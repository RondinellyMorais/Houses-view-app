
import pandas as pd
import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime

# Expand layout page
st.set_page_config(layout='wide')

@st.cache(allow_output_mutation = True)

def add_data(path):
    data = pd.read_csv(path)

    return data

def overview_data(data):
    st.sidebar.title('Costume overview')
    fe_attributes = st.sidebar.multiselect('Choose your columns in dataset overview', data.columns)
    fe_zipcode = st.sidebar.multiselect('Choose your zipcode in dataset overview', data['zipcode'].unique())


    if (fe_zipcode != []) & (fe_attributes != []):
        df = data.loc[data['zipcode'].isin(fe_zipcode), fe_attributes]

    elif (fe_zipcode != []) & (fe_attributes == []):
        df = data.loc[data['zipcode'].isin(fe_zipcode), :]

    elif (fe_zipcode == []) & (fe_attributes != []):
        df = data.loc[:, fe_attributes]

    else:
        df = data.copy()

    c1, c2 = st.beta_columns((1, 1))

    c1.dataframe(df)

    # Statistics columns
    da1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    da2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    da3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()

    m1 = pd.merge(da1, da2, on = 'zipcode', how ='inner')
    da = pd.merge(m1, da3, on = 'zipcode', how ='inner')

    da.columns = ['Zipcode', ' Total Houses', 'Price', 'Sqft Living']

    c1.header("Main Attributes")
    c1.dataframe(da, height=400)

     # Statistics
    n_attributes = data.select_dtypes(include=['int64', 'float64'])
    square = pd.DataFrame(n_attributes.apply(np.std))
    mean_ = pd.DataFrame(n_attributes.apply(np.mean))
    median_ = pd.DataFrame(n_attributes.apply(np.median))
    max_ = pd.DataFrame(n_attributes.apply(np.max))
    min_ = pd.DataFrame(n_attributes.apply(np.min))

    df1 = pd.concat([square, mean_, median_, max_, min_], axis = 1).reset_index()

    df1.columns = ['Attributes', 'Square', 'Mean', 'Median', 'Max', 'Min']

    c2.header("Describe Statistics Attributes")
    c2.dataframe(df1, height = 400)

    return None

def port_folio_density(data):
    # Houses density
    st.title("Region Overview")
    st.write("The map 'Houses Density' show the houses density distributions and some its features like 'price', 'area', year of built ... ")


    c1, c2 = st.beta_columns((1, 1))
    c1.header("Houses Density")

    density_map = folium.Map( 
    location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15
    )

    sam = data.sample(1000)

    cluster = MarkerCluster().add_to(density_map)
    for name, row in sam.iterrows():
        folium.Marker([row['lat'], row['long']],
        popup='Sold S$ on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'],
        row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(cluster)

    with c1:
        folium_static(density_map)

    return None

def Commercial_data(data):
     st.header('Avarage Price per day')
     st.sidebar.subheader('Select Max Date')
 
     data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d') # Converting data['date'] to date time type

    # filters
     min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
     max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

     f_date = st.sidebar.slider(' Select Date', min_date, max_date, min_date)

     data['date'] = pd.to_datetime(data['date'])

     df = data.loc[data['date'] < f_date]
     df = df[['date', 'price']].groupby('date').mean().reset_index()

     fig = px.line(df, x = 'date', y = 'price')
     st.plotly_chart(fig, use_container_width=True)

     # The average price per year

     # filters
     min_year_built = int(data['yr_built'].min())
     max_year_built = int(data['yr_built'].max())

     st.sidebar.subheader('Select Max Year Built')
     f_year_built = st.sidebar.slider( 'Year Built', min_year_built, max_year_built,
     min_year_built )
     st.header('The average price per year built')    

     df = data.loc[data['yr_built'] < f_year_built ]
     df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

     fig = px.line(df, x = 'yr_built', y = 'price')
     st.plotly_chart(fig, use_container_width=True)

     # Statistic Plots
     st.header('Price Distribution')
     st.sidebar.subheader('Select Max Price')
     st.write("Using 'Select Max Price' filter we can control price count distribution. ")

     price_min = int(data['price'].min())
     price_max = int(data['price'].max())
     price_avg = int(data['price'].mean())

     f_price = st.sidebar.slider('Price', price_min, price_max, price_avg)
     df = data.loc[data['price']< f_price]

     # Houses price distributions
     fig = px.histogram(df, x = 'price', nbins=50)
     st.plotly_chart(fig, use_container_width= True)

     return None
    
def Commercial_option(data):
     st.sidebar.title('Another Attibutes')
     st.title('Houses Another Attibutes')

     st.write("With the filters 'Max number of bedrooms' and 'Max number of bathrooms' we control de view of count houses with those attributes.")
     # filters
     fe_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
     fe_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

     c1, c2 = st.beta_columns(2)

     # Houses per  bedrooms
     c1.header('Houses per  bedrooms')
     df = data[data['bedrooms'] < fe_bedrooms]
     fig = px.histogram(df, x = 'bedrooms', nbins = 19)
     c1.plotly_chart(fig, use_container_width= True)

     # Houses per bathrooms
     c2.header('Houses per  bathrooms')
     df = data[data['bathrooms'] < fe_bathrooms]
     fig = px.histogram(df, x = 'bathrooms', nbins = 19)
     c2.plotly_chart(fig, use_container_width= True)

     st.write(" The filters 'Max number of floors' and 'Houses with waterview' select the houses with desired number floors and the second select houses with water view.")
     # filters
     fe_floors = st.sidebar.selectbox('Max number of floors', sorted(set(data['floors'].unique())))
     fe_waterview = st.sidebar.checkbox('Houses with waterview')

     c1, c2 = st.beta_columns(2)

     # Houses per floors
     c1.header('Houses per floors')
     df = data[data['floors'] < fe_floors]

     fig = px.histogram(data, x = 'floors', nbins = 19)
     c1.plotly_chart(fig, use_container_width= True)

     # Houses per waterview
     if fe_waterview:
         df = data[data['waterfront'] == 1]

     else:
        df = data.copy()

     fig = px.histogram(df, x = 'waterfront', nbins=10)
     c2.plotly_chart(fig, use_container_width= True)


if __name__ == "__main__":
   # Extration
   path = 'kc_house_data.csv'
   data = add_data(path)

   # transforme
   overview_data(data)

   port_folio_density(data)

   Commercial_data(data)

   Commercial_option(data)
   # Loading