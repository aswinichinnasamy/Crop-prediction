import streamlit as st
import pandas as pd
import joblib
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import plotly.express as px
from mysql.connector import Error

#Function to connect to MySql:
def get_connection():
    return mysql.connector.connect(
        host = "localhost",
        user = "root",
        database = "ma37",
        password = "AKsk1705@"
    )
@st.cache_data
def fetch_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM crops",conn)
    conn.close()
    return df

st.sidebar.title("Navigation")
page = st.sidebar.radio("Insights",["Project Introduction","Crop Analysis","Temporal Analysis",\
                                    "Input-Output Analysis",
                                    "Crop production prediction"])
if page == "Project Introduction":
    st.title("Crop production prediction")
    st.image("C:/Users/User/Desktop/guvi project1/Project3/crop1.jpg")
    st.write(""" 
             The project aims to develop a regression model that forecasts crop production in tons \ 
             based on agricultural factors like 
              Area harvested and Yield.
             
             ** Features: **
                    The findings provide valuable insights into crop analysis,temporal analysis,Comparative analysis
             
             ** Database: **
                    MySQL
             
             ** Models: **
                    Linear Regression
                    Decision Tree
                    Random Forest
             """)
    
elif page == "Crop Analysis":
    st.title("Crop Distribution Interactive Dashboard")
    #load data
    df = fetch_data()

    #Side filters:
    st.sidebar.header("Filter Data")
    year_filter = st.sidebar.multiselect("Select Year",options = df["Year"].unique())
    area_filter = st.sidebar.multiselect("Select Area",options = df['Area'].unique())
    item_filter = st.sidebar.multiselect("Select Item",options = df['Item'].unique())
    #Apply filters:
    df_filtered = df.copy()
    if year_filter:
        df_filtered = df_filtered[df_filtered['Year'].isin(year_filter)]
    if area_filter:
        df_filtered = df_filtered[df_filtered['Area'].isin(area_filter)]
    if item_filter:
        df_filtered = df_filtered[df_filtered['Item'].isin(item_filter)]

    st.subheader("Filtered Crop data")
    st.dataframe(df_filtered)

    #---Analyze Item distribution:
    st.subheader("Most and least cultivated crops across regions")
    item_summary = df_filtered.groupby("Item")['Value'].sum().reset_index().sort_values("Value",ascending = False)

    fig_item = px.bar(item_summary,
                      x = "Item",
                      y = "Value",
                      title = "Total Crop production by item",
                      color = "Value",
                      color_continuous_scale = "Viridis")
    st.plotly_chart(fig_item,use_container_width=True)

#High light top and bottom crops:
    st.markdown(f"**Most cultivated crop:**{item_summary.iloc[0]['Item']} with {item_summary.iloc[0]['Value']:.2f}total production.")
    st.markdown(f"**Least cultivated crop:**{item_summary.iloc[-1]['Item']} with {item_summary.iloc[-1]['Value']:.2f}total production.")

#Explore area distribution:
    st.subheader("Regional agricultural activity")
    area_summary = df_filtered.groupby('Area')['Value'].sum().reset_index().sort_values("Value",ascending = False)

    fig_area = px.bar(area_summary,
                      x='Area',
                      y='Value',
                      title='Total production by area',
                      color = 'Value',
                      color_continuous_scale = "Turbo"
                      )
    st.plotly_chart(fig_area,use_container_width=True)

#Crop distribution by Area:
    st.subheader("Crop distribution by Region")
    fig_heatmap = px.density_heatmap(df_filtered,
                                     x='Item',
                                     y='Area',
                                     z='Value',
                                     title="Heatmap: Crop production Across regions",
                                     color_continuous_scale ="Plasma")
    st.plotly_chart(fig_heatmap,use_container_width = True)

elif page == "Temporal Analysis":
    st.subheader("Temporal Analysis")

    #load data:
    df = fetch_data()

    #To ensure year is numeric
    df['Year']=pd.to_numeric(df['Year'],errors = "coerce")

    #sidebar filters:
    st.sidebar.header("Filter options")
    selected_area = st.sidebar.multiselect("Select region",df['Area'].unique())
    selected_item = st.sidebar.multiselect("Select Crops",df['Item'].unique())

    df_filtered = df.copy()

    if selected_area:
        df_filtered = df_filtered[df_filtered['Area'].isin(selected_area)]
    
    if selected_item:
        df_filtered = df_filtered[df_filtered['Item'].isin(selected_item)]

    #yearly trend analysis
    st.subheader("Yearly Trends: Area_Harvested, Yield, Production")

    #pivot to summarize values per year and element:
    yearly_summary = df_filtered.pivot_table(index = "Year",
                                             columns = "Element",
                                             values = "Value",
                                             aggfunc = "sum"
    ).reset_index()

    if not yearly_summary.empty:
        fig_trend = px.line(yearly_summary,x="Year",
                            y = yearly_summary.columns[1:],
                            markers = True,
                            title = "Yearly Trend of Area Harvested, Yield, Production")
        st.plotly_chart(fig_trend,use_container_width = True)

    else:
        st.warning("No data")

    #Crop/Region growth analysis
    st.subheader("Crop and Region growth Analysis")

    #selecting metric
    metric = st.radio("Select metric for growth analysis",["Yield","Production"])

    #Filter
    growth_df = df_filtered[df_filtered['Element'] == metric]

    if not growth_df.empty:
        fig_growth = px.line(growth_df,
                             x = "Year",
                             y = "Value",
                             color = "Item" if selected_area else "Area",
                             line_group="Area",
                             title = f"{metric} Trend by Crop/Region",
                             markers = True
                             )
        st.plotly_chart(fig_growth,use_container_width = True)
    else:
        st.warning("No Data available")

#Input - Output Analysis
elif page == "Input-Output Analysis":
    st.title("Crop Inout-Output Relationship and correlation Dashboard")

    #Load data:
    df = fetch_data()

    #pivotting
    pivot_df = df.pivot_table(index = ['Area','Item','Year'],
                        columns = 'Element',
                        values = 'Value',
                        aggfunc = "sum").reset_index()
    
    #sidebar filters:
    st.sidebar.header("Filter options")
    selected_area = st.sidebar.multiselect("Select Regions",pivot_df['Area'].unique())
    selected_item = st.sidebar.multiselect("Select Item",pivot_df['Item'].unique())

    df_filtered = pivot_df.copy()
    if selected_area:
        df_filtered = df_filtered[df_filtered['Area'].isin(selected_area)]

    if selected_item:
        df_filtered = df_filtered[df_filtered['Item'].isin(selected_item)]

    st.subheader("Filtered data")
    st.dataframe(df_filtered)

    #correlation analysis:
    st.subheader("Correlation between land usage and productivity")
    corr_df = df_filtered[["Area harvested",'Yield','Production']].corr()

    st.write("Correlation matrix")
    st.dataframe(corr_df.style.background_gradient(cmap = "coolwarm"))

    df_filtered = df_filtered.dropna(subset=['Production'])

    #Scatter plots for Input vs output:
    st.subheader("Input vs Output Relationship")
    col_x = st.selectbox("select x-axis (input)",['Area harvested','Yield'])
    col_y = st.selectbox("Select y-axis (output)",['Production','Yield'])

    fig_scatter = px.scatter(df_filtered,x=col_x,y=col_y,
                             color = "Item",
                             size = "Production",
                             hover_data = ['Area','Year'],
                             title = f"{col_x} vs {col_y}",
                             trendline = "ols")
    st.plotly_chart(fig_scatter,use_container_width = True)

    #3d Visualization:
    st.subheader("#3d View of Land usage Vs Productivity Vs ouptut")

    fig_3d = px.scatter_3d(df_filtered,x = "Area harvested",
                           y = "Yield",
                           z = "Production",
                           color = "Item",
                           hover_data = ['Area','Year'],
                           size = "Production",
                           title = "3D relationship Area Vs Yield Vs Production")
    st.plotly_chart(fig_3d,use_container_width=True)

                                                        
elif page == "Crop production prediction":
    #load trained model
    dt = joblib.load('C:/Users/User/Desktop/guvi project1/Project3/crop/model_dt.pkl')

#Label encoders used:
    le_area = joblib.load('C:/Users/User/Desktop/guvi project1/Project3/crop/le_area.pkl')
    le_item = joblib.load('C:/Users/User/Desktop/guvi project1/Project3/crop/le_item.pkl')

#Title:
    st.title(" Crop production prediction App")
    st.write(" Predict crop production based on selected inputs")
    st.image("C:/Users/User/Desktop/guvi project1/Project3/crop2.jpg")

#input widgets:
    area = st.selectbox("Select Area",le_area.classes_)
    item = st.selectbox("Item",le_item.classes_)
    year = st.number_input("Enter Year", min_value = 2000,max_value = 2050, step = 1)
    area_harvested = st.number_input("Area harvested (ha)",min_value = 0.0)
    yield_value = st.number_input("Yield",min_value = 0.0)

#Encode inputs(match training)
    area_encoded = le_area.transform([area])[0]
    item_encoded = le_item.transform([item])[0]

#create dataframe for prediction:
    input_data = np.array([year,area_harvested,yield_value,area_encoded,item_encoded])

#Predict button
    if st.button("Predict"):
        prediction = dt.predict(input_data.reshape(1,-1))[0]
        st.success(f"Predicted Value:{prediction:,.2f}")
