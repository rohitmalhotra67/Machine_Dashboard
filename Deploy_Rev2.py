from lib2to3.pytree import LeafPattern
import streamlit as st
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.graph_objects as go
import statsmodels.api as sm





# SECTION 1: DISPLAY OF WELCOME MESSAGE

from PIL import Image
image = Image.open('gail.png')
st.set_page_config(page_title='GAIL Chhainsa',page_icon=image)
col1, col2, col3 = st.columns(3)

with col1:
    st.write("")

with col2:
    st.image(image)
    #st.markdown("<img src="gail.png"  alt="Flowers in Chania">", unsafe_allow_html=True)

with col3:
    st.write("")


#Create a title for your app
st.markdown("<h1 style='text-align: center; color: blue;'>Chhainsa Compressor Station</h1>", unsafe_allow_html=True)

st.text("**********************************************************************************************************************************************************************")
st.markdown("<h1 style='text-align: center; color:red;'> MACHINE ANALYTICS DASHBOARD FOR GAS ENGINE GENERATOR </h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color:blue;'>1. Perform Trip Analysis</h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color:green;'>2. Indentify Key Attributes of GEG correlated with its Power Produced </h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color:purple;'>3. Visualization of GEG Attributes  Vs GEG Power Output  </h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color:orange;'>4. Exhaust Gas Temperature Profile of various GEG Cylinders   </h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color:magenta;'>5. Anomaly Detetction in various attributes(Sensors) of GEG    </h2>", unsafe_allow_html=True)

#st.text("**********************************************************************************")
st.text("**********************************************************************************************************************************************************************")

# SECTION 2: ASK USER TO MAKE SELECTION FOR TYPE OF ANALYSIS NEED TO BE PERFORMED

selection1 = st.radio("Select Type of Analysis Need to be performed: ", 
('Trip Analysis','Indentify Key Attributes of GEG correlated with its Power Produced','Visualization of GEG Attributes  Vs GEG Power Output',
'Exhaust Gas Temperature Profile of various GEG Cylinders','Anomaly Detetction in various attributes(Sensors) of GEG'))

# SECTION 3: Trip Analysis

if(selection1=='Trip Analysis'):

    st.text("**********************************************************************************")
    st.markdown("<h1 style='text-align: center; color:red;'>PERFORM TRIP ANALYSIS </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:magenta;'>Kindly upload file for Trip Analysis</h2>", unsafe_allow_html=True)

    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")


    dataset1 = st.file_uploader("Choose Trip Data",type="xlsx")
    if dataset1 is not None:
        dataset1 = pd.read_excel(dataset1)
        # Removing Unamed columns
        dataset1.drop(dataset1.columns[dataset1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        st.write(dataset1)

    #dataset1=dataset1.dropna()

    # Displaying Visualization Chart

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Throttle valve position [%]'], name='Throttle valve position',
                            line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Power actual [kW]'], name='Power actual [kW]',
                            line=dict(color='green', width=4)))
    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Turbo bypass position [%]'], name='Turbo bypass position',
                            line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Voltage excitation [V]'], name='Voltage excitation [V]',
                            line=dict(color='black', width=4)))
    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Generator voltage avg [V]'], name='Generator voltage avg [V]',
                            line=dict(color='magenta', width=4)))
    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Speed actual [1/min]'], name='Speed actual',
                            line=dict(color='orange', width=4)))

    fig.add_trace(go.Scatter(x=dataset1['Time'], y=dataset1['Gas prop. valve lambda []'], name='Gas prop. valve',
                            line=dict(color='skyblue', width=4)))

    #Update Xaxes
    #fig.update_xaxes(rangeslider_visible=True)


    # Edit the layout
    fig.update_layout(autosize=False, width=2000,height=1000,
        margin=dict(l=50,r=50,b=100,t=100,pad=4),title='GEG-2 Trip Analysis',
                    xaxis_title='Date & Time',
                    yaxis_title='Throtle Valve Position/By-Pass Valve Position/Generator Volatage')
    #fig.show()
    st.plotly_chart(fig)

# SECTION 4 : Indentify Key Attributes of GEG correlated with its Power Produced


elif(selection1=='Indentify Key Attributes of GEG correlated with its Power Produced'):

    # Ask user to upload Machine Logged Data

    st.text("**********************************************************************************************************************************************************************")

    st.markdown("<h1 style='text-align: center; color:red;'>PERFORM IDENTIFICATION OF KEY ATTRIBUTES OF GEG CORRELATED WITH POWER PRODUCED </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:magenta;'>Kindly upload file for Identifying GEG Key Attributes Correlated with Power Output</h2>", unsafe_allow_html=True)

    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")


    dataset = st.file_uploader("Choose a file",type="csv")
    if dataset is not None:
        dataset = pd.read_csv(dataset)
        # Removing Unamed columns
        dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        st.write(dataset)

    # Removing Null Values
    dataset=dataset.dropna()
    

    st.subheader("Deploying Random Forest Machine Learning Model to Identify Relation between varrious attributes of GEG Vs its Power Output")
    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")



    # Seperating Dependent and Independent Variables
    X=dataset.drop(['Electrical power P'],axis=1)
    y=dataset['Electrical power P']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Performing standardisation of Data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    # First we build and train our Random Forest Model 
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(max_depth=15, random_state=42, n_estimators = 60).fit(X_train, y_train)

        # Displaying the accuracy Of Model

    
    st.subheader("Accuracy Achived in Deployed Model to Predict the Power Output")
    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")


    from sklearn.metrics import mean_squared_error, r2_score
    #X_train, X_test, y_train, y_test
    y_pred_train=rf.predict(X_train)
    

    #X_train, X_test, y_train, y_test
    y_pred_test=rf.predict(X_test)
    

    # Performance of Model Deployed on Data to Predict Correlation


    # initialise data of lists.
    data = {'DATA':['Train Data','Test Data'], 'PREDICTION ACCURACY ':[r2_score(y_train,y_pred_train),r2_score(y_test,y_pred_test)]}
 
    # Create DataFrame
    df = pd.DataFrame(data)
    df=df.set_index('DATA')

    st.dataframe(df)

    # Plotting Importance of Attributes

    list_featured=pd.DataFrame({'Tags_Description':X.columns,'Importance_Level':rf.feature_importances_})
    list_featured=list_featured.sort_values("Importance_Level",ascending=False)

    # Reading Tags file

    Tags1=pd.read_csv('Tags-GEG.csv')
    #Tags.drop(Tags.columns[Tags.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    Tags_List=Tags1['Tags_Description'].tolist()


    # Merging Tag Description to created dataframe for Tag Importance Level
    list_featured = pd.merge(list_featured,Tags1,on='Tags_Description',how='inner')
    list_featured=list_featured[['Tags_Description','Tag Unit','Importance_Level']]



    st.subheader("Bar Plot to Visualize Importance of Various Attributes")
    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")


    # Plotting Bar & Pie Chart

    graph1=px.bar(list_featured, x='Tags_Description',y='Importance_Level',width=800, height=700)
    graph1.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightGrey",
    font_family="Courier New",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="red",
    legend_title_font_color="green")
    st.plotly_chart(graph1)


    st.subheader("Pie Plot to Visualize Importance of Various Attributes")
    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")


    graph2 = px.pie(list_featured, values='Importance_Level', names='Tags_Description',width=1000, height=800)
    st.plotly_chart(graph2)



# SECTION 5: Visualization of GEG Attributes  Vs GEG Power Output

elif(selection1=='Visualization of GEG Attributes  Vs GEG Power Output'):

    
    # Ask user to upload Machine Logged Data
    st.header("Kindly upload file for GEG Logged Data")
    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")

    st.text("**********************************************************************************************************************************************************************")

    st.markdown("<h1 style='text-align: center; color:red;'>VISUALIZATION OF GEG ATTRIBUTES Vs GEG POWER OUTPUT </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:magenta;'>Kindly upload file for Visualization</h2>", unsafe_allow_html=True)

    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")

    dataset = st.file_uploader("Choose a file",type="csv")
    if dataset is not None:
        dataset = pd.read_csv(dataset)
        # Removing Unamed columns
        dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        st.write(dataset)

    # Plotting the Visualization

    st.subheader("Visualization of plot of GEG various attributes Vs GEG Electrical Power Output")
    st.subheader("Select Attribute")
    Tags1=pd.read_csv('Tags-GEG.csv')
    Tags_List=Tags1['Tags_Description'].tolist()

    List_Tags=pd.DataFrame({'Tag_Description':Tags_List})
    st.write(List_Tags)
    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")

    sel_x = st.selectbox("SELECT GEG ATTRIBUTE FOR X-AXIS ",
                            Tags_List)


    sel_y = st.selectbox("SELECT GEG ATTRIBUTE FOR Y-AXIS ",
                            Tags_List)

    fig = px.scatter(dataset, x=dataset[sel_x], y=dataset[sel_y],trendline="ols")


    fig.update_layout(autosize=False,width=2000,height=1000,
        margin=dict(l=50,r=50,b=100,t=100,pad=4),
                    xaxis_title=sel_x,
                    yaxis_title=sel_y, yaxis_range=[dataset[sel_y].min(),dataset[sel_y].max()],xaxis_range=[dataset[sel_x].min(),dataset[sel_x].max()],
                     font=dict(family="serif",
            size=22,
            color="darkred"),title={
            'text': "{} Vs {}".format(sel_y,sel_x) ,
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    #fig.show()
    st.plotly_chart(fig)




# SECTION 6: Exhaust Gas Temperature Profile of various GEG Cylinders


elif(selection1=='Exhaust Gas Temperature Profile of various GEG Cylinders'):

    
    st.text("**********************************************************************************************************************************************************************")

    st.markdown("<h1 style='text-align: center; color:red;'>EXHAUST GAS TEMPERATURE PROFILE OF VARIOUS CYCLINDERS OF GEG </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:magenta;'>Kindly upload GEG Logged Data File</h2>", unsafe_allow_html=True)

    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")

    dataset = st.file_uploader("Choose a file",type="csv")
    if dataset is not None:
        dataset = pd.read_csv(dataset)
        # Removing Unamed columns
        dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        st.write(dataset)

    # Removing Null Values
    #dataset=dataset.dropna()
    dataset1=dataset

    # Plotting Box Plot
    fig = go.Figure()

    for i in dataset.columns[2:14]:
        fig.add_trace(go.Box(y=dataset[i], name=i))
  

    fig.update_layout(autosize=False, width=2000,height=1000,
    margin=dict(l=50,r=50,b=100,t=100,pad=4),
                   xaxis_title='Exhaus Gas Cylinder Profile',
                   yaxis_title='Exhaust Gas Cylinder Temp',yaxis_range=[500,600],font=dict(
            family="Courier New, monospace",
            size=24,
            color="black"),title={
            'text': " EXHAUST GAS CYCLINDERS TEMPERATURE PROFILE" ,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


    #fig.show()
    st.plotly_chart(fig)

# SECTION 7: Anomaly Detetction in various attributes(Sensors) of GEG


elif(selection1=='Anomaly Detetction in various attributes(Sensors) of GEG'):

    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")

    st.markdown("<h1 style='text-align: center; color:red;'>PERFORM ANOMALY DETECTION IN SENSOR DATA </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:magenta;'>Kindly upload file for Anamoly Detection</h2>", unsafe_allow_html=True)

    #st.text("**********************************************************************************")
    st.text("**********************************************************************************************************************************************************************")



    dataset1 = st.file_uploader("Upload Logged Data",type="xlsx")
    if dataset1 is not None:
        dataset1 = pd.read_excel(dataset1)
        # Removing Unamed columns
        dataset1.drop(dataset1.columns[dataset1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        st.write(dataset1)

    

    # Merge two Columnn into one column
    dataset1['DATE']=dataset1['Date'].astype(str)+' '+dataset1['Time'].astype(str)

    # Re-ordering the columns

    dataset1=dataset1[['DATE','Speed actual [1/min]','Power actual [kW]','Boostpressure [bar]','Generator current avg [A]',
               'Mixture temperature [°C]','Throttle valve position [%]','Turbo bypass position [%]','Gas prop. valve lambda []',
               'Voltage excitation [V]','Generator voltage avg [V]']]

    #st.write(dataset1)
    
    
    # Displaying Descriptive Statistics

    st.text("**********************************************************************************************************************************************************************")
    st.markdown("<h2 style='text-align: center; color:magenta;'>Displaying Descriptive Statistics of Sensor Data</h2>", unsafe_allow_html=True)

    st.write(dataset1[['Speed actual [1/min]','Power actual [kW]','Boostpressure [bar]','Generator current avg [A]',
               'Mixture temperature [°C]','Throttle valve position [%]','Turbo bypass position [%]','Gas prop. valve lambda []',
               'Voltage excitation [V]','Generator voltage avg [V]']].describe().T)
    
    # Change DATE Column formatting

    dataset1['DATE'] = pd.to_datetime(dataset1['DATE'])
    dataset1['DATE'] = pd.to_datetime(dataset1['DATE'],dayfirst=True)

    # Calculating Descriptive Statistics for the attributes

    for i in dataset1.columns[1:12]:
        dataset1['{}_MEAN'.format(i)]=dataset1['{}'.format(i)].rolling(20).mean()
        dataset1['{}_STD'.format(i)]=dataset1['{}'.format(i)].rolling(20).std()
        dataset1['{}_MIN'.format(i)]=dataset1['{}'.format(i)].rolling(20).min()
        dataset1['{}_MAX'.format(i)]=dataset1['{}'.format(i)].rolling(20).max()
    
    dataset=dataset1.fillna(dataset1.mean())

    options=dataset1.columns[1:12].tolist()

    #st.table(options)

    choice=st.multiselect(" SELECT TAG FOR WHICH YOU WANT DETECT ANAMOLY",options)


    import plotly.graph_objects as go
    fig = go.Figure()

    #for i in dataset1.columns[1:11]:
    for i in choice:

        fig.add_trace(go.Scatter(x=dataset1['DATE'], y=dataset1[i], name=i,
                         line=dict(color='orange', width=4)))

        fig.add_trace(go.Scatter(x=dataset1['DATE'], y=dataset1['{}_MEAN'.format(i)], name='MEAN_{}'.format(i),
                         line=dict(color='skyblue', width=4)))
        fig.add_trace(go.Scatter(x=dataset1['DATE'], y=dataset1['{}_STD'.format(i)], name='STD_{}'.format(i),
                         line=dict(color='yellow', width=4)))
        fig.add_trace(go.Scatter(x=dataset1['DATE'], y=dataset1['{}_MIN'.format(i)], name='MIN_{}'.format(i),
                         line=dict(color='red', width=4)))
        fig.add_trace(go.Scatter(x=dataset1['DATE'], y=dataset1['{}_MAX'.format(i)], name='MAX_{}'.format(i),
                         line=dict(color='green', width=4)))
    #Update Xaxes
    #fig.update_xaxes(rangeslider_visible=True)

    # Edit the layout
    fig.update_layout(autosize=False, width=2000,height=1000,
    margin=dict(l=50,r=50,b=100,t=100,pad=4),title='Detection of Anomaly in Sensor Data',
                   xaxis_title='Date & Time',
                   yaxis_title='Selected Attribute')
    #fig.show()
    st.plotly_chart(fig)








# END

