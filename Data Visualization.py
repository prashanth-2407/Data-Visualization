#!/usr/bin/env python
# coding: utf-8

# In[213]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[197]:


data_all = pd.read_excel(r"Final Consolidated file.xlsx", sheet_name=['On Ground','Virtual'])


# In[198]:


data = data_all.get('On Ground')
datav = data_all.get('Virtual')


# In[199]:


data.fillna(0, inplace = True)
data['Number of family members joining'].replace('Nil', 0, inplace = True)
data['EMP T Shirt Size'].replace(0, "Didn't provide T shirt size", inplace = True)
data['Number of family members joining'] = data['Number of family members joining'].astype(float)

datav.fillna(0, inplace = True)
datav['Number of family members joining'].replace('Nil', 0, inplace = True)
datav['EMP T Shirt Size'].replace(0, "Didn't provide T shirt size", inplace = True)
datav['Gender'].replace(0, "Gender not mentioned", inplace = True)
datav['Number of family members joining'] = datav['Number of family members joining'].astype(float)


# In[200]:


data.columns


# In[201]:



colors = ['lightblue', 'lightgreen']
df1=data['Gender'].value_counts()
fig = go.Figure(data=[go.Pie(labels=df1.index,values=df1.values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=600,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees Gender Distribution - On Ground Run')
fig.show()


colors = ['lightblue', 'lightgreen']
df1=datav['Gender'].value_counts()
fig = go.Figure(data=[go.Pie(labels=df1.index,values=df1.values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=750,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees Gender Distribution - Virtual Run')
fig.show()


# In[202]:


colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
df1=data['Select Category for Self'].value_counts()
df1.drop(0,axis=0,inplace=True)
fig = go.Figure(data=[go.Bar(x=df1.index,y=df1.values)])
fig.update_traces(hoverinfo='y',text=df1.values,textfont_size=15,marker=dict(color=colors,line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=900,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees registered for different categories - On Ground Run')
fig.show()


colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
df1=datav['Select Category for Self'].value_counts()
df1.drop(0,axis=0,inplace=True)
fig = go.Figure(data=[go.Bar(x=df1.index,y=df1.values)])
fig.update_traces(hoverinfo='y',text=df1.values,textfont_size=15,marker=dict(color=colors,line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=900,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees registered for different categories - Virtual')
fig.show()


# In[203]:


colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
df2=data['Current Location (Division)'].value_counts()
fig = go.Figure(data=[go.Bar(x=df2.index,y=df2.values)])
fig.update_traces(hoverinfo='y',text=df2.values,textfont_size=12,marker=dict(color=colors,line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,font=dict(
        family="Courier New, monospace",
        size=12,
        color="#0c0b0c"),

    title_text='Employees registered across different locations - On Ground Run')
fig.show()


colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
df2=datav['Current Location (Division)'].value_counts()
df2.drop(0,axis=0,inplace=True)
fig = go.Figure(data=[go.Bar(x=df2.index,y=df2.values)])
fig.update_traces(hoverinfo='y',text=df2.values,textfont_size=12,marker=dict(color=colors,line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=1000,
    height=650,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees registered across different locations - Virtual Run')
fig.show()


# In[204]:


fig = go.Figure(data=[go.Table(
                                
                                header=dict(values=['Employee Name','Number of family members participating'],
                                line_color='darkslategray',
                                fill_color='lightgreen',
                                align='center',
                                font=dict(color="black", size=20)),
    
                                cells=dict(values=[data.sort_values('Number of family members joining',ascending=False).iloc[0:5]['Employee Name'],
                                           data.sort_values('Number of family members joining',ascending=False).iloc[0:5]['Number of family members joining']],
                                line_color='darkslategray',
                                fill_color='lightblue',
                                           align='center',
                                font=dict(color="darkslategray", size=15))
)])

fig.update_layout(width=900, height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Top 5 Employees with highest number of family members participating - On Ground Run')
fig.show()


fig = go.Figure(data=[go.Table(
                                
                                header=dict(values=['Employee Name','Number of family members participating'],
                                line_color='darkslategray',
                                fill_color='lightgreen',
                                align='center',
                                font=dict(color="black", size=20)),
    
                                cells=dict(values=[datav.sort_values('Number of family members joining',ascending=False).iloc[0:5]['Employee Name'],
                                           datav.sort_values('Number of family members joining',ascending=False).iloc[0:5]['Number of family members joining']],
                                line_color='darkslategray',
                                fill_color='lightblue',
                                           align='center',
                                font=dict(color="darkslategray", size=15))
)])

fig.update_layout(width=900, height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Top 5 Employees with highest number of family members participating - Virtual Run')
fig.show()


# In[205]:


colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
df3=data['EMP T Shirt Size'].value_counts()
fig = go.Figure(data=[go.Pie(labels=df3.index,values=df3.values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=600,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees T shirt size Distribution - On Ground Run')
fig.show()


colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
df3=datav['EMP T Shirt Size'].value_counts()
fig = go.Figure(data=[go.Pie(labels=df3.index,values=df3.values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=600,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Employees T shirt size Distribution - Virtual Run')
fig.show()


# In[206]:


datan = pd.DataFrame()
data1 = pd.DataFrame()
data2 = pd.DataFrame()
data3 = pd.DataFrame()
data4 = pd.DataFrame()
data5 = pd.DataFrame()
data6 = pd.DataFrame()
data7 = pd.DataFrame()
data8 = pd.DataFrame()
data9 = pd.DataFrame()
data10 = pd.DataFrame()


data1[['Name','Category']] = data[['Name of Family Member','Family Member Category -1.1']]
data2[['Name','Category']] = data[['Name of Family Member 1','Family Member Category -2.1']]
data3[['Name','Category']] = data[['Name of Family Member 2','Family Member Category -2.2']]
data4[['Name','Category']] = data[['Name of Family Member 1.1','Family Member Category -3.1']]
data5[['Name','Category']] = data[['Name of Family Member 2.1','Family Member Category -3.2']]
data6[['Name','Category']] = data[['Name of Family Member 3','Family Member Category -3.3']]
data7[['Name','Category']] = data[['Name of Family Member 1.2','Family Member Category -4.1']]
data8[['Name','Category']] = data[['Name of Family Member 2.2','Family Member Category -4.2']]
data9[['Name','Category']] = data[['Name of Family Member 3.1','Family Member Category -4.3']]
data10[['Name','Category']] = data[['Name of Family Member 4','Family Member Category -4.4']]



Result = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], ignore_index=True)
Result

colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
Result['Category'].replace(0, np.nan, inplace = True)
Result.dropna(inplace = True)
df4=Result['Category'].value_counts()

fig = go.Figure(data=[go.Bar(x=df4.index,y=df4.values)])
fig.update_traces(hoverinfo='y',text=df4.values,textfont_size=12,marker=dict(color=colors,line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=900,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Family members registered for different categories - On Ground Run')
fig.show()



datan = pd.DataFrame()
data1 = pd.DataFrame()
data2 = pd.DataFrame()
data3 = pd.DataFrame()
data4 = pd.DataFrame()
data5 = pd.DataFrame()
data6 = pd.DataFrame()
data7 = pd.DataFrame()
data8 = pd.DataFrame()
data9 = pd.DataFrame()
data10 = pd.DataFrame()


data1[['Name','Category']] = datav[['Name of Family Member','Family Member Category -1.1']]
data2[['Name','Category']] = datav[['Name of Family Member 1','Family Member Category -2.1']]
data3[['Name','Category']] = datav[['Name of Family Member 2','Family Member Category -2.2']]
data4[['Name','Category']] = datav[['Name of Family Member 1.1','Family Member Category -3.1']]
data5[['Name','Category']] = datav[['Name of Family Member 2.1','Family Member Category -3.2']]
data6[['Name','Category']] = datav[['Name of Family Member 3','Family Member Category -3.3']]
data7[['Name','Category']] = datav[['Name of Family Member 1.2','Family Member Category -4.1']]
data8[['Name','Category']] = datav[['Name of Family Member 2.2','Family Member Category -4.2']]
data9[['Name','Category']] = datav[['Name of Family Member 3.1','Family Member Category -4.3']]
data10[['Name','Category']] = datav[['Name of Family Member 4','Family Member Category -4.4']]



Resultn = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], ignore_index=True)
Resultn

colors = ['lightgrey', 'lightgreen','lightyellow','lightblue','lightpink']
Resultn['Category'].replace(0, np.nan, inplace = True)
Resultn.dropna(inplace = True)
df5=Resultn['Category'].value_counts()

fig = go.Figure(data=[go.Bar(x=df5.index,y=df5.values)])
fig.update_traces(hoverinfo='y',text=df5.values,textfont_size=12,marker=dict(color=colors,line=dict(color='#000000', width=2)))
fig.update_layout(
    autosize=False,
    width=900,
    height=500,font=dict(
        family="Courier New, monospace",
        size=15,
        color="#0c0b0c"),

    title_text='Family members registered for different categories - Virtual Run')
fig.show()

