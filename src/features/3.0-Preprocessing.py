#!/usr/bin/env python
# coding: utf-8

# ## Data Pre-processing
# This notebook includes the data pre-processing steps in the solar energy project. <br>
# As inputs it requires the data files "Solar_Footprints_V2_7811899327930675815.csv" and "Population-Density By County". <br>
# As output it returns the cleaned, encoded, re-sampled, and scaled data frames X_train, X_test, y_train, y_test. <br>

# ### 1. Data collection

# #### 1.1. Import libraries

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt


# #### 1.2. Import data and display it

# In[4]:


path1 = r"..\data\raw\Solar_Footprints_V2_7811899327930675815.csv" # for Manuel
path2 = r"..\data\raw\Population-Density By County.csv" # for Manuel

dfs = pd.read_csv(path1, index_col="OBJECTID") # data frame for solar data
dfp = pd.read_csv(path2) # data frame for US population data

# Filter rows with "California" in the "GEO.display-label" column
dfp = dfp[dfp["GEO.display-label"].str.contains("California", na=False)]


# In[5]:


print("Solar DataFrame Shape:", dfs.shape)
display("Solar DataFrame:", dfs.head())
print("Population DataFrame Shape:", dfp.shape)
display("Population DataFrame:", dfp.head())


# #### 1.3. Check for missing values and rename columns

# In[6]:


dfs.columns
# Multiple columns can be dropped due to their lack of informative value. 
dfs=dfs.drop(['Combined Class','Substation Name GTET 100 Max Voltage','HIFLD ID (GTET 100 Max Voltage)','Substation Name GTET 200 Max Voltage',
         'HIFLD ID (GTET 200 Max Voltage)','Substation CASIO Name', 'HIFLD ID (CAISO)', 'Shape__Area', 'Shape__Length'],axis=1)


# In[7]:


# Multiple columns have very long names in dfs, which can be shortened
abbrev_dict = {"Distance to Substation (Miles) GTET 100 Max Voltage" : "Distance to GTET 100", # "GTET" likely stands for "Grid-Connected Transformer Equipment Terminal"           
               "Percentile (GTET 100 Max Voltage)" : "Percentile (GTET 100)",
               "Distance to Substation (Miles) GTET 200 Max Voltage" : "Distance to GTET 200",
               "Percentile (GTET 200 Max Voltage)" : "Percentile (GTET 200)",
               "Distance to Substation (Miles) CAISO" : "Distance to CAISO substation"}
dfs.rename(abbrev_dict, axis=1, inplace = True)
print(dfs.shape)
dfs.head()


# In[8]:


# Checking the missing values of both DataFrames
def print_missing_info(df, df_name):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / df.shape[0]) * 100
    print(f"\nMissing Values in {df_name}:\n{missing_count}")
    print(f"\nMissing Values (Percentage) in {df_name}:\n{missing_percentage}")

print_missing_info(dfs, "Solar Data")
print_missing_info(dfp, "Population Data")

# There are very few missing values in the solar dataframe while there are no missing values in the population dataframe.


# #### 1.4. Feature engineering

# In[9]:


# Constructing data frame for longitudes and latitudes of the counties
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapi")
california_counties=dfs.County.unique()
latitude = []
longitude = []

for county in california_counties:
    location = geolocator.geocode(f"{county}, California")
    latitude.append(location.latitude)
    longitude.append(location.longitude)

dgeo=pd.DataFrame({"County":dfs.County.unique(),"Latitude":latitude,"Longitude":longitude})

dgeo.head()


# In[10]:


# Population data frame processing
dfp=dfp[["GCT_STUB.display-label","Density per square mile of land area"]]
dfp.rename(columns={"GCT_STUB.display-label":"County","Density per square mile of land area":"Population Density"},inplace=True)
dfp.head()


# In[11]:


# Merge data frames to achieve final feature-engineered dataset
dfs_temp=pd.merge(dfs, dgeo, on="County", how="left").set_index(dfs.index)
df=pd.merge(dfs_temp, dfp, on="County", how="left")
df=df.set_index(dfs_temp.index)

print(df.shape)
df.head()


# ### 2. Exploratory Data Analysis

# #### 2.1. Basic overview

# In[12]:


# Basic Information about the full data frame
print(df.info(), "\n")  # Check for missing values and data types

print("There is", df.duplicated().sum() ,"duplicated data in the dataset.")

print("\nSummary Statistics:\n")
df.describe() # Summary statistics for numerical features

# The data types seem correct and we have no missing values anymore. But the dataset consists of data with outliers.


# #### 2.2. Boxplot: Distance of solar installations from substations

# In[13]:


# Create a boxplot matrix

def plot_boxplot_matrix():

    # Reshape the DataFrame from wide to long format in order to plot it clean
    df_with_index = df.reset_index()
    #print(df_with_index)
    df_long = df_with_index.melt(
        id_vars=["OBJECTID", "Install Type"],  # Keep "Install Type" as the identifier
        value_vars=[
            "Distance to GTET 100",
            "Distance to GTET 200",
            "Distance to CAISO substation"
        ],  # Columns to melt
        var_name="Substation Type",  # New column for substation types
        value_name="Distance"  # New column for distance values
    )

    #df_long

    fig = px.box(
        df_long,
        x="Install Type",  # X-axis: Install Type
        y="Distance",  # Y-axis: Distance
        color="Substation Type",  # Color by substation type
        facet_col="Substation Type",  # Create subplots for each substation type
        title="Boxplot: Distance to Substations by Install Type and Substation Type",
        labels={"Distance": "Distance to Substation (Miles)", "Install Type": "Install Type"},  # Customize axis labels
        width=1200,
        height=700,
        facet_col_spacing=0.05  # Reduce spacing between subplots
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Install Type",  # X-axis label
        yaxis_title="Distance to Substation (Miles)",  # Y-axis label
        showlegend=False,  # Show legend to differentiate substation types
        font=dict(size=12),  # Adjust font size for better readability
        boxmode="group",  # Group boxplots by install type
        #margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins to prevent overlap

    )

    # Adjust facet column spacing and titles
    fig.update_xaxes(matches=None, showticklabels=True)  # Ensure x-axis labels are shown for each subplot
    fig.update_yaxes(matches=None, showticklabels=True)  # Ensure y-axis labels are shown for each subplot
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # Clean up facet titles

    # Show the figure
    fig.show()
   
    # Save the figure as a PNG file (ONCE!)
    # pio.write_json(fig, "/home/ubuntu/dec24_bds_solar_energy/data/processed/Boxplotmatrix.json")
    # fig.write_image("/home/ubuntu/dec24_bds_solar_energy/reports/figures/Boxplotmatrix.png")

print("Boxplot matrix")
plot_boxplot_matrix()


# #### 2.3. Geographical distribution of solar panels

# In[14]:


# California Map initiator / Geolocator
import geopandas as gpd
url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
states = gpd.read_file(url)
california = states[states["name"] == "California"]


# In[15]:


# Geographical plot of Total solar installations per coounty
total_solar=df[["County" ,"Urban or Rural","Latitude","Longitude"]].value_counts().reset_index()
#total_solar.head()
total_solar.columns=["County" ,"Urban or Rural","Latitude","Longitude","frequency"]
total_solar_rural=total_solar[total_solar["Urban or Rural"]!="Urban"]
total_solar_urban=total_solar[total_solar["Urban or Rural"]!="Rural"]
#total_solar_rural.head()


# In[16]:


# Plotting
california.plot(edgecolor="black", facecolor="none", figsize=(8, 8))
plt.title("Number of solar panels")
plt.scatter(total_solar_rural.Longitude,total_solar_rural.Latitude,s=total_solar_rural.frequency,c="r",label="Rural")
plt.scatter(total_solar_urban.Longitude,total_solar_urban.Latitude,s=total_solar_urban.frequency,c="b",label="Urban",alpha=0.2)
plt.axis("off")
plt.legend();


# #### 2.4. Solar panels number versus population distribution

# In[17]:


solar_area=df.groupby("County")["Acres"].sum()
solar_area_df=pd.merge(solar_area,df.drop("Acres",axis=1),on=["County"],how="outer")
#solar_area_df.info()


# In[18]:


# Plotting
california.plot(edgecolor="black", facecolor="none", figsize=(8, 8))
plt.title("Relative area encroached by solar installations")
plt.scatter(solar_area_df.Longitude,solar_area_df.Latitude,s=solar_area_df.Acres/20,c="orange",label="area occupied by solar field")
plt.scatter(solar_area_df.Longitude,solar_area_df.Latitude,s=solar_area_df["Population Density"]/10,c="g",label="population density",alpha=0.2)
plt.legend()
plt.axis("off");


# #### 2.5. Area distribution by Urban/Rural

# In[19]:


# Distribution of acres in log scale
sns.set() # to change theme
sns.catplot(data=df, y="Acres", kind="box", hue="Urban or Rural")
plt.title("Solar power plant area distribution (log scale)")
plt.yscale("log")
plt.show();


# #### 2.6. Area distribution by Urban/Rural

# In[20]:


# Distribution of acres in log scale
sns.set() # to change theme
sns.catplot(data=df, y="Acres", kind="box", hue="Install Type")
plt.title("Solar power plant area distribution (log scale)")
plt.yscale("log")
plt.show();


# #### 2.7. Distribution of install types in urban and rural areas

# In[21]:


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 5))
sns.countplot(x=df[(df["Urban or Rural"]=="Urban")]["County"], hue="Install Type",data=df,ax=axes[0])
axes[0].set_title("Urban")
sns.countplot(x=df[(df["Urban or Rural"]=="Rural")]["County"], hue="Install Type",data=df,ax=axes[1])
axes[1].set_title("Rural")
plt.xticks(rotation=90);


# #### 2.8. Searching for correlation between target variable and longitude/latitude by grouping of counties

# In[22]:


df.head(1)


# In[23]:


# Create 3x3 grid based on longitude and latitude, then check for correlation with STI
# Step 1: Create latitude and longitude bins (3 bins each)
df['lat_bin'], lat_bins = pd.cut(df['Latitude'], bins=3, labels=['Low', 'Medium', 'High'], retbins=True)
df['lon_bin'], lon_bins = pd.cut(df['Longitude'], bins=3, labels=['Low', 'Medium', 'High'], retbins=True)
# Step 2: Create a unique Area ID (Grid Cell)
df['Area_ID'] = df['lat_bin'].astype(str) + "_" + df['lon_bin'].astype(str)
# Step 3: Check correlation between Area_ID and STI
# Convert STI to binary (Within -> 1, Outside -> 0)
df['STI_binary'] = df['Solar Technoeconomic Intersection'].map({'Within': 1, 'Outside': 0})

# Group by Area_ID and calculate the mean STI rate per area
area_correlation = df.groupby('Area_ID')['STI_binary'].mean()

#California Map initiator
min_lon, min_lat, max_lon, max_lat = california.total_bounds  # (west, south, east, north)


# In[24]:


#### Plot
fig, ax = plt.subplots(figsize=(8, 8))
california.plot(ax=ax, edgecolor='black', facecolor='none')
ax.scatter(df.Longitude,df.Latitude,s=df["Population Density"]/10,c='green',label='population density',alpha=0.2)
# Draw Grid Lines
for lon in lon_bins:
    ax.vlines(lon, ymin=min_lat, ymax=max_lat, color='red', linestyle="--", linewidth=1)
for lat in lat_bins:
    ax.hlines(lat, xmin=min_lon, xmax=max_lon, color='red', linestyle="--", linewidth=1)
# Label Each Grid Cell with Area_ID
for i, lat_cat in enumerate(['Low', 'Medium', 'High']):
    for j, lon_cat in enumerate(['Low', 'Medium', 'High']):
        center_lon = (lon_bins[j] + lon_bins[j+1]) / 2
        center_lat = (lat_bins[i] + lat_bins[i+1]) / 2
        area_id = f"{lat_cat}_{lon_cat}"
        area_count = df[df["Area_ID"]==area_id]["Area_ID"].count()
        if area_id in area_correlation.keys():
            area_value = round(area_correlation[area_id],2)
        else:
            area_value=0
        area_text = f"{area_id}\nCount: {area_count}\nMean: {area_value}"
        ax.text(center_lon, center_lat, area_text, fontsize=10, color='blue', ha='center', va='center')

# Remove temporal columns of df
df = df.drop(["lat_bin", "lon_bin", "Area_ID", "STI_binary"], axis=1)
# display(df)

# Display Plot
plt.title("California Divided into 3x3 Grid")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show();
print("The plot groups counties based on their latitude and longitude into 1 of 9 categories.") 
print("The number of counties appears as count.")
print("The mean Solar Technoeconomic Intersection of the counties in the sector is displayed in mean.")
print("There is no apparent correlation between sectors and mean STI.")


# ### 3. Statistical tests

# In[25]:


# On basis of the previous findings some relationships could be further examined by statstical tests:
# 1. Relationship between Population Density (continous, non-normal distributed variable) and the target variable (binary variable) -> Point-Biserial Correlation
# 2. Relationship between Distance GTET 200 (continous, non-normal distributed variable) and the target variable (binary variable) -> Point-Biserial Correlation

# Since they were not included in the correlation analysis (because they are categorical features), additional relationships could be explored:
# 3. Relationship between Install Type (categorical variable) and the target variable (binary variable) -> Chi-Square Test of Independence
# 4. Relationship between "Urban and Rural" (categorical variable) and the target variable (binary variable) -> Chi-Square Test of Independence

import scipy.stats as stats
from scipy.stats import pointbiserialr

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# 1. Point-Biserial Correlation

# Check the normality of "Population Density"

# Shapiro-Wilk test for normality
shapiro_pop_density = stats.shapiro(df["Population Density"])
print(f"Shapiro-Wilk test for Population Density: Statistic={shapiro_pop_density.statistic}, p-value={shapiro_pop_density.pvalue}")

# If the p-value < 0.05, the data is not normally distributed
if shapiro_pop_density.pvalue < 0.05:
    print("Population Density is NOT normally distributed (reject the null hypothesis). Proceeding with non-parametric methods.")
else:
    print("Population Density is normally distributed (fail to reject the null hypothesis).")

# Perform Point-Biserial Correlation between "Population Density" and the binary target variable "Solar Technoeconomic Intersection"
corr_pop_density, p_value = stats.pointbiserialr(df["Population Density"], df["Solar Technoeconomic Intersection"].apply(lambda x: 1 if x == "Within" else 0))

# Print the results
print(f"\nPoint-Biserial Correlation between Population Density and Solar Technoeconomic Intersection: {corr_pop_density}")
print(f"p-value: {p_value}")

# Interpret the correlation result
if p_value < 0.05:
    print("There is a significant relationship between Population Density and Solar Technoeconomic Intersection.", "\n\n")
else:
    print("There is no significant relationship between Population Density and Solar Technoeconomic Intersection.", "\n\n")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# 2. Point-Biserial Correlation

# Check the normality of "Distance to GTET 200"

# Shapiro-Wilk test for normality
shapiro_Distance_GTET_200 = stats.shapiro(df["Distance to GTET 200"])
print(f"Shapiro-Wilk test for Distance to GTET 200: Statistic={shapiro_Distance_GTET_200.statistic}, p-value={shapiro_Distance_GTET_200.pvalue}")

# If the p-value < 0.05, the data is not normally distributed
if shapiro_Distance_GTET_200.pvalue < 0.05:
    print("Distance to GTET 200 is NOT normally distributed (reject the null hypothesis). Proceeding with non-parametric methods.")
else:
    print("Distance to GTET 200 is normally distributed (fail to reject the null hypothesis).")

# Perform Point-Biserial Correlation between "Distance to GTET 200" and the binary target variable "Solar Technoeconomic Intersection"
corr_Distance_GTET_200, p_value = stats.pointbiserialr(df["Distance to GTET 200"], df["Solar Technoeconomic Intersection"].apply(lambda x: 1 if x == "Within" else 0))

# Print the results
print(f"\nPoint-Biserial Correlation between Distance to GTET 200 and Solar Technoeconomic Intersection: {corr_Distance_GTET_200}")
print(f"p-value: {p_value}")

# Interpret the correlation result
if p_value < 0.05:
    print("There is a significant relationship between Distance to GTET 200 and Solar Technoeconomic Intersection.", "\n\n")
else:
    print("There is no significant relationship between Distance to GTET 200 and Solar Technoeconomic Intersection.", "\n\n")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# 3. Chi-Square Test for categorical variable "Install Type"
from scipy.stats import chi2_contingency

# Create contingency table and perform the Chi-Square test
contingency_table = pd.crosstab(df["Install Type"], df["Solar Technoeconomic Intersection"])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square statistic between 'Install Type' and 'Solar Technoeconomic Intersection': {chi2}, p-value: {p}", "\n\n")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# 4. Chi-Square Test for categorical variable "Urban or Rural"

# Create contingency table and perform the Chi-Square test
contingency_table = pd.crosstab(df["Urban or Rural"], df["Solar Technoeconomic Intersection"])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square statistic between 'Urban or Rural' and 'Solar Technoeconomic Intersection': {chi2}, p-value: {p}", "\n\n")


# ### 4. Pre-Model preparation (Split, Encoding, Re-Sampling, Scaling)

# #### 4.1. Encoding

# In[26]:


# Values of categorical variables:
print("Install Type: ",df["Install Type"].unique())
print("Urban or Rural: ",df["Urban or Rural"].unique())
print("Percentile (GTET 100): ",df["Percentile (GTET 100)"].unique())
print("Percentile (GTET 200): ",df["Percentile (GTET 200)"].unique())
print("Percentile (CAISO): ",df["Percentile (CAISO)"].unique())
print("Solar Technoeconomic Intersection: ",df["Solar Technoeconomic Intersection"].unique())


# In[27]:


# Variables with Percentile will be ordinally encoded
from sklearn.preprocessing import OrdinalEncoder

df_enc=df

encoder_ordinal = OrdinalEncoder(categories=[["0 to 25th","25th to 50th","50th to 75th","75th to 100th"]])

df_enc["percentile_GTET100"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (GTET 100)"]])).set_index(df_enc.index)
df_enc["percentile_GTET200"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (GTET 200)"]])).set_index(df_enc.index)
df_enc["percentile_CAISO"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (CAISO)"]])).set_index(df_enc.index)

# get dummies for the variable "install type" (XXX Avoiding multicollinearity: If we do not drop one category (e.g the first one by: drop_first=True), it will introduce a dummy variable 
# trap, meaning the created features will be highly correlated (or linearly dependent) with each other. This can lead to issues with model training, especially when using linear models
# like logistic regression. In this case the Install Type "Ground" is dropped and the remaining categories are used as binary (0/1) columns. This ensures also that there is no redundant
# information in the dataset.)
df_enc = pd.concat([df_enc, pd.get_dummies(df_enc["Install Type"], dtype=int, drop_first=True)], axis=1)

# Variables "Urban or Rural" and "Solar technoeconomic Intersection" are simply converted to binary
df_enc["Urban or Rural"]=df_enc["Urban or Rural"].map({"Urban":0,"Rural":1}).astype(int)
df_enc["Solar Technoeconomic Intersection"]=df_enc["Solar Technoeconomic Intersection"].map({"Within":1,"Outside":0}).astype(int)

# We also drop "County" variable
df_enc=df_enc.drop(["Percentile (GTET 100)","Percentile (GTET 200)","Percentile (CAISO)","Install Type","County"],axis=1)

# Reordering the columns
df_enc=pd.concat([df_enc.drop("Solar Technoeconomic Intersection",axis=1),df_enc["Solar Technoeconomic Intersection"]],axis=1)

df_enc.tail()


# #### 4.2. Train-test-split

# In[28]:


# We are performing the train-test-split before other model pre-processing steps in order to prevent data leakage and overfitting

from sklearn.model_selection import train_test_split

X=df_enc.drop("Solar Technoeconomic Intersection",axis=1)
y=df_enc["Solar Technoeconomic Intersection"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# After the encoding the dataset consists only of numerical columns
print(X_train.dtypes)


# #### 4.3. Correlation analysis

# In[29]:


# Create correlation Heatmap (for numerical features of the training data)

def plot_correlation_heatmap(dataframe):

    # Calculate the correlation coefficients (we do only have numeric columns now after the encoding)
    corr = dataframe.corr(numeric_only=True)

    # Use mask to cover redundant values
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.mask(mask)
    #print(mask, corr.values, corr_masked.values)
    
    # Flip the correlation matrix and the mask vertically
    corr_flipped = np.flip(corr_masked.values, axis=0)
    mask_flipped = np.flip(mask, axis=0)

    # Create text for the heatmap tiles
    text_values = np.where(mask_flipped, "", np.around(corr_flipped, 2))  # Set text to empty string for masked tiles

    #print(text_values) # this will be flipped in go.Heatmap() automatically

    fig = go.Figure(data=go.Heatmap(
        x=corr.columns, 
        y=corr.columns[::-1],
        z=np.flip(corr_masked.values, axis=0),  # Flip the correlation matrix vertically  # Apply mask to hide upper triangle
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=dict(title="Correlation Coefficient"),
        text = text_values,
        texttemplate="%{text:.2f}",  # Format text to 2 decimal places
        hovertemplate=(
            "<b>X</b>: %{x}<br>"  # Feature name on the x-axis
            "<b>Y</b>: %{y}<br>"  # Feature name on the y-axis
            "<b>Correlation</b>: %{z:.2f}<br>"  # Correlation value (formatted to 2 decimal places)
            "<extra></extra>"  # Remove the extra hover info
        )
    )
    )
    
    fig.update_layout(
        title="Correlation Heatmap",
        height=700, width=1100,
        xaxis=dict(title="Features", showgrid=False),
        yaxis=dict(title="Features", showgrid=False)    
    )

    fig.show()
    
    # Save the figure as a PNG file (ONCE!)
    # pio.write_json(fig, "/home/ubuntu/dec24_bds_solar_energy/data/processed/Heatmap.json")
    # fig.write_image("/home/ubuntu/dec24_bds_solar_energy/reports/figures/Heatmap.png")
    
train_data = X_train.copy()  # Make a copy of X_train
train_data['Solar Technoeconomic Intersection'] = y_train  # Add target variable
print("Correlation heatmap for all numerical features")
plot_correlation_heatmap(train_data)


# In[30]:


# Since longitude and latitude have very little correlation with the target variable, we will drop them from the training set
X_train = X_train.drop(["Longitude","Latitude"], axis=1)
X_test = X_test.drop(["Longitude","Latitude"], axis=1)
display("X_train:", X_train.head())
display("X_test:", X_test.head())


# #### 4.4 Re-sampling

# In[31]:


# Countplot of the target variable

sns.countplot(x="Solar Technoeconomic Intersection",data=pd.DataFrame(y_train)) 
#sns.countplot(x="Solar Technoeconomic Intersection",data=pd.DataFrame(y_test)) # XXX We should not gaze into the test data
plt.show()

print(y_train.value_counts(normalize = True))
print("\nWe are dealing with imbalanced classes so that we need to apply oversampling or undersampling on the training data (we do not want to alter the test set).", "\n"
      "The over-sampling or under-sampling should happen after encoding and before scaling.")
#print("\nWe have an uneven distribution of classes in the target. To circuvent this moderate class imbalance, we will choose oversampling because we don't want to create unreal class elements using SMOTE. We don't want to remove data using undersampling.")


# ##### 4.4.1 Oversampling

# In[32]:


# I put your optional block here:
#
# "#optional block
# from imblearn.over_sampling import RandomOverSampler
# X_train_resampled, y_train_resampled = RandomOverSampler().fit_resample(X_train_norm, y_train)
# We can train the model either with (X_train_resampled, y_train_resampled) and (X_train_norm, y_train)"
#
# also into a function if you want to use it:
from imblearn.over_sampling import RandomOverSampler

def random_oversample(X_train, y_train):
    """
    Applies RandomOverSampler to balance the class distribution by over-sampling the minority class.

    Args:
    - X_train: Features of the training dataset.
    - y_train: Target labels of the training dataset.

    Returns:
    - X_train_resampled: Resampled feature set.
    - y_train_resampled: Resampled target labels.
    """
    # Apply RandomOverSampler
    X_train_resampled_rOs, y_train_resampled_rOs = RandomOverSampler().fit_resample(X_train, y_train)

    # Return the resampled data for further processing or model training
    return X_train_resampled_rOs, y_train_resampled_rOs
X_train_os,y_train_os=random_oversample(X_train, y_train)
sns.countplot(x=y_train_os)
sns.countplot(x=y_train)
X_train_os.shape,y_train_os.shape


# ##### 4.4.2 Undersampling

# In[33]:


# This code is here for documentation purposes, the under- and oversampling methods were compared 
# and there is virtually no difference in model performance, so we went with oversampling

# Re-sampling method: ClusterCentroids 
# This technique reduces the size of the majority class without randomly removing data, ensuring that the representative centroids of the 
# majority class are kept, which can lead to more efficient training and better performance in some cases.
from imblearn.under_sampling import ClusterCentroids
from collections import Counter

def cluster_centroids_undersample(X_train, y_train, random_state=888):
    """
    Applies ClusterCentroids undersampling on the majority class.

    Args:
    - X_train: Features of the training dataset.
    - y_train: Target labels of the training dataset.
    - random_state: Random seed for reproducibility (default is 888).

    Returns:
    - X_train_resampled_cc: Resampled feature set.
    - y_train_resampled_cc: Resampled target labels.
    """
    # Define the ClusterCentroids undersampler
    cluster_centroids = ClusterCentroids(random_state=random_state) #, n_init=10)
    
    # Apply the undersampling
    X_train_resampled_cc, y_train_resampled_cc = cluster_centroids.fit_resample(X_train, y_train)
    
    # Print the class distribution before and after resampling
    print("Original class distribution:", Counter(y_train))
    print("Resampled class distribution:", Counter(y_train_resampled_cc))
    
    # Plotting class distribution before and after resampling
    plt.figure(figsize=(8, 4))

    # Get the maximum y-value for setting the same y-limits
    original_counts = Counter(y_train).values()
    resampled_counts = Counter(y_train_resampled_cc).values()
    max_y_value = max(max(original_counts), max(resampled_counts))

    # Original distribution
    plt.subplot(1, 2, 1)
    plt.bar(Counter(y_train).keys(), original_counts, color="skyblue")
    plt.title("Original Class Distribution")
    plt.ylim(0, max_y_value)  # Set y-axis limit

    # Resampled distribution
    plt.subplot(1, 2, 2)
    plt.bar(Counter(y_train_resampled_cc).keys(), resampled_counts, color="lightcoral")
    plt.title("Resampled Class Distribution")
    plt.ylim(0, max_y_value)  # Set y-axis limit

    # Return the resampled data for further processing or model training
    return X_train_resampled_cc, y_train_resampled_cc

X_train_us, y_train_us = cluster_centroids_undersample(X_train, y_train)
# We can now use the resampled data (X_train_resampled_cc, y_train_resampled_cc) for model training.


# #### 4.5. Scaling

# ##### 4.5.1 Robust Scaling

# In[34]:


# Robust Scaling with a function using a pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def scale_data(X_train, X_test):

    # In our dataset all numerical columns contain outliers (need scaling)
    numerical_columns_with_outliers = [
        "Acres", "Distance to GTET 100", "Distance to GTET 200", 
        "Distance to CAISO substation", "Population Density"
    ]

    # Other columns that do not need to be scaled (all categorical)
    columns_not_to_scale = [
        "Urban or Rural", "percentile_GTET100", "percentile_GTET200", "percentile_CAISO", "Parking", "Rooftop"
    ]

    # Define the column transformer for scaling (for a production setup all other transformation steps like encoding etc. would also fit here)
    preprocessor = ColumnTransformer(
        transformers=[ 
            ("num_with_outliers", RobustScaler(), numerical_columns_with_outliers), # Apply RobustScaler to numerical columns with outliers      
            ("columns_not_to_scale", "passthrough", columns_not_to_scale) # Keep categorical columns without scaling
        ]
    )

    # Creating a pipeline with preprocessing step
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit the pipeline on the training data and transform X_train_resampled_cc
    # X_train_scaled = pipeline.fit_transform(X_train_us) # I think this should be X_train (added by Manuel)
    X_train_scaled = pipeline.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_columns_with_outliers + columns_not_to_scale)

    # Using the same preprocessor to transform the test set
    X_test_scaled = pipeline.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_columns_with_outliers + columns_not_to_scale)

    # Check the scaled data shape
    print(f"Scaled X_train shape: {X_train_scaled.shape}")
    print(f"Scaled X_test shape: {X_test_scaled.shape}")

    pd.set_option('display.float_format', '{:.2f}'.format) # avoiding notation like 2.570000e+03 in the output
    display("X_train_scaled_df:", X_train_scaled_df.describe())  # We can see that the training data is now scaled
    display("X_test_scaled_df:", X_test_scaled_df.describe())  # We can see that the test data is now scaled, too and also has the same columns!
    
    return X_train_scaled_df, X_test_scaled_df

X_train_os_robust, X_test_robust = scale_data(X_train_os, X_test)
X_train_us_robust, X_test_robust = scale_data(X_train_us, X_test)


# In[35]:


print(X_train_os.columns)
print('Columns Acres, Distance to GTET 100,Distance to GTET 200, Distance to CAISO substation, and Population Density are the coulmns to be scaled.' )
print('None of these columns should have negative entries. Robust scaler is not suitable for any of them.')
print('Moreover the so-called outliers in distance distribution are not actual outliers. Ouliers are considered when one suppose of have white (normal) or colored distribution')
print('min-max scaler will hopefully do fine.')


# In[36]:


# MinMax scalling
from sklearn.preprocessing import MinMaxScaler

X_train_os_minmax = X_train_os.copy()
X_test_minmax = X_test.copy()
scaler = MinMaxScaler()
X_train_os_minmax[['Acres','Distance to GTET 100','Distance to GTET 200', 'Distance to CAISO substation','Population Density']] = scaler.fit_transform(X_train_os[['Acres','Distance to GTET 100','Distance to GTET 200', 'Distance to CAISO substation','Population Density']])
X_test_minmax[['Acres','Distance to GTET 100','Distance to GTET 200', 'Distance to CAISO substation','Population Density']] = scaler.transform(X_test[['Acres','Distance to GTET 100','Distance to GTET 200', 'Distance to CAISO substation','Population Density']])


# In[37]:


X_test.head()


# ### 5. Export Data

# In[38]:


def export_data(list_of_dfs, list_of_names=None, path=''):
    """ For every df in list_of_dfs, we will perform df.to_csv to the specified path. 
    A list of names can be provided to each df, else the file names will be the variable names
    """
    for i, dataframe in enumerate(list_of_dfs):
        if list_of_names is None:
            print(f'{path}dataframe{i}')
            dataframe.to_csv(f'{path}dataframe{i}.csv', index=False)
        else:
            dataframe.to_csv(f'{path}{list_of_names[i]}.csv', index=False)
    return

data_path = r"..\data\processed\\" # save it to processed folder

#data_path = "~/dec24_bds_solar_energy/data/processed/" # save it to processed folder (for Eren)

#export_data([X_train_scaled_df, X_test_scaled_df, y_train_resampled_cc, y_test], list_of_names=["X_train", "X_test", "y_train", "y_test"], path=data_path)
export_data([X_train_os_minmax, X_test_minmax, y_train_os, y_test], 
            list_of_names=["X_train_os_minmax", "X_test_minmax", "y_train_os", "y_test"], 
            path=data_path)
export_data([X_train_us_robust, X_test_robust, y_train_us, y_test], 
            list_of_names=["X_train_us_robust", "X_test_robust", "y_train_us", "y_test"], 
            path=data_path)
export_data([X_train_os_robust, X_test_robust, y_train_os, y_test], 
            list_of_names=["X_train_os_robust", "X_test_robust", "y_train_os", "y_test"], 
            path=data_path)


# In[ ]:




