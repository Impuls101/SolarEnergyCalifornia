import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid", palette="muted", context="talk")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    dfs = pd.read_csv('data/raw/Solar_Footprints_V2_7811899327930675815.csv', index_col="OBJECTID")
    dfp = pd.read_csv('data/raw/Population-Density By County.csv')
    dfp = dfp[dfp["GEO.display-label"].str.contains("California", na=False)]
    
    dfs = dfs.drop(['Combined Class', 'Substation Name GTET 100 Max Voltage', 'HIFLD ID (GTET 100 Max Voltage)',
                    'Substation Name GTET 200 Max Voltage', 'HIFLD ID (GTET 200 Max Voltage)', 'Substation CASIO Name',
                    'HIFLD ID (CAISO)', 'Shape__Area', 'Shape__Length'], axis=1)
    abbrev_dict = {"Distance to Substation (Miles) GTET 100 Max Voltage": "Distance to GTET 100",
                   "Percentile (GTET 100 Max Voltage)": "Percentile (GTET 100)",
                   "Distance to Substation (Miles) GTET 200 Max Voltage": "Distance to GTET 200",
                   "Percentile (GTET 200 Max Voltage)": "Percentile (GTET 200)",
                   "Distance to Substation (Miles) CAISO": "Distance to CAISO substation"}
    dfs.rename(abbrev_dict, axis=1, inplace=True)
    
    dfp = dfp[["GCT_STUB.display-label", "Density per square mile of land area"]]
    dfp.rename(columns={"GCT_STUB.display-label": "County", "Density per square mile of land area": "Population Density"}, inplace=True)
    
    dgeo = pd.read_csv('data/raw/dgeo.csv', index_col=0)
    dfs_temp = pd.merge(dfs, dgeo, on="County", how="left").set_index(dfs.index)
    df = pd.merge(dfs_temp, dfp, on="County", how="left").set_index(dfs_temp.index)
    
    return dfs, dfp, dgeo, df

dfs, dfp, dgeo, df = load_and_preprocess_data()

st.title("Solar Energy Generation in California 🌞")
st.sidebar.title("Table of contents")
pages=["1. Introduction", "2. Data Pre-Processing", "3. Data Visualization", "4. Model Preprocessing", "5. Modeling and Evaluation", "6. Conclusion", "7. Bonus: Stacking Approach"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0]: 
    st.write("## Introduction")
    st.write("#### Welcome to the 'Solar Energy Power Plants in California' project!")
    st.write("""
            This project uses a dataset containing detailed information about solar power plants in California, with a focus on their geographic distribution,
            technical characteristics, and proximity to critical energy infrastructure such as substations.
             
            In our approach we focus on creating a binary classification machine learning model which, based on this dataset, predicts the feasibility of solar power plant
            installations for solar energy.
             """)

    st.image("StreamlitX/CaliforniaStreamlitIntro.png") #https://www.arcgis.com/apps/mapviewer/index.html?panel=gallery&layers=9398e39a0424434b9e95ccf8e8938807
    st.caption("Population Density Map of California overlapped with Solar Installations Types (Green: Ground, Blue: Parking, Red: Rooftop)")

    st.write("**Key Features:**")
    st.markdown("""
    *   Location information, including county and urban/rural classification.
    *   Proximity to high-voltage substations (≥100 kV and ≥200 kV) and CAISO substations.
    *   Solar installation type (e.g., rooftop) and area in acres.
    *   Percentile rankings for distances to substations.
    """)

    st.write("**Target:**")
    st.write("The Solar Technoeconomic Intersection column is the target variable because it classifies the installation's suitability for solar energy production.")

if page == pages[1]: 
    st.write("## 2. Data Pre-Processing") 
    st.write("The dataset comprises 5,397 rows and 21 columns (excluding the OBJECTID index column). First we cleaned the data and enhanced it by adding features.")
    st.write('#### Solar panels data after dropping unnecessary columns and shortening column names')
    if st.checkbox('Solar Data'):
        st.dataframe(dfs.head())

    st.write('#### Population density data of counties in California')
    if st.checkbox('Population Density Data'):
        st.dataframe(dfp.head())

    st.write("#### Geographical data of the counties in California")
    st.write("Like population density, the geographical location may have influence on the Solar stations")
    if st.checkbox('Geographical Data'):
        st.dataframe(dgeo.head())
    
    st.write('#### Merged DataFrame')
    st.dataframe(df.head(10))
    st.write("#### DataFrame Description")
    st.write("DataFrame Shape: ", df.shape) # the df.info() information is already in df.shape and df.isna().sum(). And the feature types are given in df.dtypes
    st.dataframe(df.describe())

    if st.checkbox("Show Data Types and NULL values"):
        st.write(df.dtypes)
        st.dataframe(df.isna().sum())


if page == pages[2]:
    st.write("## 3. Data Visualization")

    fig = plt.figure()
    sns.countplot(x = 'Solar Technoeconomic Intersection', data = df)
    st.pyplot(fig)

    fig2 = plt.figure()
    sns.countplot(x = 'Install Type', data = df)
    plt.title("Distribution of Install Type")
    st.pyplot(fig2)

    import geopandas as gpd
    from PIL import Image
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    states = gpd.read_file(url)
    california = states[states["name"] == "California"]

    # Geographical plot of Total solar installations per coounty
    total_solar=df[["County" ,"Urban or Rural","Latitude","Longitude"]].value_counts().reset_index()
    total_solar.columns=["County" ,"Urban or Rural","Latitude","Longitude","frequency"]
    total_solar_rural=total_solar[total_solar["Urban or Rural"]!="Urban"]
    total_solar_urban=total_solar[total_solar["Urban or Rural"]!="Rural"]

    image_gtet=Image.open("StreamlitX/distance_gtet.png")
    image_area1=Image.open("StreamlitX/area_dist.png")
    image_area2=Image.open("StreamlitX/area_dist2.png")

    if st.checkbox('GTET distance'):
        fig= plt.figure(figsize=(18,10))
        plt.imshow(image_gtet)
        plt.axis("off")
        st.pyplot(fig)
    if st.checkbox('Area distribution'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 10))    
        axes[0].imshow(image_area1)    
        axes[1].imshow(image_area2)    
        st.pyplot(fig)

    # Plotting
    if st.checkbox("Number of solar panels"):
        fig=plt.figure(figsize=(8,8))
        #fig=california.plot(edgecolor="black", facecolor="none", figsize=(8, 8))
        plt.plot(california.get_coordinates().x,california.get_coordinates().y)
        #plt.title("Number of solar panels")
        plt.scatter(total_solar_rural.Longitude,total_solar_rural.Latitude,s=total_solar_rural.frequency,c="r",label="Rural")
        plt.scatter(total_solar_urban.Longitude,total_solar_urban.Latitude,s=total_solar_urban.frequency,c="b",label="Urban",alpha=0.2)
        plt.axis("off")
        plt.legend()
        st.pyplot(fig)

    if st.checkbox("Relative area encroached by solar installations"):
        solar_area=df.groupby("County")["Acres"].sum()
        solar_area_df=pd.merge(solar_area,df.drop("Acres",axis=1),on=["County"],how="outer")
        fig=plt.figure(figsize=(8,8))
        plt.plot(california.get_coordinates().x,california.get_coordinates().y)
        #plt.title("Relative area encroached by solar installations")
        plt.scatter(solar_area_df.Longitude,solar_area_df.Latitude,s=solar_area_df.Acres/20,c="orange",label="area occupied by solar field")
        plt.scatter(solar_area_df.Longitude,solar_area_df.Latitude,s=solar_area_df["Population Density"]/10,c="g",label="population density",alpha=0.2)
        plt.legend()
        plt.axis("off")
        st.pyplot(fig)


if page == pages[3]:
    st.write("## 4. Model Preprocessing")
    st.write("### Data Split and Encoding")
    st.write("""
            To ensure robust model performance, the dataset was split into training and testing sets, with the target variable 
             "Solar Technoeconomic Intersection" classifying areas as "Within" or "Outside." The training and testing split was 
             0% and 20% respectively, with the testing set preserving the original class distribution. 
             """)
    
    with st.expander("Description of the Encoding", expanded=False):
        st.markdown("""
            <div class="description-text">
            Categorical features in the dataset were encoded to convert them into a numerical format suitable for modeling. 
            Percentile-related variables ("Percentile (GTET 100)," "Percentile (GTET 200)," and "Percentile (CAISO)") were ordinally 
            encoded using predefined categories ("0 to 25th," "25th to 50th," "50th to 75th," "75th to 100th") to preserve their inherent order, 
            creating new columns ("percentile_GTET100," "percentile_GTET200," "percentile_CAISO"). The original percentile columns were then dropped.
            <br><br>
            The "Install Type" variable was one-hot encoded using dummy variables, with the "Ground" category dropped to avoid multicollinearity and the dummy variable trap, ensuring no redundant information for linear models. This resulted in binary columns (“Parking” and “Rooftop”) for the remaining categories.
            <br><br>
            The "Urban or Rural" and "Solar Technoeconomic Intersection" variables were binary encoded, mapping "Urban" to 0 and "Rural" to 1, and "Within" to 1 and "Outside" to 0, respectively, to create integer representations. The "County" variable was dropped due to its high cardinality and lack of informative value for the prediction. Finally, the dataset was reordered, placing the target variable as the last column for clarity.
            </div>
            """, unsafe_allow_html=True)
        
    @st.cache_data
    def encoding(df):
        from sklearn.preprocessing import OrdinalEncoder
        df_enc=df
        encoder_ordinal = OrdinalEncoder(categories=[["0 to 25th","25th to 50th","50th to 75th","75th to 100th"]])

        df_enc["percentile_GTET100"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (GTET 100)"]])).set_index(df_enc.index)
        df_enc["percentile_GTET200"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (GTET 200)"]])).set_index(df_enc.index)
        df_enc["percentile_CAISO"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (CAISO)"]])).set_index(df_enc.index)

        df_enc = pd.concat([df_enc, pd.get_dummies(df_enc["Install Type"], dtype=int, drop_first=True)], axis=1)
        df_enc["Urban or Rural"]=df_enc["Urban or Rural"].map({"Urban":0,"Rural":1}).astype(int)
        df_enc["Solar Technoeconomic Intersection"]=df_enc["Solar Technoeconomic Intersection"].map({"Within":1,"Outside":0}).astype(int)

        df_enc=df_enc.drop(["Percentile (GTET 100)","Percentile (GTET 200)","Percentile (CAISO)","Install Type","County"],axis=1)

        df_enc=pd.concat([df_enc.drop("Solar Technoeconomic Intersection",axis=1),df_enc["Solar Technoeconomic Intersection"]],axis=1)

        return df_enc
    
    st.write("Result of the encoding:")
    df_enc = encoding(df)
    st.dataframe(df_enc.head(10), use_container_width=True)

    from sklearn.model_selection import train_test_split
    import pandas as pd
    X=df_enc.drop("Solar Technoeconomic Intersection",axis=1)
    y=df_enc["Solar Technoeconomic Intersection"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    X_train = X_train.drop(["Longitude","Latitude"], axis=1)
    X_test = X_test.drop(["Longitude","Latitude"], axis=1)

    st.write("### Re-Sampling")
    st.write("""
        We are dealing with imbalanced classes so that we need to apply oversampling or undersampling on the training data.
        """)
    with st.expander("Description of the Re-Sampling Strategies", expanded=False):
        st.markdown("""
            <div class="description-text">
            Two sampling strategies were tested to address class imbalances and maintain data representativeness: oversampling and undersampling. 
            <br><br>
            The first strategy, random oversampling, targeted the minority class to balance the dataset. Synthetic examples of the minority class were generated, increasing its representation in the training set. The testing set remained unchanged to reflect the original class distribution, ensuring realistic evaluation. Oversampling was valuable for enhancing model performance on the minority class, critical for identifying suitable solar deployment areas.
            <br><br>
            The second strategy, undersampling with cluster centroids, reduces the size of the majority class without randomly removing data, ensuring that the representative centroids of the majority class are kept, which can lead to more efficient training and better performance in some cases. While this approach also addressed class imbalance, it risked losing valuable information. 
            <br><br>
            After evaluation, oversampling was selected as the final approach. In our case it provided a more robust solution for handling class imbalances without sacrificing data, ensuring the model generalizes well across both classes.
            </div>
            """, unsafe_allow_html=True)
        
    st.write("Before Re-Sampling:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="Solar Technoeconomic Intersection", data=df, ax=ax)
    st.pyplot(fig)
    st.write("Share of the classes:")
    st.dataframe(df["Solar Technoeconomic Intersection"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    from imblearn.over_sampling import RandomOverSampler
    @st.cache_data
    def random_oversample(X_train, y_train):
        X_train_resampled_rOs, y_train_resampled_rOs = RandomOverSampler().fit_resample(X_train, y_train)
        return X_train_resampled_rOs, y_train_resampled_rOs
    
    X_train_os,y_train_os=random_oversample(X_train, y_train)
    st.write("After Re-Sampling:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=y_train_os, ax=ax, color='blue', alpha=0.5, label='y_train_os')
    sns.countplot(x=y_train, ax=ax, color='orange', alpha=0.5, label='y_train')
    ax.legend()
    st.pyplot(fig)

    st.write("### Scaling")
    st.write("""
            Feature scaling was applied to standardize numerical features, such as "Acres," distances to substations, and "Population Density," which vary widely in scale. Two scaling strategies were tested: min-max scaling and robust scaling.
            """)
    with st.expander("Description of the Scaling Strategies", expanded=False):
        st.markdown("""
            <div class="description-text">
            The first strategy, min-max scaling, transformed features to a fixed range, typically [0, 1], by using the MinMaxScaler function of Scikit-learn. Scaling parameters were computed from the training data and applied to both training and testing sets to prevent data leakage. While min-max scaling is effective for features with known bounds, it is highly sensitive to outliers. Extreme values in features like "Acres" and distances to substations could compress the majority of the data into a narrow range, reducing the effectiveness of the scaling.
            <br><br>
            The second strategy, robust scaling, used the median and interquartile range (IQR) to transform features, making it less sensitive to outliers. This method subtracts the median and divides by the IQR (difference between the 75th and 25th percentiles), with parameters computed solely from the training data to ensure no leakage. Robust scaling preserved feature relationships despite extreme values, making it particularly suitable for this dataset, where outliers were present in several numerical features.
            <br><br>
            After evaluation, robust scaling was chosen as the final approach. It handled outliers more effectively than min-max scaling, ensuring fair feature contributions to models sensitive to magnitudes, such as logistic regression or neural networks. Categorical features, already encoded, were not scaled.
            </div>
            """, unsafe_allow_html=True)

    from sklearn.preprocessing import RobustScaler
    @st.cache_data
    def scale_data(X_train, X_test):
        numerical_columns_with_outliers = ["Acres", "Distance to GTET 100", "Distance to GTET 200", "Distance to CAISO substation", "Population Density"]    
        columns_not_to_scale = ["Urban or Rural", "percentile_GTET100", "percentile_GTET200", "percentile_CAISO", "Parking", "Rooftop"]

        scaler = RobustScaler()
        X_train[numerical_columns_with_outliers] = scaler.fit_transform(X_train[numerical_columns_with_outliers])
        X_test[numerical_columns_with_outliers] = scaler.transform(X_test[numerical_columns_with_outliers])

        return X_train, X_test

    X_train_scaled, X_test_scaled = scale_data(X_train_os, X_test)
    st.write("Result of the scaling:")
    st.dataframe(X_train_scaled)

    st.write("### Results of different Combinations of Re-Samplers and Scalers")
    st.write('##### Oversampled and  Robust Scaled')
    st.code("""
    Best RandomForest Params: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
    Best SVM Params: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
    Best XGBoost Params: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
    Best KNN Params: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}
    """)

    st.write('##### Undersampled and Robust Scaled')
    st.code("""
    Best RandomForest Params: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
    Best SVM Params: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
    Best XGBoost Params: {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 50}
    Best KNN Params: {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
    """)

    st.write('##### Oversampled and Standard Scaled')
    st.code("""
    Best RandomForest Params: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
    Best SVM Params: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
    Best XGBoost Params: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
    Best KNN Params: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}
    """)

    st.write('#### Summary Table')
    st.dataframe(pd.read_csv('StreamlitX/summary_table.csv', index_col=0))
    
    st.write("### Export of Pre-Processed Data")
    st.write("""
            Once the sampling and scaling processes were complete, the preprocessed data was exported to ensure accessibility and reproducibility for subsequent analysis and modeling phases. The training and testing sets (X_train, X_test, y_train, y_test) were saved as separate files in a structured format, facilitating their use by other team members and enabling seamless integration into the broader project workflow. 
            """)


if page == pages[4]:
    st.write("## 5. Modeling")
    choice = ['Random Forest', 'SVC', 'XGBoost', "KNN"]
    option = st.selectbox('Choice of a basic model', choice)
    st.write('The chosen basic model is :', option)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix, roc_auc_score, roc_curve
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    #path = r"..\data\processed\\" # for Manuel
    path = "~/dec24_bds_solar_energy/data/processed/" # for Eren

    X_train = pd.read_csv(path+'X_train_os_robust.csv')
    X_test = pd.read_csv(path+'X_test_robust.csv')
    y_train = pd.read_csv(path+'y_train_os.csv')
    y_test = pd.read_csv(path+'y_test.csv')

    y_train = y_train.values.ravel() # to prevent warning messages

    @st.cache_data
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier(random_state=42)
        elif classifier == 'SVC':
            clf = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
        elif classifier == 'XGBoost':
            clf = XGBClassifier(random_state=42)
        elif classifier == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        return clf
    
    @st.cache_data
    def scores(_clf, choice, set_type="Test Set"): # "_" tells Streamlit not to try and hash the clf object (which is not hashable); set_type default is "Test Set"
        if choice == 'Classification Report' and set_type == "Test Set":
            y_pred = _clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        elif choice == 'Classification Report' and set_type == "Training Set":
            y_pred = _clf.predict(X_train)
            report = classification_report(y_train, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        elif choice == 'Confusion Matrix' and set_type == "Test Set":
            cm = confusion_matrix(y_test, _clf.predict(X_test))
            fig = px.imshow(cm, 
                            labels={'x': 'Predicted', 'y': 'True'}, 
                            x=['Class 0', 'Class 1'],
                            y=['Class 0', 'Class 1'],
                            color_continuous_scale='Blues', 
                            text_auto=True,
                            title="Confusion Matrix")
            st.plotly_chart(fig)
        elif choice == 'Confusion Matrix' and set_type == "Training Set":
            cm = confusion_matrix(y_train, _clf.predict(X_train))
            fig = px.imshow(cm, 
                            labels={'x': 'Predicted', 'y': 'True'}, 
                            x=['Class 0', 'Class 1'],
                            y=['Class 0', 'Class 1'],
                            color_continuous_scale='Blues', 
                            text_auto=True,
                            title="Confusion Matrix")
            st.plotly_chart(fig)
        elif choice == "ROC Curve":
            y_pred_proba_test = _clf.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {test_roc_auc:.2f}',
                                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{text}',
                                    text=[f'{t:.2f}' for t in thresholds]))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True,
                height=800 
            )
            st.plotly_chart(fig)
        
    clf = prediction(option)
    display = st.radio('Choose an evaluation:', ('Classification Report', 'Confusion Matrix'))
    if display == 'Classification Report':
        scores(clf, display)
    elif display == 'Confusion Matrix':
        scores(clf, display)

    st.write("### GridSearchCV")
    st.code(
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import f1_score, make_scorer

        param_grids = {
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "KNN": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        }

        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "KNN": KNeighborsClassifier()
        }

        scorer = make_scorer(f1_score, average='binary')
        best_models = {}

        for name, model in models.items():
            print(f" Tuning {name} for F1-score...")
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring=scorer, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_models[name] = grid_search.best_estimator_
            print(f" Best {name} Params: {grid_search.best_params_}")
            print(f" Best {name} F1-score: {grid_search.best_score_:.4f}")

        # Evaluate the best models on the test set
        print("Final Model Evaluations on Test Data:")
        for name, model in best_models.items():
            y_train_pred = model.predict(X_train)
            f1 = f1_score(y_train, y_train_pred)
            print(f" {name} train: F1-score = {f1:.4f}")
            y_test_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_test_pred)
            print(f" {name} test: F1-score = {f1:.4f}")
        """
            )

    #import joblib
    #joblib.dump(clf, "models/model.joblib")
    #import pickle
    #pickle.dump(clf, open("models/model.pkl", 'wb'))
    # loaded_model = pickle.load(open("model", 'rb'))

    st.write("### Evaluation of the Best model after GridSearchCV")
    st.write("We decided for the SVC model with its best parameters ({'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}) because it has the lowest overfitting.")

    @st.cache_data
    def best_model_prediction():
            clf_best = SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42)
            clf_best.fit(X_train, y_train)
            return clf_best

    clf_best = best_model_prediction()
    display2 = st.radio('Choose an evaluation:', ('Classification Report', 'Confusion Matrix'), key=123)
    if display2 == 'Classification Report':
        scores(clf_best, display2, "Test Set")
    elif display2 == 'Confusion Matrix':
        st.write("For the Test Set:")
        scores(clf_best, display2, "Test Set")
        st.write("For the Training Set:")
        scores(clf_best, display2, "Training Set")
    
    st.write("### ROC Curve")
    scores(clf_best, "ROC Curve")



if page == pages[5]:
    st.write("## 6. Conclusion")

    st.image("reports/figures/SHAPplot.png")

    with st.expander("Description of the SHAP Plot", expanded=False):
        st.markdown("""
            <div class="description-text">
            Urban or Rural: This feature appears to be the most important predictor in the model. However, the concentration of red dots (high values - representing rural areas) on the negative SHAP value side suggests that being in an rural area negatively impacts the model's output (lower probability of solar energy feasibility).
            <br><br>
            Acres: The second most important feature is "Acres." Here, the cluster of blue dots on the positive SHAP value side suggests that lower acreages tend to positively impact the model's prediction. This could indicate that smaller properties are more likely to be suited for solar energy.
            <br><br>
            Distance to GTET 200 (“GTET” refers to “Grid-Connected Transformer Equipment Terminal”): Lower distances on this feature tend to have a positive impact on solar feasibility according to the SHAP values in the plot. 
            <br><br>
            Population Density: Higher population density seems to have in particular an negative impact on the model's output. This means that, according to the SVC model, in areas with high population density, the probability of solar energy feasibility decreases.
            <br><br>
            percentile_CAISO: This feature has a distribution of both high and low values contributing negatively and positively to the SHAP value, depending on if the value is low or high. This feature has more impact to the model’s output than the corresponding percentile_GTET100 or percentile_GTET200. 
            <br><br>
            Distance to GTET 100: This feature shows some effect by showing an increase for the model output when this distance decreases. 
            <br><br>
            percentile_GTET100 and percentile_GTET200: These percentile features seem to have both weak positive and negative impacts depending on their values.
            <br><br>
            Rooftop and Parking: These features have a positive SHAP value. “Parking” suggests that installations on parking lots are more likely to be feasible for solar energy. Higher values lead to a positive impact on the model's output. Similarly, "Rooftop" has a positive influence.
            <br><br>
            Distance to CAISO substation: According to the SHAP values, this feature has the lowest effect on the model output.</div>
            """, unsafe_allow_html=True)

    st.write("### Outlook")
    st.write("""
            The model could be deployed as a robust and scalable service through the use of technologies such as a Flask API and Docker, enabling seamless integration into a production environment. By exposing the model via an endpoint, it would facilitate real-time predictions on the techno-economic feasibility of solar power plant installations, leveraging incoming data to provide actionable insights for decision-makers.
            """)



if page == pages[6]:
    st.write("## 7. Bonus: Stacking Approach")
    st.write("""
            We also explored training a stacking classifier using nested cross-validation, with hyperparameter optimization performed via Optuna. 
            Although stacking is a powerful technique, it proved to be computationally expensive and led to overfitting, which were the primary reasons besides its complexity 
            for not pursuing this approach further. Below is a simplified breakdown of the key steps and components involved.
            """)
    
    st.write("1. Defining the Base Models and Meta-Learner")
    st.code("""
            # Define base models (level-0 models) and meta-learner (level-1 model)
            base_models = [
                ("rf", RandomForestClassifier),
                ("svc", SVC),
                ("knn", KNeighborsClassifier),
                ("xgb", XGBClassifier)
            ]

            meta_model = LogisticRegression(random_state=888)  # Meta-learner
            """)
    
    st.write("2. Hyperparameter Tuning with Optuna")
    st.code("""
            # Function to tune hyperparameters for the base models
            def tune_base_model_hyperparameters(trial, X_train, y_train, X_val, y_val):
                params = {
                    "rf": {
                        "n_estimators": trial.suggest_int("rf_n_estimators", 80, 1000)
                    },
                    "svc": {
                        "C": trial.suggest_float("svc_C", 0.01, 100.0, log=True),
                        "gamma": trial.suggest_float("svc_gamma", 0.001, 1.0, log=True)
                    },
                    "knn": {
                        "n_neighbors": trial.suggest_int("knn_n_neighbors", 1, 30),
                        "leaf_size": trial.suggest_int("knn_leaf_size", 5, 200)
                    },
                    "xgb": {
                        "n_estimators": trial.suggest_int("xgb_n_estimators", 5, 100),
                        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.001, 0.1)
                    }
                }

                # Tune models with suggested hyperparameters and evaluate on validation data
                base_models_tuned = []
                for name, model_class in base_models:
                    model_params = params[name]
                    model_params["random_state"] = 888
                    base_models_tuned.append((name, model_class(**model_params)))

                # Initialize the stacking classifier
                stacking_clf = StackingClassifier(
                    estimators=base_models_tuned,
                    final_estimator=meta_model,
                    cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=888),
                    n_jobs=-1
                )

                # Fit and evaluate on the validation set
                stacking_clf.fit(X_train, y_train)
                y_pred = stacking_clf.predict(X_val)
                return accuracy_score(y_val, y_pred)
                """)
    
    st.write("3. Outer loop for nested cross-validation")
    st.code("""
            def nested_cv_evaluation(X, y, outer_splits=5, inner_splits=5, n_trials=50, random_state=888):
                outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
                scores, auc_scores = [], []
                
                for fold_idx, (train_idx, val_idx) in enumerate(outer_kf.split(X, y)):
                    print(f"Outer fold {fold_idx + 1}/{outer_splits}")
                    
                    # Split data for outer loop
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Inner loop: hyperparameter tuning using Optuna
                    def objective(trial):
                        return tune_base_model_hyperparameters(trial, X_train, y_train, X_val, y_val)

                    # Perform hyperparameter tuning using Optuna
                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=n_trials, n_jobs=4)
                    
                    # Evaluate model on outer validation set
                    best_params = study.best_params
                    stacking_clf_final = StackingClassifier(
                        estimators=[("rf", RandomForestClassifier(**best_params))],
                        final_estimator=meta_model,
                        cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
                    )
                    stacking_clf_final.fit(X_train, y_train)
                    y_pred = stacking_clf_final.predict(X_val)
                    
                    scores.append(accuracy_score(y_val, y_pred))
                    auc_scores.append(roc_auc_score(y_val, y_pred_proba))
                    print(f"Outer fold {fold_idx + 1} accuracy: {scores[-1]:.4f}")
                    
                return {
                    "scores": scores,
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores)
                }

            """)
    
    st.write("4. Main Function")
    st.code("""              
            # Main function to tie everything together
            def main(X, y, outer_splits=5, inner_splits=5, n_trials=50, random_state=888):
                # Perform Nested Cross-Validation for model selection and hyperparameter tuning
                results = nested_cv_evaluation(X, y, outer_splits=outer_splits, 
                                            inner_splits=inner_splits, n_trials=n_trials, 
                                            random_state=random_state)
                
                # Print the results of the outer loop cross-validation
                print(f"Mean accuracy over {outer_splits} outer folds: {results['mean_score']:.4f}")
                print(f"Standard deviation of accuracy: {results['std_score']:.4f}")

                # Optionally: After the nested cross-validation, we can train the final model
                final_model = StackingClassifier(
                    estimators=[("rf", RandomForestClassifier(**best_params))],
                    final_estimator=meta_model,
                    cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
                )
                final_model.fit(X, y)

                # Feature importance analysis using SHAP
                background_summary = shap.kmeans(X, 1)  # Using k-means to summarize background data
                explainer = shap.KernelExplainer(final_model.predict_proba, background_summary)
                shap_values = explainer.shap_values(X)

                # SHAP summary plot
                shap.summary_plot(shap_values, X, max_display=10, show=True)
                
                return results, final_model
            """)
 
