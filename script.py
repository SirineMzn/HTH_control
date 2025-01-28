#This is a clean version of HTH 
import pyodbc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.neighbors import NearestNeighbors
import anthropic
import re
import time
import os 
######################### FUNCTIONS DEFENETIONS ##########################################################################
def get_the_class(num):
    left_num = int(str(num)[0])
    return str(left_num)
def transform_account(num):
    three_num = str(num)[0:3]
    return three_num
def label_encode(X, categorical_columns):
    
    X_encoded = X.copy()
    for col in categorical_columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
    return X_encoded
def get_preprocessor(numerical_cols, categorical_cols):
    return make_column_transformer(
        (StandardScaler(), numerical_cols),  # Numeric columns normalization
        (FunctionTransformer(lambda x: x), categorical_cols),  # Include encoded categorical columns as they are
        sparse_threshold=0
    )
def knn_anomaly_detection(X, n_neighbors=5):
    """
    KNN pour la détection d'anomalies sans boucle.

    Parameters:
    -----------
    X : ndarray
        Les données d'entrée.
    n_neighbors : int
        Nombre de voisins à considérer.

    Returns:
    --------
    mean_distances : ndarray
        Moyenne des distances aux voisins les plus proches pour chaque point.
    distances : ndarray
        Distances aux voisins pour chaque point.
    indexes : ndarray
        Indices des voisins pour chaque point.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1)
    nbrs.fit(X)
    distances, indexes = nbrs.kneighbors(X)
    
    mean_distances = distances.mean(axis=1)
    
    return mean_distances, distances, indexes
def analyze_anomaly_dimensions(X_original, X_encoded, anomaly_indices, neighbor_indices,colonnes_selectionnees,user_dimension):
    """
    Analyse les dimensions contribuant le plus aux anomalies

    Parameters:
    -----------
    X_original : DataFrame original avant encoding
    X_encoded : Array après encoding/scaling
    anomaly_indices : Indices des anomalies détectées
    neighbor_indices : Indices des k plus proches voisins

    Returns:
    --------
    DataFrame avec les dimensions et leurs pourcentages de contribution pour chaque anomalie
    """
    results = []
    # we analyze only the first 20 anomalies 
    for idx in anomaly_indices[:20]:
        # Anomaly point and its neighbors
        point = X_encoded[idx]
        neighbors = X_encoded[neighbor_indices[idx]]

        # Calculating contributions by mean squared distance
        dim_contributions = np.mean((point - neighbors) ** 2, axis=0)
        total_contribution = np.sum(dim_contributions)

        # Calculating contribution percentages
        dim_percentages = (dim_contributions / total_contribution) * 100

        

        # Combine dimensions and percentages into a single list
        dimensions_with_percentages = [
            (colonnes_selectionnees[i], dim_percentages[i]) for i in range(len(colonnes_selectionnees))
        ]

        # Sort by percentage contribution in descending order
        dimensions_with_percentages.sort(key=lambda x: x[1], reverse=True)
        original_values = X_original.iloc[idx].to_dict()
         # Si une dimension spécifique est fournie, remplacer sa valeur par celle de X_encoded
        if user_dimension and user_dimension in colonnes_selectionnees:
            col_index = colonnes_selectionnees.index(user_dimension)
            original_values[user_dimension] = X_encoded[idx, col_index]

        # Create a result object
        anomaly_result = {
            'anomaly_index': idx,
            'original_values': original_values,  #original values
            'dimensions_contributions': dimensions_with_percentages  # dimensions and %
        }

        results.append(anomaly_result)


    return pd.DataFrame(results)
def find_ids_for_points(indexes, y,id_column):
    ids = y.loc[indexes,id_column].tolist()
            
    return ids if ids else None  # Return None if no IDs found
###################################################################### MAIN FUNCTION ##############################################################################################
def ExecuteTR(server,database,username,password,source,nombre_top_indices ,attm_cle,attm_createur, desc_cle,desc_createur, *args):
## FIRST DATA EXTRACTION ################################################################################################
    user_dimension = None
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    query='SELECT '
    select_clauses = []
    for arg in args:
        if not arg.startswith("IS_DIMENSION_"): 
            id_column = arg
        arg_without_is_dimension = arg.replace("IS_DIMENSION_", "")
        select_clauses.append(arg_without_is_dimension )  
    if source.startswith('TSMCG_ECRITURE_MOUVEMENT'):
        lib = 'CG_ECRM_MOUVEMENT_LIBELLE'
        general_account = 'CG_ECRM_COMPTE_GENERAL'
        user_dimension = 'CG_ECRM_ECRITURE_CREATED_BY'
        devise = 'CG_ECRM_DEVISE'
        select_clauses.append(general_account)
    if source.startswith('TALNF_NOTES_FRAIS') :
        lib = 'CALNF_NOTF_LIBELLE'  
        devise = 'CALNF_NOTF_DEVISE'
        general_account = None
    select_clauses.append(devise)
    select_clauses.append(lib)
    

    query += ' , '.join(select_clauses)
    query+=' FROM ' + source #toute la partie FROm avec les join, where,...
    print (query)
    Y = pd.read_sql(query, cnxn)
###########################DATA PREPROCESSING##############################################################################
    if Y.empty:
        print(f"THE'{query}' CANNOT SELECT ANY DATA")
    else:
        
        #Select columns where the noun starts with IS_DIMENSION_
        colonnes_selectionnees =[colonne[len("IS_DIMENSION_"):] for colonne in args if colonne.startswith("IS_DIMENSION_")]
        empty_columns = Y.columns[Y.isnull().all()]
        Y = Y.drop(empty_columns, axis=1).dropna(how="any")
        Y_filtered = Y.dropna(subset=colonnes_selectionnees)
        Y_filtered= Y_filtered.reset_index(drop=True)
        X = Y_filtered[colonnes_selectionnees]
        
        # Recreate X without the empty columns
        # Now X has the dimensions to encode + the mouvement libelle
        
        if 'CG_ECRM_COMPTE_GENERAL_THEORIQUE' in X.columns:
            X['Classe_compte'] = X['CG_ECRM_COMPTE_GENERAL_THEORIQUE'].apply(lambda x: get_the_class(x))
            X['CG_ECRM_COMPTE_GENERAL_THEORIQUE'] = X['CG_ECRM_COMPTE_GENERAL_THEORIQUE'].apply(lambda x: transform_account(x))
        # Loop to verify that string columns are not numeric, and convert them if they are
        for col in X.columns:
            # VERIFY IF THE COLUMN IS CATEGORICAL
            if X[col].apply(lambda x: isinstance(x, str) or pd.isna(x)).all():
                print(f"La colonne '{col}' est catégorique (lettres, chiffres ou caractères spéciaux). Elle est conservée telle quelle.")
                continue 
           
            print(f"Avant conversion : {col} -> {X[col].dtype}")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            print(f"Après conversion : {col} -> {X[col].dtype}")
              
        numerical_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns ] # Remove the mouvement libelle column
        X_encoded= label_encode(X, categorical_columns)
        preprocessor = get_preprocessor(tuple(numerical_columns), tuple(categorical_columns))
        X_reduced = preprocessor.fit_transform(X_encoded)
        
#################### knn algorithm #######################################################################################
        mean_distances,distances,neighbor_indices = knn_anomaly_detection(X_reduced)
        # Get the top X anomalies by KNN
        sorted_indices = X.index[np.argsort(mean_distances)[::-1][:int(nombre_top_indices)]]
        anomaly_indices = sorted_indices.tolist()
        top_X_anomalies = X.loc[sorted_indices]
        
############################## Dimensions analysis #######################################################################
        colonnes_resultats = colonnes_selectionnees.copy()
        colonnes_resultats.append(lib)
        if general_account :
            colonnes_resultats.append(general_account)
        results = analyze_anomaly_dimensions(
    X_original=Y_filtered[colonnes_resultats],  
    X_encoded=X_reduced,
    anomaly_indices=anomaly_indices,
    neighbor_indices=neighbor_indices,
    colonnes_selectionnees=colonnes_selectionnees,
    user_dimension=user_dimension
)
##################################################### PROMPT ##############################################################
        prompt_ec = f"""You are an expert data analyst specializing in anomaly detection and interpretation. Your task is to analyze anomalous points detected by a KNN algorithm and provide clear business-focused explanations of why these points are considered anomalies.

Important note about account numbers:
- CG_ECRM_COMPTE_GENERAL_THEORIQUE represents the standardized PCG account number used for analysis
- CG_ECRM_COMPTE_GENERAL represents the client's actual account number
- Use CG_ECRM_COMPTE_GENERAL_THEORIQUE for your analysis of the anomaly
- In your justification output, reference CG_ECRM_COMPTE_GENERAL (client's account) instead
- These numbers may be different for international clients due to account mapping
- For French clients using PCG, both account numbers will be identical
- Never use proper names of people given in CG_ECRM_ECRITURE_CREATED_BY, say 'the user' instead
Analyze each anomaly following these steps:
1. Identify dimensions with highest impact from the impact distribution
2. Examine the result line to understand specific values for these high-impact dimensions
3. Consider business significance of these unusual values within French accounting context
For each atypical result, provide a concise business justification that:
- Uses exactly 10-15 words
- Focuses on most impactful dimensions
- Explains business implications of unusual values
- Uses the client's account number (CG_ECRM_COMPTE_GENERAL) in the output text
Provide your justification within <justification> tags for each anomaly.
Focus on business insights rather than technical aspects of the KNN script.
Responses must be in US English.
Remember: Every explanation must support why the point is anomalous - never provide arguments that contradict its anomalous nature.
You will be given the following dimensions to analyze:
<dimensions>
{colonnes_selectionnees}
</dimensions>
"""
        prompt_ntf = f"""You are an expert data analyst tasked with explaining why specific results are flagged as anomalies by a KNN-based control system.
 
For each anomaly, follow these steps:
1. Identify the most deviant dimensions from the impact distribution
2. Focus on how these values significantly deviate from normal patterns
3. Explain why these deviations represent a business risk or unusual situation
 
Your response for each anomaly must:
- Be exactly 10-15 words
- Highlight what makes the result abnormal/concerning
- Focus only on the most impactful unusual dimensions
- Never justify or normalize the anomaly
 
Remember: Each explanation must emphasize the anomalous nature and never suggest the result is normal or expected.
 
Format: Provide justifications within <justification> tags.
Language: Use US English.
 
Available dimensions for analysis:
<dimensions>
{colonnes_selectionnees}
</dimensions>
"""
############################################################## LLM PART ########################################################
        if source.startswith('TSMCG_ECRITURE_MOUVEMENT'):
            prompt = prompt_ec
        else: 
            prompt = prompt_ntf
            API_KEY = os.environ.get("API_KEY")   

        client = anthropic.Anthropic(
            # par défaut os.environ.get("ANTHROPIC_API_KEY")
            api_key=API_KEY,
        )
        i = 0
        batch_size = 5
        prompt_batch = ""
        responses = []
        while i < 20:  # Treat only 20 anomalies
            # the subset of the results to process
            subset = results.iloc[i:i+batch_size]
            prompt_batch = prompt
            for j, row in subset.iterrows():
                prompt_batch += f"""

        ----
        <abnormal_point>
        {row['original_values']}
        </abnormal_point>

        <impact_distribution>
        {row['dimensions_contributions']}
        </impact_distribution>
                ----
                """
            client = anthropic.Anthropic(
            # par défaut os.environ.get("ANTHROPIC_API_KEY")
            api_key=API_KEY,
        )
            message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt_batch}
            ]
        )   
            time.sleep(1)
            flattened_responses = re.findall(r"<justification>(.*?)</justification>", message.content[0].text, re.DOTALL)
            responses.extend(flattened_responses)  # Ajoute directement les éléments aplatis dans responses
   
            prompt_batch = ""
            
            # Incremet by 5 until the end
            i += batch_size
        cleaned_b = [value.replace("\n", "") for value in responses]
        # responses is a list of inner lists, the inner list has the response for each anomaly of the subset of 5
########################################################################## FINAL TABLE TO INSERT ########################################
        # knn table
        top_X_acc_records = find_ids_for_points(anomaly_indices, Y_filtered,id_column)
        result_KNN = pd.DataFrame({'id_record':top_X_acc_records})
        result_KNN['Classement'] = range(1, len(result_KNN) + 1)
        # LLM table
        result_LLM = pd.DataFrame(result_KNN['Classement'])
        result_LLM['Justification'] = None
        
        
        # The case of not having 20 justifications 
        if len(cleaned_b) < 20:
            cleaned_b.extend(['AI is in preview mode, only available for the top 20 results'] * (20 - len(cleaned_b)))
        result_LLM.loc[:19,'Justification'] = cleaned_b[:20]
        result_LLM.loc[20:, 'Justification'] = 'AI is in preview mode, no justification available'
        #Now construct the final dataframe to insert in in TTEMR_RESULTATS
        
       
        attm_cle_justif = 88885  
        
        
        merged_result = pd.merge(result_KNN,result_LLM, on='Classement',how="left")
        alternated_rows = []
        for _, row in merged_result.iterrows():
            # one for the knn
            alternated_rows.append({
                "MR_RESU_ATTM_CLE": attm_cle,
                "MR_RESU_ATTM_CREATEUR": attm_createur,
                "MR_RESU_CLASSEMENT_ML": row["Classement"],
                "MR_RESU_DESC_CLE": desc_cle,
                "MR_RESU_DESC_CREATEUR": desc_createur,
                "MR_RESU_ID": row["Classement"],
                "MR_RESU_VALUE": row["id_record"]
            })
            # one for the LLM
            alternated_rows.append({
                "MR_RESU_ATTM_CLE": attm_cle_justif,
                "MR_RESU_ATTM_CREATEUR": attm_createur,
                "MR_RESU_CLASSEMENT_ML": row["Classement"],
                "MR_RESU_DESC_CLE": desc_cle,
                "MR_RESU_DESC_CREATEUR": desc_createur,
                "MR_RESU_ID": row["Classement"],
                "MR_RESU_VALUE": row["Justification"]
            })

        # Convert it into DataFrame
        resultats_ttemr = pd.DataFrame(alternated_rows)
#################################################################################################### INSERTION IN SQL TABLE ########################################################

        cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
        cursor = cnxn.cursor()       
        insert_query = """
    INSERT INTO [TTEMR_RESULTATS] (
        [MR_RESU_ATTM_CLE],
        [MR_RESU_ATTM_CREATEUR],
        [MR_RESU_CLASSEMENT_ML],
        [MR_RESU_DESC_CLE],
        [MR_RESU_DESC_CREATEUR],
        [MR_RESU_VALUE]
    ) VALUES (?, ?, ?, ?, ?, ?)
    """
        try:
            # Loop to insert each line in the SQL table
            for index, row in resultats_ttemr.iterrows():
                mr_resu_value = row['MR_RESU_VALUE']
                if row['MR_RESU_ATTM_CLE'] in ['147', '88885']:
                    mr_resu_value = f"'{mr_resu_value}'"
                formatted_query = insert_query.replace("?", "{}").format(
                    row['MR_RESU_ATTM_CLE'],
                    row['MR_RESU_ATTM_CREATEUR'],
                    row['MR_RESU_CLASSEMENT_ML'],
                    row['MR_RESU_DESC_CLE'],
                    row['MR_RESU_DESC_CREATEUR'],
                    mr_resu_value
                )
                
                mr_resu_value = row['MR_RESU_VALUE']
                if row['MR_RESU_ATTM_CLE'] in ['147', '88885']:
                    mr_resu_value = f"''{row['MR_RESU_VALUE']}''"
                cursor.execute(insert_query, row['MR_RESU_ATTM_CLE'], row['MR_RESU_ATTM_CREATEUR'],
                            row['MR_RESU_CLASSEMENT_ML'], row['MR_RESU_DESC_CLE'],
                            row['MR_RESU_DESC_CREATEUR'], row['MR_RESU_VALUE'])
            # Commit THE transactions
            cnxn.commit()
        except Exception as e:
                print(f"Erreur lors de l'insertion des données : {e}")
                cnxn.rollback() 
        finally:
        #Close the cursor and the connection
            cursor.close()
            cnxn.close()