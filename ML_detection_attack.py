import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import yagmail
import datetime
import os # Pour vérifier l'existence des fichiers

# Définir les noms de colonnes pour le dataset NSL-KDD
feature_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type' # Dernière colonne pour le type d'attaque
]

# Charger les datasets
try:
    train_data = pd.read_csv('KDDTrain+.txt', names=feature_names)
    test_data = pd.read_csv('KDDTest+.txt', names=feature_names)
except FileNotFoundError:
    print("Erreur: Assurez-vous que 'KDDTrain+.txt' et 'KDDTest+.txt' sont dans le même répertoire que le script.")
    exit()

print(f"Forme des données d'entraînement: {train_data.shape}")
print(f"Forme des données de test: {test_data.shape}")

# --- Correction de type pour 'duration' et 'dst_host_srv_rerror_rate' ---
for df in [train_data, test_data]:
    for col in ['duration', 'dst_host_srv_rerror_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)

# --- Correction de type pour 'attack_type' et identification de la classe 'normal' ---
# Convertir 'attack_type' en string pour s'assurer que le mapping fonctionne, même si ce sont des nombres
train_data['attack_type'] = train_data['attack_type'].astype(str)
test_data['attack_type'] = test_data['attack_type'].astype(str)

print("\n--- Analyse des types d'attaque avant le mapping binaire ---")
print("Veuillez examiner la distribution ci-dessous et identifier quelle(s) valeur(s) correspond(ent) au trafic 'normal'.")
print("Distribution des types d'attaques dans les données d'entraînement (après conversion en string):")
print(train_data['attack_type'].value_counts())
print("\nDistribution des types d'attaques dans les données de test (après conversion en string):")
print(test_data['attack_type'].value_counts())

# Mapper les types d'attaque en 'normal' (0) ou 'attack' (1)
# IMPORTANT: Vous DEVEZ modifier cette liste 'normal_attack_labels'
# pour inclure la/les valeur(s) exacte(s) (en tant que chaîne de caractères)
# qui représentent le trafic 'normal' dans VOTRE dataset.
# Basé sur des datasets NSL-KDD numériques courants, '0' ou '21' pourrait être 'normal'.
# Si après l'exécution, vous voyez que la valeur la plus fréquente est '21' et que c'est le trafic normal, mettez ['21'].
# normal_attack_labels = ['<IDENTIFIEZ_ICI_LE_LABEL_NORMAL_COMME_STRING>']
# Si vous avez utilisé un dataset où 'normal.' est la valeur pour la classe normale, utilisez :
# normal_attack_labels = ['normal.']
# Si vous avez un dataset où '0' est le label numérique pour normal :
# normal_attack_labels = ['0']

# Pour le moment, je vais mettre une valeur de placeholder pour que le script puisse s'exécuter,
# mais vous DEVEZ la modifier après avoir examiné la sortie 'value_counts' ci-dessus.
# Par exemple, si "21" représente la classe normale dans vos fichiers, mettez normal_attack_labels = ['21']
# Si votre output `attack_type` ne contient que des nombres (21, 18 etc.)
# Et si 21 est la plus fréquente, c'est souvent la classe normale.
normal_attack_labels = ['21'] # <--- ***Ceci est une SUPPOSITION !*** Modifiez-le si '21' n'est pas votre classe normale !

train_data['attack'] = train_data['attack_type'].apply(lambda x: 0 if x in normal_attack_labels else 1)
test_data['attack'] = test_data['attack_type'].apply(lambda x: 0 if x in normal_attack_labels else 1)

print("\nDistribution de 'normal' (0) vs 'attack' (1) dans les données d'entraînement (après mapping):")
print(train_data['attack'].value_counts())
print("\nDistribution de 'normal' (0) vs 'attack' (1) dans les données de test (après mapping):")
print(test_data['attack'].value_counts())


# Séparer les caractéristiques (X) et la cible (y)
X_train = train_data.drop(columns=['attack_type', 'attack'])
y_train = train_data['attack']
X_test = test_data.drop(columns=['attack_type', 'attack'])
y_test = test_data['attack'] # y_test est la colonne binaire 'attack'

# Identifier les colonnes numériques et catégorielles
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

print("\nValeurs uniques pour les colonnes catégorielles:")
for col in categorical_cols:
    print(f"{col}: {len(X_train[col].unique())} valeurs uniques")

print(f"\nColonnes numériques: {numerical_cols}")
print(f"Colonnes catégorielles: {categorical_cols}")

# Créer un préprocesseur qui applique OneHotEncoder aux colonnes catégorielles et StandardScaler aux numériques
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Appliquer le préprocesseur aux données d'entraînement et de test
# --- CORRECTION: Suppression de .toarray() car l'output est déjà dense ---
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Appliquer MinMaxScaler sur les données déjà traitées
scaler_X = MinMaxScaler()
X_train_processed = scaler_X.fit_transform(X_train_processed)
X_test_processed = scaler_X.transform(X_test_processed)

print(f"\nForme des données après pré-traitement (entraînement): {X_train_processed.shape}")
print(f"Forme des données après pré-traitement (test): {X_test_processed.shape}")

# --- Entraînement du Modèle ---
print("\nEntraînement du modèle Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_processed, y_train)
print("Entraînement terminé.")

# --- Évaluation du Modèle ---
print("\n--- Évaluation du Modèle ---")
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision (Accuracy): {accuracy:.4f}")

classification_rep = classification_report(y_test, y_pred)
print("\nRapport de Classification:")
print(classification_rep)

# Définir confusion_mat ici pour qu'elle soit toujours disponible
confusion_mat = confusion_matrix(y_test, y_pred)
print("\nMatrice de Confusion:")
print(confusion_mat)

# Initialiser roc_auc pour s'assurer qu'il est toujours défini
roc_auc = 0.0
fpr, tpr, thresholds = None, None, None # Initialiser fpr, tpr, thresholds

# Courbe ROC
y_proba = model.predict_proba(X_test_processed)
if y_proba.shape[1] > 1:
    y_proba = y_proba[:, 1] # Probabilités de la classe positive (attaque)

    # Assurez-vous que y_test a les deux classes pour calculer l'AUC
    if np.sum(y_test) > 0 and np.sum(y_test == 0) > 0:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title('Courbe ROC du Système de Détection d\'Intrusion')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("\nAVERTISSEMENT: La courbe ROC ne peut pas être tracée car l'une des classes (0 ou 1) est absente dans 'y_test'.")
        print("Vérifiez la distribution de 'y_train' et 'y_test' après le mapping 'normal'/'attack'.")
else:
    print("\nAVERTISSEMENT: Le modèle n'a prédit des probabilités que pour une seule classe.")
    print("Cela indique un problème avec la distribution des classes dans les données d'entraînement (y_train).")
    print("Vérifiez que 'y_train' contient bien des '0' (Normal) et des '1' (Attaque) après le mapping.")
    # Si predict_proba n'a qu'une colonne, roc_auc reste à 0.0 de l'initialisation.
    # Et la courbe ROC ne peut pas être tracée de manière significative.
    print("Le calcul de la courbe ROC et de l'AUC peut être inexact ou échouer en raison de la classification à classe unique.")


# --- Sauvegarde du modèle et des préprocesseurs ---
joblib.dump(model, 'ids_random_forest_model.pkl')
print("Modèle sauvegardé sous 'ids_random_forest_model.pkl'")

joblib.dump(preprocessor, 'ids_preprocessor.pkl')
print("Préprocesseur (ColumnTransformer) sauvegardé sous 'ids_preprocessor.pkl'")
joblib.dump(scaler_X, 'ids_minmax_scaler.pkl')
print("MinMaxScaler sauvegardé sous 'ids_minmax_scaler.pkl'")

# --- Partie Système d'Alerte par E-mail et Simulation ---

# Configuration E-mail pour les alertes (Utilisez VOTRE mot de passe d'application ici)
SENDER_EMAIL = "duvalkouatchou@gmail.com"  # VOTRE adresse Gmail
SENDER_PASSWORD = "bjob gmrs hnhp sbkk"  # VOTRE mot de passe d'application Google (16 caractères)
RECEIVER_EMAIL = "rubendeffo@gmail.com"  # L'adresse où envoyer les alertes

def send_alert_email(attack_details, current_accuracy, current_roc_auc, current_classification_rep, current_confusion_mat):
    """
    Envoie un e-mail d'alerte en cas de détection d'intrusion.
    :param attack_details: Chaîne de caractères contenant les détails de l'attaque détectée.
    :param current_accuracy: La précision globale du modèle.
    :param current_roc_auc: Le score AUC ROC du modèle.
    :param current_classification_rep: Le rapport de classification complet.
    :param current_confusion_mat: La matrice de confusion.
    """
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASSWORD)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        subject = f"🔔 ALERTE IDS : Intrusion Détectée à {current_time}!"
        
        body = f"""
Cher Administrateur,

Je suis ERATOS votre IDS.

Je viens de détecter une activité suspecte sur le réseau.

Détails de l'incident :
- **Type d'incident :** ATTENTION - INTRUSION DÉTECTÉE
- **Heure de l'attaque :** {current_time}
- **Détails additionnels :** {attack_details}

Veuillez examiner cette alerte et prendre les mesures nécessaires au plus vite, car comme le dit PARKINSON "plus on a de temps pour réaliser une tâche, plus cette tâche prend du temps."

Cordialement,

ERATOS.

--- Rapport de Performance du Modèle (Global) ---
Précision (Accuracy): {current_accuracy:.4f}
Score AUC: {current_roc_auc:.4f}

Rapport de Classification:
{current_classification_rep}
Matrice de Confusion:
{current_confusion_mat}
"""
        yag.send(to=RECEIVER_EMAIL, subject=subject, contents=body)
        print(f"🚨 Alerte par e-mail envoyée avec succès à {RECEIVER_EMAIL}.")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi de l'e-mail d'alerte (Yagmail) : {e}")
        print("Vérifiez vos identifiants Yagmail (SENDER_EMAIL, SENDER_PASSWORD), l'adresse RECEIVER_EMAIL et votre connexion internet.")

print("\n--- Détection d'Intrusion sur de Nouvelles Data (Simulation) ---")

# Chargement des objets nécessaires pour la détection en temps réel (pour s'assurer qu'ils sont disponibles)
loaded_model = joblib.load('ids_random_forest_model.pkl')
loaded_preprocessor = joblib.load('ids_preprocessor.pkl')
loaded_minmax_scaler = joblib.load('ids_minmax_scaler.pkl')

# --- SIMULATION D'UNE ATTAQUE GARANTIE POUR LE TEST DE L'EMAIL ---
simulated_attack_data_for_prediction = None
attack_details_for_email = ""

# Trouver les indices des attaques dans y_test (qui est la colonne binaire 'attack')
attack_indices_in_test = np.where(y_test == 1)[0] # Indices des vraies attaques
normal_indices_in_test = np.where(y_test == 0)[0] # Indices du trafic normal

# Nous allons prendre un échantillon d'attaque réel des données de test
# pour nous assurer que la prédiction est '1' et que l'e-mail est envoyé.
if len(attack_indices_in_test) > 0:
    # Prendre le premier échantillon d'attaque du X_test ORIGINAL
    original_attack_log_df = X_test.iloc[[attack_indices_in_test[0]]].copy()
    
    # Appliquer le préprocesseur et le scaler sur cet échantillon
    processed_for_prediction = loaded_preprocessor.transform(original_attack_log_df)
    simulated_attack_data_for_prediction = loaded_minmax_scaler.transform(processed_for_prediction)
    
    print(f"Utilisation d'un exemple d'attaque réel (indice {attack_indices_in_test[0]} de X_test original) pour la simulation.")
    
    # Pour les détails de l'e-mail, utilisons les données originales pour une meilleure lisibilité
    attack_details_for_email = f"Ligne de log suspecte (index original {X_test.index[attack_indices_in_test[0]]}):\n{original_attack_log_df.to_dict(orient='records')[0]}"
    attack_details_for_email += f"\nType d'attaque original (selon le dataset): '{test_data['attack_type'].iloc[attack_indices_in_test[0]]}'"
elif len(normal_indices_in_test) > 0:
    # Fallback: S'il n'y a pas d'attaques dans le jeu de test, mais des normales.
    # Pourrait arriver si le mapping initial était défectueux.
    print("AVERTISSEMENT: Aucune attaque (y_test == 1) n'a été trouvée dans votre jeu de test pour la simulation d'attaque.")
    print("Simulation avec un échantillon 'Normal' de votre dataset de test.")
    original_normal_log_df = X_test.iloc[[normal_indices_in_test[0]]].copy()
    processed_for_prediction = loaded_preprocessor.transform(original_normal_log_df)
    simulated_attack_data_for_prediction = loaded_minmax_scaler.transform(processed_for_prediction)
    
    attack_details_for_email = f"Ligne de log simulée (normal) (index original {X_test.index[normal_indices_in_test[0]]}):\n{original_normal_log_df.to_dict(orient='records')[0]}"
    attack_details_for_email += f"\nType original (selon le dataset): '{test_data['attack_type'].iloc[normal_indices_in_test[0]]}'"

else:
    print("AVERTISSEMENT GRAVE: Le jeu de test ne contient ni attaques ni trafic normal après le mapping.")
    print("La simulation ne peut pas être effectuée correctement.")
    exit() # Quitter car le dataset de test est inutilisable pour la simulation

# Faire la prédiction
prediction = loaded_model.predict(simulated_attack_data_for_prediction)
prediction_proba = loaded_model.predict_proba(simulated_attack_data_for_prediction)[0]

# --- Bloc de Déclenchement de l'Alerte ---
print(f"Prédiction pour la nouvelle entrée : {prediction[0]} ({'Attaque' if prediction[0] == 1 else 'Normal'})")

if prediction[0] == 1:
    print("DEBUG: Entrée dans le bloc d'envoi d'e-mail. Attaque détectée !!!")
    # Envoyer l'e-mail avec les détails de l'attaque et le rapport de métriques global
    send_alert_email(attack_details_for_email, accuracy, roc_auc, classification_rep, confusion_mat)
    print(f"Probabilité d'attaque: {prediction_proba[1]*100:.2f}%")
    print(f"Détails du log suspect envoyé par e-mail:\n{attack_details_for_email}")
else:
    print("DEBUG: Prédiction 'Normal', pas d'envoi d'e-mail pour cette simulation unique.")
    print(f"\nTrafic réseau normal.")
    # Vérifier si prediction_proba a bien deux colonnes avant d'accéder à l'index 0 ou 1
    # Pour un cas normal, predict_proba[0] est la probabilité de la classe 0 (normal)
    if len(prediction_proba) > 0:
        # Si le modèle est binaire, prediction_proba[0] est prob de normal, prediction_proba[1] est prob d'attaque
        # Si le modèle a une seule classe, prediction_proba[0] est la prob de l'unique classe.
        # Dans ce cas, nous assumons que c'est la prob de "normal" si la prediction est 0.
        prob_normal = prediction_proba[0] if prediction[0] == 0 and len(prediction_proba) > 0 else 0.0
        print(f"Probabilité de normal: {prob_normal*100:.2f}%")
    print("Note: L'alerte e-mail ne sera envoyée que si le modèle prédit 'Attaque' (1).")
