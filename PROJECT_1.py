import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import warnings
import os
import joblib
from datetime import datetime
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION - CHANGE DATASET FILE NAME HERE!
# ================================================================
DATASET_FILE = "AV-GPS-Dataset-1.csv"  # ⬅️ ⬅️ ⬅️ غير هنا فقط!
# ================================================================

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GPSDetectorOptimized:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                oob_score=True
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), # طبقتين مخفيتين للتعلم العميق
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True # للتوقف فور الوصول لأفضل دقة ومنع الـ Overfitting
            ),
            
            'hist_gradient_boosting': HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=10,
                learning_rate=0.1,
                l2_regularization=0.1,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.normalizer = PowerTransformer(method='yeo-johnson')
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
        self.performance = {
            'train_accuracy': 0, 
            'test_accuracy': 0, 
            'cv_accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
        self.all_predictions = []
        self.all_confidences = []
        self.best_model_name = ""
        
    def load_and_prepare_data(self, csv_file):
        print("=" * 80)
        print("OPTIMIZED GPS SPOOFING DETECTOR")
        print("=" * 80)
        print("Using ONLY 3 powerful models:")
        print("   1. Random Forest (300 trees)")
        print("   2. Neural Network (MLP)") 
        print("   3. HistGradientBoosting (Fast)")
        print("=" * 80)
        print(f"Loading data from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
        except FileNotFoundError:
            print(f"[ERROR] File '{csv_file}' not found!")
            print(f"   Current directory files: {os.listdir('.')}")
            return None
        
        if 'Data Type' not in df.columns:
            print("[ERROR] 'Data Type' column not found!")
            return None
        
        df['label_numeric'] = df['Data Type']
        df['label_text'] = df['Data Type'].map({0: 'normal', 1: 'fake'})
        
        print(f"\nLabel Distribution:")
        normal_count = (df['label_numeric'] == 0).sum()
        fake_count = (df['label_numeric'] == 1).sum()
        print(f"   Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
        print(f"   Fake:   {fake_count} ({fake_count/len(df)*100:.1f}%)")
        
        self.create_features(df)
        self.analyze_data_quality(df)
        
        return df
    
    def create_features(self, df):
        print(f"\nCreating optimized features...")
        
        if 'Satellite Count' in df.columns:
            df['sat_count'] = df['Satellite Count']
        else:
            df['sat_count'] = 0
            
        if 'Satellite Locks' in df.columns:
            df['sat_locks'] = df['Satellite Locks']
        else:
            df['sat_locks'] = 0
        
        df['sat_ratio'] = df['sat_locks'] / (df['sat_count'] + 1e-6)
        df['sat_discrepancy'] = abs(df['sat_count'] - df['sat_locks'])
        
        if 'Velocity (m/s)' in df.columns:
            df['velocity'] = df['Velocity (m/s)']
            df['velocity_diff'] = df['velocity'].diff().abs().fillna(0)
        else:
            df['velocity'] = 0
            df['velocity_diff'] = 0
        
        if 'Heading (deg)' in df.columns and 'GPS Course' in df.columns:
            df['heading_diff'] = abs(df['Heading (deg)'] - df['GPS Course'])
            df['heading_diff'] = df['heading_diff'].apply(lambda x: 360 - x if x > 180 else x)
            df['heading_abs_diff'] = df['heading_diff']
        else:
            df['heading_diff'] = 0
            df['heading_abs_diff'] = 0
        
        window = 5
        for col in ['velocity', 'sat_count', 'sat_ratio']:
            if col in df.columns:
                df[f'{col}_mean_5'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_std_5'] = df[col].rolling(window, min_periods=1).std().fillna(0)
        
        self.feature_names = [
            'sat_count', 'sat_locks', 'sat_ratio', 'sat_discrepancy',
            'velocity', 'velocity_diff',
            'heading_abs_diff'
        ]
        
        for col in df.columns:
            if any(x in col for x in ['_mean_', '_std_']):
                self.feature_names.append(col)
        
        print(f"[OK] Created {len(self.feature_names)} optimized features")
        print(f"   Top features: {self.feature_names[:5]}...")
        
        return df
    
    def analyze_data_quality(self, df):
        print(f"\nData Quality Analysis:")
        
        missing = df[self.feature_names].isnull().sum().sum()
        if missing > 0:
            print(f"   [WARNING] Found {missing} missing values (will be imputed)")
        else:
            print(f"   [OK] No missing values")
        
        if 'sat_count' in df.columns and 'sat_locks' in df.columns:
            normal_mean = df[df['label_numeric'] == 0][['sat_count', 'sat_locks']].mean()
            fake_mean = df[df['label_numeric'] == 1][['sat_count', 'sat_locks']].mean()
            
            print(f"\nClass Separation Analysis:")
            print(f"   Satellite Count - Normal: {normal_mean['sat_count']:.1f}, Fake: {fake_mean['sat_count']:.1f}")
            print(f"   Satellite Locks - Normal: {normal_mean['sat_locks']:.1f}, Fake: {fake_mean['sat_locks']:.1f}")
            
            separation = abs(normal_mean['sat_count'] - fake_mean['sat_count'])
            if separation > 2:
                print(f"   [OK] Good separation between classes")
            else:
                print(f"   [WARNING] Weak separation - models will need to work harder")
    
    def train_optimized_models(self, df):
        print("\n" + "=" * 80)
        print("TRAINING OPTIMIZED MODELS (3 POWERFUL MODELS)")
        print("=" * 80)
        
        X = df[self.feature_names].copy()
        y = df['label_numeric'].copy()
        
        print(f"\nTraining Data Summary:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Normal: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
        print(f"   Fake: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        X_normalized = self.normalizer.fit_transform(X_scaled)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"\nData Split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing:  {X_test.shape[0]} samples")
        
        print(f"\nTraining Individual Models...")
        model_results = []
        
        for model_name, model in self.models.items():
            print(f"\n   Training {model_name.replace('_', ' ').title()}...")
            
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred) * 100
            test_acc = accuracy_score(y_test, y_test_pred) * 100
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean() * 100
            cv_std = cv_scores.std() * 100
            
            model_results.append({
                'Model': model_name,
                'Train_Acc': train_acc,
                'Test_Acc': test_acc,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std
            })
            
            print(f"      Train Accuracy: {train_acc:.2f}%")
            print(f"      Test Accuracy:  {test_acc:.2f}%")
            print(f"      CV Accuracy:    {cv_mean:.2f}% (+/-{cv_std:.2f}%)")
            
            if test_acc >= 97 and test_acc > self.performance.get('best_test_acc', 0):
                self.best_model_name = model_name
                self.performance['best_test_acc'] = test_acc
        
        print(f"\nCreating Soft Voting Ensemble...")
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            weights=[1, 1, 1],
            n_jobs=-1
        )
        
        self.ensemble_model.fit(X_train, y_train)
        y_train_ens = self.ensemble_model.predict(X_train)
        y_test_ens = self.ensemble_model.predict(X_test)
        
        self.performance['train_accuracy'] = accuracy_score(y_train, y_train_ens)
        self.performance['test_accuracy'] = accuracy_score(y_test, y_test_ens)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_ens, average='binary'
        )
        
        self.performance['precision'] = precision
        self.performance['recall'] = recall
        self.performance['f1_score'] = f1
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ensemble_cv = cross_val_score(self.ensemble_model, X_train, y_train, cv=cv, scoring='accuracy')
        self.performance['cv_accuracy'] = ensemble_cv.mean()
        
        print(f"\nENSEMBLE RESULTS (Soft Voting):")
        print(f"   Training Accuracy:   {self.performance['train_accuracy']*100:.2f}%")
        print(f"   Testing Accuracy:    {self.performance['test_accuracy']*100:.2f}%")
        print(f"   CV Accuracy:         {self.performance['cv_accuracy']*100:.2f}%")
        print(f"   Precision:           {precision*100:.2f}%")
        print(f"   Recall:              {recall*100:.2f}%")
        print(f"   F1-Score:            {f1*100:.2f}%")
        
        cm = confusion_matrix(y_test, y_test_ens)
        self.plot_confusion_matrix(cm, 'test_set')
        
        print(f"\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, y_test_ens, target_names=['Normal', 'Fake']))
        
        if self.performance['test_accuracy'] >= 0.97:
            print(f"\n[TARGET ACHIEVED] Accuracy >97%")
            print(f"   Final Accuracy: {self.performance['test_accuracy']*100:.2f}%")
        else:
            print(f"\n[TARGET NOT ACHIEVED] {self.performance['test_accuracy']*100:.2f}%")
            print(f"   Missing: {(0.97 - self.performance['test_accuracy'])*100:.2f}%")
        
        return X_normalized, y, X_train, X_test, y_train, y_test, model_results
    
    def plot_confusion_matrix(self, cm, suffix=''):
        plt.figure(figsize=(8, 6))
        labels = ['Normal', 'Fake']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        plt.title(f'Confusion Matrix - {suffix.title()}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        filename = f'plots/confusion_matrix_{suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {filename}")
    
    def visualize_model_performance(self, model_results):
        print("\n" + "-" * 50)
        print("MODEL PERFORMANCE VISUALIZATION")
        print("-" * 50)
        
        df_results = pd.DataFrame(model_results)
        df_results = df_results.sort_values('Test_Acc', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df_results))
        width = 0.25
        
        bars1 = ax.bar(x - width, df_results['Train_Acc'], width, 
                      label='Training', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, df_results['Test_Acc'], width, 
                      label='Testing', color='lightcoral', alpha=0.8)
        bars3 = ax.bar(x + width, df_results['CV_Mean'], width, 
                      label='CV Mean', color='goldenrod', alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Performance of 3 Optimized Models', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in df_results['Model']], 
                          rotation=0, fontsize=11)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=97, color='red', linestyle='--', alpha=0.7, 
                  label='Target (97%)')
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = 'plots/model_performance.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved performance plot: {filename}")
        
        print(f"\nPERFORMANCE SUMMARY TABLE:")
        print("-" * 70)
        print(f"{'Model':<20} {'Training':<10} {'Testing':<10} {'CV Mean':<10} {'CV Std':<10}")
        print("-" * 70)
        for _, row in df_results.iterrows():
            model_name = row['Model'].replace('_', ' ').title()
            print(f"{model_name:<20} {row['Train_Acc']:<9.1f}% {row['Test_Acc']:<9.1f}% "
                  f"{row['CV_Mean']:<9.1f}% {row['CV_Std']:<9.1f}%")
        print("-" * 70)
    
    def analyze_feature_importance(self, df):
        print("\n" + "-" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("-" * 50)
        
        rf_model = self.models['random_forest']
        
        if hasattr(rf_model, 'feature_importances_'):
            X = df[self.feature_names].copy()
            y = df['label_numeric'].copy()
            
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            X_normalized = self.normalizer.transform(X_scaled)
            
            rf_model.fit(X_normalized, y)
            importances = rf_model.feature_importances_
            
            feat_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances,
                'Importance_%': importances * 100
            }).sort_values('Importance_%', ascending=False)
            
            print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
            print("-" * 50)
            for idx, row in feat_imp.head(10).iterrows():
                print(f"   {row['Feature']:25}: {row['Importance_%']:.2f}%")
            
            plt.figure(figsize=(12, 6))
            top_10 = feat_imp.head(10)
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_10)))
            bars = plt.barh(top_10['Feature'], top_10['Importance_%'], color=colors)
            
            plt.xlabel('Importance (%)', fontsize=12)
            plt.title('Top 10 Most Important Features for GPS Spoofing Detection', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            filename = 'plots/feature_importance.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n[OK] Saved feature importance plot: {filename}")
            feat_imp.to_csv('plots/feature_importance.csv', index=False)
    
    def make_predictions(self, df):
        print("\n" + "=" * 80)
        print("MAKING PREDICTIONS ON ALL DATA")
        print("=" * 80)
        
        X = df[self.feature_names].copy()
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_normalized = self.normalizer.transform(X_scaled)
        
        predictions = self.ensemble_model.predict(X_normalized)
        probabilities = self.ensemble_model.predict_proba(X_normalized)
        
        pred_labels = ['normal' if p == 0 else 'fake' for p in predictions]
        
        confidences = []
        for i, pred in enumerate(predictions):
            confidences.append(probabilities[i][pred] * 100)
        
        self.all_predictions = pred_labels
        self.all_confidences = confidences
        
        true_labels = df['label_text'].values
        correct = sum(1 for i in range(len(df)) if true_labels[i] == pred_labels[i])
        overall_acc = correct / len(df) * 100
        
        print(f"\nOVERALL ACCURACY:")
        print(f"   Correct: {correct}/{len(df)}")
        print(f"   Accuracy: {overall_acc:.2f}%")
        
        cm_all = confusion_matrix(
            [0 if lbl == 'normal' else 1 for lbl in true_labels],
            [0 if lbl == 'normal' else 1 for lbl in pred_labels]
        )
        
        print(f"\nCOMPLETE DATASET CONFUSION MATRIX:")
        print(f"{'Normal (Actual)':<15} | {'Normal (Pred)':<6} {'Fake (Pred)':<6}")
        print("-" * 40)
        print(f"{'Normal':<15} | {cm_all[0,0]:<6} {cm_all[0,1]:<6}")
        print(f"{'Fake':<15} | {cm_all[1,0]:<6} {cm_all[1,1]:<6}")
        
        self.plot_confusion_matrix(cm_all, 'full_dataset')
        
        errors = []
        for i in range(len(df)):
            if true_labels[i] != pred_labels[i]:
                errors.append({
                    'index': i,
                    'true': true_labels[i],
                    'predicted': pred_labels[i],
                    'confidence': confidences[i]
                })
        
        if errors:
            print(f"\n[WARNING] Found {len(errors)} errors ({len(errors)/len(df)*100:.2f}%)")
            if len(errors) <= 10:
                print(f"   Sample errors:")
                for err in errors[:5]:
                    print(f"      Index {err['index']}: True={err['true']}, "
                          f"Pred={err['predicted']}, Conf={err['confidence']:.1f}%")
        else:
            print(f"\n[PERFECT] ACCURACY ACROSS ALL DATA!")
        
        return pred_labels, confidences
    
    def save_models_and_results(self, df):
        print("\n" + "-" * 50)
        print("SAVING MODELS AND RESULTS")
        print("-" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ensemble_path = f'models/ensemble_model_{timestamp}.pkl'
        joblib.dump(self.ensemble_model, ensemble_path)
        print(f"[OK] Ensemble model saved: {ensemble_path}")
        
        if self.best_model_name:
            best_model = self.models[self.best_model_name]
            best_path = f'models/best_{self.best_model_name}_{timestamp}.pkl'
            joblib.dump(best_model, best_path)
            print(f"[OK] Best individual model ({self.best_model_name}) saved: {best_path}")
        
        preprocessing = {
            'scaler': self.scaler,
            'normalizer': self.normalizer,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        prep_path = f'models/preprocessing_{timestamp}.pkl'
        joblib.dump(preprocessing, prep_path)
        print(f"[OK] Preprocessing objects saved: {prep_path}")
        
        results_df = df.copy()
        results_df['prediction'] = self.all_predictions
        results_df['confidence'] = self.all_confidences
        results_df['is_correct'] = results_df['label_text'] == results_df['prediction']
        
        results_path = f'gps_predictions_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"[OK] Predictions saved: {results_path}")
        
        report_path = f'model_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("GPS SPOOFING DETECTOR - MODEL REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {DATASET_FILE}\n")
            f.write(f"Total Records: {len(df)}\n\n")
            f.write("MODELS USED:\n")
            for name in self.models.keys():
                f.write(f"  • {name.replace('_', ' ').title()}\n")
            f.write(f"\nENSEMBLE PERFORMANCE:\n")
            f.write(f"  Training Accuracy:   {self.performance['train_accuracy']*100:.2f}%\n")
            f.write(f"  Testing Accuracy:    {self.performance['test_accuracy']*100:.2f}%\n")
            f.write(f"  CV Accuracy:         {self.performance['cv_accuracy']*100:.2f}%\n")
            f.write(f"  Precision:           {self.performance['precision']*100:.2f}%\n")
            f.write(f"  Recall:              {self.performance['recall']*100:.2f}%\n")
            f.write(f"  F1-Score:            {self.performance['f1_score']*100:.2f}%\n\n")
            
            if self.performance['test_accuracy'] >= 0.97:
                f.write("[TARGET ACHIEVED] Accuracy >97%\n")
            else:
                f.write("[TARGET NOT ACHIEVED]\n")
                f.write(f"   Current: {self.performance['test_accuracy']*100:.2f}%\n")
                f.write(f"   Target:  97.00%\n")
            
            f.write(f"\nTOP FEATURES ({len(self.feature_names)} total):\n")
            for i, feat in enumerate(self.feature_names[:10], 1):
                f.write(f"  {i:2}. {feat}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"[OK] Model report saved: {report_path}")
        print(f"\nAll files saved with timestamp: {timestamp}")
    
    def run_full_analysis(self, csv_file):
        print("\n" + "="*80)
        print("STARTING OPTIMIZED GPS SPOOFING DETECTOR")
        print("="*80)
        
        df = self.load_and_prepare_data(csv_file)
        if df is None:
            return
        
        X, y, X_train, X_test, y_train, y_test, model_results = self.train_optimized_models(df)
        self.visualize_model_performance(model_results)
        self.analyze_feature_importance(df)
        predictions, confidences = self.make_predictions(df)
        self.save_models_and_results(df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*80)
        print(f"Final Ensemble Accuracy: {self.performance['test_accuracy']*100:.2f}%")
        
        if self.performance['test_accuracy'] >= 0.97:
            print(f"[SUCCESS] TARGET ACHIEVED: Accuracy >97%")
        else:
            print(f"[WARNING] Target not achieved")
            print(f"   Current: {self.performance['test_accuracy']*100:.2f}%")
            print(f"   Needed:  97.00%")
        
        print(f"\nGenerated Files:")
        print(f"   plots/ - Visualizations")
        print(f"   models/ - Saved models")
        print(f"   gps_predictions_*.csv - Predictions")
        print(f"   model_report_*.txt - Detailed report")
        print("\n" + "="*80)

def main():
    # استخدام المتغير من الأعلى مباشرة
    detector = GPSDetectorOptimized()
    detector.run_full_analysis(DATASET_FILE)  # ⬅️ هنا فقط مكان التغيير

if __name__ == "__main__":
    main()