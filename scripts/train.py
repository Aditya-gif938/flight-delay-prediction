"""
train.py
--------
End-to-end training script: collect data → engineer features → train → save.
Usage: python scripts/train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_collection      import FlightDataCollector
from feature_engineering  import FlightFeatureEngineer
from model                import FlightDelayModel


def main():
    print("=" * 55)
    print("  Flight Delay Prediction — Training Pipeline")
    print("=" * 55)

    # 1. Collect data
    collector = FlightDataCollector(n_samples=5000)
    df = collector.collect()

    # 2. Feature engineering
    print("\nEngineering features...")
    engineer = FlightFeatureEngineer()
    df = engineer.fit_transform(df)
    features = engineer.get_feature_names()
    print(f"  Features created: {len(features)}")
    print(f"  Delay rate: {df.is_delayed.mean():.1%}")

    # 3. Train models
    print("\nTraining models...")
    model = FlightDelayModel()
    metrics = model.train(df, features)

    # 4. Feature importance
    print("\nTop 10 Feature Importances:")
    fi = model.feature_importance().head(10)
    for _, row in fi.iterrows():
        bar = "█" * int(row.importance * 50)
        print(f"  {row.feature:<25} {bar} {row.importance:.3f}")

    # 5. Save
    print("\nSaving models...")
    model.save("models")

    print("\n" + "=" * 55)
    print("  Training Complete!")
    print(f"  Accuracy : {metrics['accuracy']:.1%}")
    print(f"  F1-Score : {metrics['f1']:.3f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.3f}")
    print(f"  MAE      : {metrics['mae']} min")
    print("=" * 55)
    print("\nNext steps:")
    print("  streamlit run src/dashboard.py")
    print("  uvicorn api.app:app --reload")


if __name__ == "__main__":
    main()
