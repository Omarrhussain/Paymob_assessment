import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RFMSegmenter:
    def __init__(self, reference_date: str = "2025-06-30"):
        self.reference_date = pd.to_datetime(reference_date, utc=True)

    def calculate_rfm(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RFM metrics for each customer.
        Only successful transactions are considered for revenue/GMV-based RFM.
        """
        logger.info("Starting RFM calculation on successful transactions...")
        
        # Filter for successful transactions
        success_df = transactions_df[transactions_df['status'] == 'success'].copy()
        
        # Ensure timestamp is timezone-aware UTC
        if success_df['timestamp_utc'].dt.tz is None:
            success_df['timestamp_utc'] = success_df['timestamp_utc'].dt.tz_localize('UTC')
            
        # Group by customer and compute metrics
        # Recency: days since last success relative to reference_date
        # Frequency: count of success transactions
        # Monetary: sum of amount_egp
        rfm = success_df.groupby('customer_id').agg(
            last_date=('timestamp_utc', 'max'),
            Frequency=('transaction_id', 'count'),
            Monetary=('amount_egp', 'sum')
        ).reset_index()

        rfm['Recency'] = (self.reference_date - rfm['last_date']).dt.days
        
        # Check for any negative recency (can happen if data exceeds reference date)
        # If so, clip at 0
        rfm['Recency'] = rfm['Recency'].clip(lower=0)

        # Drop the intermediate last_date column
        rfm = rfm.drop(columns=['last_date'])
        
        logger.info(f"Calculated RFM metrics for {len(rfm)} unique customers.")
        return rfm

    def score_rfm(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns scores from 1 to 5 for Recency, Frequency, and Monetary.
        R_score: 5 is most recent (lowest Recency days), 1 is least recent.
        F_score: 5 is most frequent (highest Frequency), 1 is least frequent.
        M_score: 5 is highest spend (highest Monetary), 1 is lowest spend.
        """
        logger.info("Scoring RFM quintiles...")
        df = rfm_df.copy()

        # Recency: lower is better (use labels [5, 4, 3, 2, 1])
        # To avoid problems with duplicate bin edges, we can use rank method if needed,
        # but qcut works well for continuous days. Let's use qcut.
        df['R_score'] = pd.qcut(df['Recency'], q=5, labels=[5, 4, 3, 2, 1])

        # Frequency: higher is better. Because frequency has discrete integer counts,
        # standard qcut might fail due to identical bin edges. We can use ranking first.
        df['F_score'] = pd.qcut(df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

        # Monetary: higher is better
        df['M_score'] = pd.qcut(df['Monetary'], q=5, labels=[1, 2, 3, 4, 5])

        # Convert scores to integers for easier logic matching
        df['R_score'] = df['R_score'].astype(int)
        df['F_score'] = df['F_score'].astype(int)
        df['M_score'] = df['M_score'].astype(int)

        df['RFM_Cell'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
        df['RFM_Score'] = df[['R_score', 'F_score', 'M_score']].mean(axis=1)

        logger.info("RFM quintiles scored successfully.")
        return df

    def assign_segments(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns customers to business segments based on RFM scores:
        - Champions: R>=4, F>=4, M>=4
        - Loyal: F>=4, M>=3 (excluding Champions)
        - Potential Loyalists: R>=3, F=2-3 (excluding above)
        - At-Risk: R=2-3, F>=3, M>=3 (excluding above)
        - Hibernating: R<=2, F<=2 (excluding above)
        - Lost: R=1, F=1 (excluding above)
        - Others: any remaining customers (labeled "About to Sleep / Needs Attention")
        """
        logger.info("Assigning customer segments based on RFM scores...")
        df = scored_df.copy()
        
        segments = []
        for idx, row in df.iterrows():
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            
            # Champions
            if r >= 4 and f >= 4 and m >= 4:
                segment = "Champions"
            # Loyal
            elif f >= 4 and m >= 3:
                segment = "Loyal"
            # At-Risk
            elif r in [2, 3] and f >= 3 and m >= 3:
                segment = "At-Risk"
            # Potential Loyalists
            elif r >= 3 and f in [2, 3]:
                segment = "Potential Loyalists"
            # Lost
            elif r == 1 and f == 1:
                segment = "Lost"
            # Hibernating
            elif r <= 2 and f <= 2:
                segment = "Hibernating"
            else:
                segment = "About to Sleep / Needs Attention"
                
            segments.append(segment)
            
        df['Segment'] = segments
        logger.info(f"Segment counts:\n{df['Segment'].value_counts()}")
        return df

    def generate_and_export_at_risk(self, segmented_df: pd.DataFrame, output_path: str = "target_customers_at_risk.csv") -> pd.DataFrame:
        """
        Filters and exports only the At-Risk customer list to a CSV file.
        """
        at_risk_df = segmented_df[segmented_df['Segment'] == "At-Risk"].copy()
        logger.info(f"Filtered {len(at_risk_df)} At-Risk customers. Exporting to {output_path}...")
        
        # Reorder columns for clean presentation
        columns_to_export = [
            'customer_id', 'Recency', 'Frequency', 'Monetary',
            'R_score', 'F_score', 'M_score', 'RFM_Cell', 'Segment'
        ]
        at_risk_df = at_risk_df[columns_to_export]
        at_risk_df.to_csv(output_path, index=False)
        logger.info("Export completed.")
        return at_risk_df

if __name__ == "__main__":
    from src.data_processor import DataProcessor
    
    dp = DataProcessor()
    tx_df, _ = dp.process_all()
    
    segmenter = RFMSegmenter()
    rfm = segmenter.calculate_rfm(tx_df)
    scored = segmenter.score_rfm(rfm)
    segmented = segmenter.assign_segments(scored)
    
    at_risk = segmenter.generate_and_export_at_risk(segmented)
    print(f"Exported At-Risk count: {len(at_risk)}")
