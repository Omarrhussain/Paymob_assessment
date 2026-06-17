import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.transactions_path = self.data_dir / "transactions.csv"
        self.campaign_path = self.data_dir / "campaign_responses.csv"

    def load_raw_transactions(self) -> pd.DataFrame:
        """Loads raw transactions dataset."""
        logger.info(f"Loading raw transactions from {self.transactions_path}")
        if not self.transactions_path.exists():
            raise FileNotFoundError(f"Transactions file not found at {self.transactions_path}")
        return pd.read_csv(self.transactions_path)

    def load_raw_campaign_responses(self) -> pd.DataFrame:
        """Loads raw campaign responses dataset."""
        logger.info(f"Loading raw campaign responses from {self.campaign_path}")
        if not self.campaign_path.exists():
            raise FileNotFoundError(f"Campaign file not found at {self.campaign_path}")
        return pd.read_csv(self.campaign_path)

    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the transactions DataFrame based on the data quality findings in the PRD:
        1. Normalizes payment method text (case insensitivity, spaces, grouping CODs).
        2. Drops duplicate transaction_id rows.
        3. Cleans negative amount_egp:
           - In refunded transactions, absolute value is used (amount refunded).
           - In successful transactions, rows with negative amounts are dropped as corrupt.
        4. Handles null city values by labeling them 'Unknown'.
        5. Sets failure_reason to NaN where status is not 'failed'.
        6. Converts timestamp_utc to datetime.
        """
        logger.info("Starting cleaning pipeline for transactions...")
        df = df.copy()

        # 1. Normalize payment_method
        # Expected categories: 'wallet', 'card', 'instapay', 'installment', 'cash_on_delivery'
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].astype(str).str.strip().str.lower()
            method_mapping = {
                'cod': 'cash_on_delivery',
                'cash on delivery': 'cash_on_delivery',
                'cash_on_delivery': 'cash_on_delivery',
                'wallet': 'wallet',
                'card': 'card',
                'credit card': 'card',
                'debit card': 'card',
                'instapay': 'instapay',
                'installment': 'installment'
            }
            # Fallback if any unmapped methods appear, clean them
            df['payment_method'] = df['payment_method'].map(lambda x: method_mapping.get(x, x))
            logger.info(f"Normalized payment methods. Distinct categories: {df['payment_method'].unique().tolist()}")

        # 2. Drop duplicate transaction_id
        if 'transaction_id' in df.columns:
            initial_len = len(df)
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
            duplicates_dropped = initial_len - len(df)
            logger.info(f"Dropped {duplicates_dropped} duplicate transaction_id rows. Remaining: {len(df)}")

        # 3. Clean negative amount_egp values
        if 'amount_egp' in df.columns and 'status' in df.columns:
            # For refunded status, use absolute amount (assuming they were recorded as negative during export)
            refunded_mask = df['status'] == 'refunded'
            df.loc[refunded_mask, 'amount_egp'] = df.loc[refunded_mask, 'amount_egp'].abs()
            
            # For success/failed statuses, negative amounts are data entry corruption; drop them
            corrupt_mask = (df['status'] != 'refunded') & (df['amount_egp'] < 0)
            corrupt_count = corrupt_mask.sum()
            if corrupt_count > 0:
                df = df[~corrupt_mask]
                logger.info(f"Dropped {corrupt_count} corrupt rows with negative amount_egp and non-refunded status.")

        # 4. Handle null city values
        if 'city' in df.columns:
            null_cities = df['city'].isnull().sum()
            df['city'] = df['city'].fillna("Unknown")
            logger.info(f"Labeled {null_cities} missing city values as 'Unknown'")

        # 5. Fix failure_reason mapping inconsistencies
        if 'failure_reason' in df.columns and 'status' in df.columns:
            invalid_failure_reasons = (df['failure_reason'].notnull()) & (df['status'] != 'failed')
            invalid_count = invalid_failure_reasons.sum()
            df.loc[invalid_failure_reasons, 'failure_reason'] = np.nan
            logger.info(f"Cleared failure_reason for {invalid_count} rows where status is not 'failed'.")

        # 6. Parse timestamps
        if 'timestamp_utc' in df.columns:
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
            # Add Africa/Cairo local time helper column
            df['timestamp_cairo'] = df['timestamp_utc'].dt.tz_localize('UTC').dt.tz_convert('Africa/Cairo')
            logger.info("Parsed timestamp_utc and created timezone-converted timestamp_cairo.")

        logger.info("Transactions cleaning completed successfully.")
        return df

    def get_leakage_free_transactions(self, transactions_df: pd.DataFrame, cutoff_date: str = "2025-04-15") -> pd.DataFrame:
        """
        Filters transactions to strictly include those before the campaign send date to prevent data leakage.
        By default, the cutoff is 2025-04-15 00:00:00 UTC (strictly before 2025-04-15).
        """
        logger.info(f"Applying data leakage prevention. Cutoff timestamp: {cutoff_date} UTC.")
        cutoff = pd.to_datetime(cutoff_date, utc=True)
        # Ensure the timestamp column is timezone-aware UTC for accurate comparison
        if transactions_df['timestamp_utc'].dt.tz is None:
            tx_utc = transactions_df['timestamp_utc'].dt.tz_localize('UTC')
        else:
            tx_utc = transactions_df['timestamp_utc']
            
        filtered_df = transactions_df[tx_utc < cutoff].copy()
        logger.info(f"Filtered transactions count before {cutoff_date}: {len(filtered_df)} (from {len(transactions_df)})")
        return filtered_df

    def process_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper method to run the entire loading and cleaning process.
        Returns cleaned transactions and campaign responses.
        """
        raw_tx = self.load_raw_transactions()
        cleaned_tx = self.clean_transactions(raw_tx)
        
        raw_camp = self.load_raw_campaign_responses()
        # Campaign response does not need complex cleaning, but let's check it
        cleaned_camp = raw_camp.copy()
        logger.info("Campaign response data loaded.")
        
        return cleaned_tx, cleaned_camp

if __name__ == "__main__":
    # Test script run
    processor = DataProcessor()
    tx, camp = processor.process_all()
    print(f"Transactions Shape: {tx.shape}")
    print(f"Campaign Responses Shape: {camp.shape}")
